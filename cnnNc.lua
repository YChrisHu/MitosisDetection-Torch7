----------------------------------------------------------------------
-- This file is used to train coarse net
-- 
-- I: mitosis database 94x94
-- O: trained cnn model (nn)
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'dp'
require 'optim'
require 'paths'
require 'dataset'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--savedir       (default "logs")      subdirectory to save logs
   -s,--save                                save trained network
   -n,--network       (default "")          reload pretrained network
   -f,--full                                use the full dataset
   -l,--log                                 log while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.01)        learning rate, for SGD only
   -b,--batchSize     (default 50)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   -e,--epoch         (default 0)           maximum training epoches, 0 for infinite epoches
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 6)           number of threads
   -c,--cuda          (default "y")         use cuda? [y]/n
]]

-- fix seed
torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.cuda == 'y' then
   -- use cuda
   print('<torch> use cuda')
   require 'cutorch'
   require 'cunn'
   torch.setdefaulttensortype('torch.CudaTensor')
elseif opt.optimization == 'SGD' then
   -- use floats, for SGD
   torch.setdefaulttensortype('torch.FloatTensor')
end

-- batch size?
if opt.optimization == 'LBFGS' and opt.batchSize < 100 then
   error('LBFGS should not be used with small mini-batches; 1000 is recommended')
end

----------------------------------------------------------------------
-- define model to train
-- on the 2-class classification problem
--
classes = {'NA','Mitosis'}

-- geometry: width and height of input images
geometry = {94,94}

if opt.network == '' then
   -- define model to train
   model = nn.Sequential()
   ------------------------------------------------------------
   -- convolutional network 
   -- SpatialConvolutionMM could be changed to SpatialConvolution regarding on CUDA speed
   ------------------------------------------------------------
   -- stage 1 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(3, 32, 5, 5))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(3, 3, 3, 3))
   -- stage 2 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(32, 32, 3, 3))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- stage 3 : mean suppresion -> filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolutionMM(32, 32, 3, 3))
   model:add(nn.ReLU())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- stage 4 : standard 2-layer MLP:
   model:add(nn.Collapse(3))
   model:add(nn.Linear(32*6*6, 100))
   model:add(nn.ReLU())
   model:add(nn.Linear(100, #classes))
   ------------------------------------------------------------
else
   print('<trainer> reloading previously trained network')
   model = torch.load(opt.network)
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<dataset> using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches = 11700
   nbTestingPatches = 1000
else
   nbTrainingPatches = 200
   nbTestingPatches = 20
   print('<warning> only using 200 samples to train quickly (use flag -full to use 11700 samples)')
end

-- initial st to store mean std info
st = torch.FloatTensor(4)

-- create training set and normalize
trainData = ds.loadTrainSet(nbTrainingPatches, geometry)
st[1],st[2] = trainData:normalizeGlobal(mean, std)

-- create test set and normalize
testData = ds.loadTestSet(nbTestingPatches, geometry)
st[3],st[4] = testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.savedir, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.savedir, 'test.log'))

require 'fntt'

if opt.cuda == 'y' then
   model = model:cuda()
   trainData.data = trainData.data:cuda()
   testData.data = testData.data:cuda()
end

----------------------------------------------------------------------
-- and train!
--
iter = 1
while opt.epoch == 0 or iter <= opt.epoch do -- better not infinite loop here
   -- train/test
   train(trainData)
   test(testData)

   -- log errors
   if opt.log then
      trainLogger:style{['% mean class accuracy (train set)'] = '-'}
      testLogger:style{['% mean class accuracy (test set)'] = '-'}
   end
   iter = iter+1
end

if opt.save then
   -- create saving folder
   local filename = paths.concat(opt.savedir, 'mitosis.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   
   -- clear temperary info and save
   model:clearState()
   torch.save(filename, model)
   
   -- save statistics info
   local filename2 = paths.concat(opt.savedir, 'mitosis.info')
   torch.save(filename2,st)
end
