----------------------------------------------------------------------
-- This file is used to fine tune trained
-- caffeNet on mitosis detection
-- 
-- I: caffeNet model
-- O: trained cnn model (nn & ccn)
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'loadcaffe'
require 'ccn2'
require 'paths'
require 'optim'
require 'xlua'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
    -g,--savedir        (default "logs/caffe")  subdirectory to save logs
    -d,--loaddir        (default "caffe_net")   subdirectory to load nets
    -a,--arch           (default 1)             architecture to use [1]/2/3
    -s,--save                                   save trained network
    -f,--full                                   use the full dataset
    -l,--log                                    log while training
    -z,--zsave                                  save during training
    -o,--optimization   (default "SGD")         optimization: SGD | LBFGS 
    -r,--learningRate   (default 0.01)          learning rate, for SGD only
    -b,--batchSize      (default 32)            batch size, MUST be multiple of 32
    -m,--momentum       (default 0)             momentum, for SGD only
    -i,--maxIter        (default 3)             maximum nb of iterations per batch, for LBFGS
    -e,--epoch          (default 0)             maximum training epoches, 0 for infinite epoches
    --coefL1            (default 0)             L1 penalty on the weights
    --coefL2            (default 0)             L2 penalty on the weights
    -t,--threads        (default 6)             number of threads
    -c,--cuda           (default "y")           use cuda? [y]must use cuda
]]

-- geometry: width and height of input images
geometry = {227,227}

-- 2 classes
classes = {'NA','Mitosis'}

if opt.arch == 1 then
   -- use architecture 1024-512-2
   FC6 = 1024
   FC7 = 512
elseif opt.arch == 2 then
   -- use architecture 1024-256-2
   FC6 = 1024
   FC7 = 256
elseif opt.arch == 3 then
   -- use architecture 512-256-2
   FC6 = 512
   FC7 = 256
else
   error('Unknown architecture!')
end

----------------------------------------------------------------------
-- load model to perform transfer learning
--
path_deploy = paths.concat(opt.loaddir, 'deploy.prototxt')
path_model = paths.concat(opt.loaddir, 'bvlc_reference_caffenet.caffemodel')
model = loadcaffe.load(path_deploy, path_model, 'ccn2')
torch.setdefaulttensortype('torch.CudaTensor')

-- remove 3 FC layers
model:remove(25)
model:remove(22)
model:remove(19)

-- replace 3 FC layers
model:insert(nn.Linear(9216,FC6),19)
model:insert(nn.Linear(FC6,FC7),22)
model:insert(nn.Linear(FC7,2),25)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--
model:remove(26)
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()
model:clearState()

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('<dataset> using model:')
print(model)

----------------------------------------------------------------------
-- get/create dataset
--
if opt.full then
   nbTrainingPatches1 = 4000 --21623+4000
   nbTrainingPatches2 = 6368
   nbTestingPatches = 384
else
   nbTrainingPatches1 = 192
   nbTrainingPatches2 = 192
   nbTestingPatches = 32
   print('<warning> only using 192 samples to train quickly (use flag -full to use 4000 samples)')
end

-- initial st to store mean std info
st = torch.FloatTensor(4)

-- set the database paths
ds_dir = './database'
ds_trset = paths.concat(ds_dir, 'trainF.t7')
ds_teset = paths.concat(ds_dir, 'testF.t7')

-- bring in database paths
require 'dataset'

-- load training set and normalize
trainData = ds.loadTrainSet(nbTrainingPatches, geometry)
st[1],st[2] = trainData:normalizeGlobal(mean, std)

-- load test set and normalize
testData = ds.loadTestSet(nbTestingPatches, geometry)
st[3],st[4] = testData:normalizeGlobal(mean, std)

-- setup reinforce training dataset parameters
ds.path_trainset = paths.concat('./database/', 'trainReCF.t7')
ds.ascii = true

-- create training set and normalize (reinforce)
trainData2 = ds.loadTrainSet(nbTrainingPatches2, geometry)
trainData2:normalizeGlobal(st[1],st[2])
trainData.data = torch.cat(trainData.data,trainData2.data,1)
trainData.labels = torch.cat(trainData.labels,trainData2.labels,1)
function trainData:size()
   return trainData.data:size(1)
end
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.savedir, 'trainC.log'))
testLogger = optim.Logger(paths.concat(opt.savedir, 'testC.log'))

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
   if opt.zsave and (iter-math.floor(iter/25)*25 == 0) then
      local tempname = paths.concat(opt.savedir, 'mitosis.fine.mid'..iter)
      print('<trainer> saving network to '..tempname)
      torch.save(tempname, model)
   end
end

----------------------------------------------------------------------
-- save the trained network
--
if opt.save then
   local filename = paths.concat(opt.savedir, 'mitosis.fine.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   model:clearState()
   torch.save(filename, model)
end
