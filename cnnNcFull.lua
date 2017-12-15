----------------------------------------------------------------------
-- This file is used to train coarse net with reinforced database
-- 
-- I: mitosis database 94x94
-- O: trained cnn model (nn)
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
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
   -l,--log                                 log while training
   -o,--optimization  (default "SGD")       optimization: SGD | LBFGS 
   -r,--learningRate  (default 0.01)        learning rate, for SGD only
   -b,--batchSize     (default 64)          batch size
   -m,--momentum      (default 0)           momentum, for SGD only
   -i,--maxIter       (default 3)           maximum nb of iterations per batch, for LBFGS
   -e,--epoch         (default 0)           maximum training epoches, 0 for infinite epoches
   --coefL1           (default 0)           L1 penalty on the weights
   --coefL2           (default 0)           L2 penalty on the weights
   -t,--threads       (default 12)          number of threads
   -c,--cuda          (default "y")         use cuda? [y]/n
]]

-- fix seed
-- torch.manualSeed(1)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to '..torch.getnumthreads())
print('<torch> use cuda')
torch.setdefaulttensortype('torch.CudaTensor')

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
print('<torch> using model:')
print(model)

-- loss function: negative log-likelihood
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

----------------------------------------------------------------------
-- get/create dataset
--
nbTrainingPatches = 983040
nbTestingPatches = 27648

-- initial st to store mean std info
st = torch.FloatTensor(4):fill(1)

-- create training set and normalize
ds.ascii = false
ds.path_trainset = paths.concat('./database/', 'trainAllC.t7')
print('<dataset> loading training dataset')
trainData = ds.loadTrainSet(nbTrainingPatches1, geometry)
print('<dataset> conducting normalization')
st[1],st[2] = trainData:normalizeGlobal(mean, std)
print('<dataset> done')

-- create test set and normalize
ds.path_testset = paths.concat('./database/', 'testFullC.t7')
print('<dataset> loading testing dataset')
testData = ds.loadTestSet(nbTestingPatches, geometry)
print('<dataset> conducting normalization')
st[3],st[4] = testData:normalizeGlobal(mean, std)
print('<dataset> done')

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
if opt.log then
   trainLogger = optim.Logger(paths.concat(opt.savedir, 'train.log'))
   testLogger = optim.Logger(paths.concat(opt.savedir, 'test.log'))
   trainLogger:style{['% mean class accuracy (train set)'] = '-'}
   testLogger:style{['% mean class accuracy (test set)'] = '-'}
end

if opt.save then
   -- save statistics info
   local filename2 = paths.concat(opt.savedir, 'coarse.info')
   torch.save(filename2,st)
end

----------------------------------------------------------------------
-- define training function
-- return N/A
-- 
function train(data,label,epoch,iter)
   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # "..epoch..' in iteration '..iter..' [batchSize = '..opt.batchSize..']')
   for t = 1,data:size(1),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,data:size(1)) do
         -- load new sample
         local input = data[i]:clone()
         local _,target = label[i]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- just in case:
         collectgarbage()

         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- evaluate function for complete mini batch
         local outputs = model:forward(inputs)
         local f = criterion:forward(outputs, targets)

         -- estimate df/dW
         local df_do = criterion:backward(outputs, targets)
         model:backward(inputs, df_do)

         -- penalties (L1 and L2):
         if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
            -- locals:
            local norm,sign= torch.norm,torch.sign

            -- Loss:
            f = f + opt.coefL1 * norm(parameters,1)
            f = f + opt.coefL2 * norm(parameters,2)^2/2

            -- Gradients:
            gradParameters:add( sign(parameters):mul(opt.coefL1) + parameters:clone():mul(opt.coefL2) )
         end

         -- update confusion
         for i = 1,opt.batchSize do
            confusion:add(outputs[i], targets[i])
         end

         -- return f and df/dX
         return f,gradParameters
      end

      -- optimize on current mini-batch
      if opt.optimization == 'LBFGS' then

         -- Perform LBFGS step:
         lbfgsState = lbfgsState or {
            maxIter = opt.maxIter,
            lineSearch = optim.lswolfe
         }
         optim.lbfgs(feval, parameters, lbfgsState)
       
         -- disp report:
         print('LBFGS step')
         print(' - progress in batch: '..t..'/'..data:size(1))
         print(' - nb of iterations: '..lbfgsState.nIter)
         print(' - nb of function evalutions: '..lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, data:size(1))

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   print("<trainer> time to learn 1 epoch = "..time..'s')

   -- print confusion matrix
   print(confusion)
   if opt.log then
      trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   end
   confusion:zero()
end

----------------------------------------------------------------------
-- define testing function
-- return N/A
-- 
function test(data,label)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<tester> on testing Set:')
   for t = 1,data:size(1),opt.batchSize do
      -- disp progress
      xlua.progress(t, data:size(1))

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,data:size(1)) do
         -- load new sample
         local input = data[i]:clone()
         local _,target = label[i]:clone():max(1)
         target = target:squeeze()
         inputs[k] = input
         targets[k] = target
         k = k + 1
      end

      -- test samples
      local preds = model:forward(inputs)

      -- confusion:
      for i = 1,opt.batchSize do
      confusion:add(preds[i], targets[i])
      end
   end

   -- timing
   time = sys.clock() - time
   print("<tester> time to test 1 epoch = "..time..'s')

   -- print confusion matrix
   print(confusion)
   if opt.log then
      testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   end
   confusion:zero()
end

----------------------------------------------------------------------
-- cuda setting and train!
--

print('<torch> data rearrange')
-- change data format and ready for shuffle and batch
trData = trainData.data
teData = testData.data
tmpLabel1 = trainData.labels:clone()
tmpLabel2 = trainData.labels:clone()
tmpLabel1[tmpLabel1:eq(2)] = 0
tmpLabel2[tmpLabel2:eq(1)] = 0
tmpLabel2[tmpLabel2:eq(2)] = 1
trLabel = torch.cat(tmpLabel1,tmpLabel2,2)
tmpLabel1 = testData.labels:clone()
tmpLabel2 = testData.labels:clone()
tmpLabel1[tmpLabel1:eq(2)] = 0
tmpLabel2[tmpLabel2:eq(1)] = 0
tmpLabel2[tmpLabel2:eq(2)] = 1
teLabel = torch.cat(tmpLabel1,tmpLabel2,2)

-- release memory
tmpLabel1 = nil
tmpLabel2 = nil
trainData = nil
testData = nil
collectgarbage()

-- use cuda
print('<torch> casting model')
model = model:cuda()

-- start training
iter = 1
while opt.epoch == 0 or iter <= opt.epoch do -- better not infinite loop here
   torch.setdefaulttensortype('torch.FloatTensor')
   local shuffle = torch.randperm(nbTrainingPatches)
   torch.setdefaulttensortype('torch.CudaTensor')
   
   -- train
   for epoch = 1,20 do -- 20*49152
      local cuData = trData:index(1,shuffle[{{epoch*49152-49151,epoch*49152}}]:long()):cuda()
      local cuLabel = trLabel:index(1,shuffle[{{epoch*49152-49151,epoch*49152}}]:long()):clone()
      train(cuData,cuLabel,epoch,iter)
      collectgarbage()
   end
   
   -- validate
   local cuData = teData:cuda()
   local cuLabel = teLabel:clone()
   test(cuData,cuLabel)
   cuData = nil
   cuLabel = nil
   collectgarbage()
   
   -- save mid result
   local filename = paths.concat(opt.savedir, 'coarse.iter'..iter..'.t7')
   torch.save(filename, model)
   iter = iter+1
end

----------------------------------------------------------------------
-- save model
--
if opt.save then
   -- create saving folder
   local filename = paths.concat(opt.savedir, 'coarse.net')
   os.execute('mkdir -p '..sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv '..filename..' '..filename..'.old')
   end
   print('<trainer> saving network to '..filename)
   
   -- clear temperary info and save
   model:clearState()
   torch.save(filename, model)  
end
