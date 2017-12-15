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
    -d,--loaddir        (default "caffe_net")       subdirectory to load nets
    -a,--arch           (default 1)             architecture to use [1]/2/3
    -s,--save                                   save trained network
    -f,--full                                   use the full dataset
    -l,--log                                    log while training
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
   nbTrainingPatches = 12000  -- 16000
   nbTestingPatches = 4096   -- 5600
else
   nbTrainingPatches = 320
   nbTestingPatches = 96
   print('<warning> only using 320 samples to train quickly (use flag -full to use 4000 samples)')
end

-- set the database paths
ds_dir = './database'
ds_trset = paths.concat(ds_dir, 'finetrain.t7')
ds_teset = paths.concat(ds_dir, 'finetest.t7')

-- bring in database paths
require 'dataset2'

-- load training set and normalize
trainData = ds.loadTrainSet(nbTrainingPatches, geometry)
trainData:normalizeGlobal(mean, std)

-- load test set and normalize
testData = ds.loadTestSet(nbTestingPatches, geometry)
testData:normalizeGlobal(mean, std)

----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.savedir, 'trainC.log'))
testLogger = optim.Logger(paths.concat(opt.savedir, 'testC.log'))

-- training function
function train(dataset)
   -- epoch tracker
   epoch = epoch or 1

   -- local vars
   local time = sys.clock()

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   for t = 1,dataset:size(),opt.batchSize do
      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
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
         print(' - progress in batch: ' .. t .. '/' .. dataset:size())
         print(' - nb of iterations: ' .. lbfgsState.nIter)
         print(' - nb of function evalutions: ' .. lbfgsState.funcEval)

      elseif opt.optimization == 'SGD' then

         -- Perform SGD step:
         sgdState = sgdState or {
            learningRate = opt.learningRate,
            momentum = opt.momentum,
            learningRateDecay = 5e-7
         }
         optim.sgd(feval, parameters, sgdState)
      
         -- disp progress
         xlua.progress(t, dataset:size())

      else
         error('unknown optimization method')
      end
   end
   
   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
   confusion:zero()

   -- next epoch
   epoch = epoch + 1
end

-- test function
function test(dataset)
   -- local vars
   local time = sys.clock()

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local targets = torch.Tensor(opt.batchSize)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         local _,target = sample[2]:clone():max(1)
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
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   confusion:zero()
end

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
