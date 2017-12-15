----------------------------------------------------------------------
-- This file is used to train fine net with reinforced database
-- 
-- I: mitosis database 94x94
-- O: trained cnn model (nn)
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'loadcaffe'
require 'ccn2'
require 'optim'
require 'paths'
require 'dataset'
require 'fnpp'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--loaddir       (default "caffe_net") subdirectory to load model
   -v,--savedir       (default "logs")      subdirectory to save logs
   -s,--save                                save trained network
   -l,--log                                 log while training
   -a,--arch          (default 1)           architecture to use [1]/2/3
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

-- geometry: width and height of input images
geometry = {227,227}

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
print('<torch> using model:')
print(model)

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

-- init vats
nbTrPatches = 81920
local infoname = paths.concat(opt.savedir, 'mitosis/mitosis.info')
st = torch.load(infoname)
mean = st[1]
std = st[2]
torch.setdefaulttensortype('torch.FloatTensor')
local shuffleT = torch.randperm(27889):long()
torch.setdefaulttensortype('torch.CudaTensor')

-- use cuda
print('<torch> casting model')
model = model:cuda()

-- start training
iter = 1
while opt.epoch == 0 or iter <= opt.epoch do -- better not infinite loop here
   for dbIndex = 1,13 do --13
      print('<dataset> loading training dataset part '..dbIndex)
      local dbPath = paths.concat('./database/partial', dbIndex..'CF.t7')
      local dataBase = torch.load(dbPath)
      print('<dataset> done')
      
      local trData = normalize(dataBase[1], mean, std)
      local tmpLabel1 = dataBase[2]:clone()
      local tmpLabel2 = dataBase[2]:clone()
      tmpLabel1[tmpLabel1:eq(2)] = 0
      tmpLabel2[tmpLabel2:eq(1)] = 0
      tmpLabel2[tmpLabel2:eq(2)] = 1
      local trLabel = torch.cat(tmpLabel1,tmpLabel2,2)
      
      -- release memory
      dataBase = nil
      collectgarbage()
      
      torch.setdefaulttensortype('torch.FloatTensor')
      local shuffle = torch.randperm(nbTrPatches):long()
      torch.setdefaulttensortype('torch.CudaTensor')
      
      -- train
      for epoch = 1,8 do -- 8*10240
         local cuData = trData:index(1,shuffle[{{epoch*10240-10239,epoch*10240}}]):cuda()
         local cuLabel = trLabel:index(1,shuffle[{{epoch*10240-10239,epoch*10240}}]):clone()
         local counter = dbIndex*8-8+epoch
         train(cuData,cuLabel,counter,iter)
         collectgarbage()
      end
   end
   
   -- validate data arrange
   collectgarbage()
   print('<dataset> loading training dataset part testing')
   local dbTPath = paths.concat('./database/partial', 'testCF.t7')
   local dataTBase = torch.load(dbTPath)
   local teData = normalize(dataTBase[1], mean, std)
   tmpLabel1 = dataTBase[2]:clone()
   tmpLabel2 = dataTBase[2]:clone()
   tmpLabel1[tmpLabel1:eq(2)] = 0
   tmpLabel2[tmpLabel2:eq(1)] = 0
   tmpLabel2[tmpLabel2:eq(2)] = 1
   local teLabel = torch.cat(tmpLabel1,tmpLabel2,2)
   print('<dataset> done')
   local cuData = teData:index(1,shuffleT[{{1,15360}}]):cuda()
   local cuLabel = teLabel:clone()
   
   -- release memory
   dataTBase = nil
   teData = nil
   teLabel = nil
   collectgarbage()
   
   -- validating
   test(cuData,cuLabel)
   
   -- release memory
   cuData = nil
   cuLabel = nil
   collectgarbage()
   
   -- save mid result
   local filename = paths.concat(opt.savedir, 'fine.iter'..iter..'.t7')
   torch.save(filename, model)
   iter = iter+1
end

----------------------------------------------------------------------
-- save model
--
if opt.save then
   -- create saving folder
   local filename = paths.concat(opt.savedir, 'fine.net')
   os.execute('mkdir -p '..sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv '..filename..' '..filename..'.old')
   end
   print('<trainer> saving network to '..filename)
   
   -- clear temperary info and save
   model:clearState()
   torch.save(filename, model)  
end
