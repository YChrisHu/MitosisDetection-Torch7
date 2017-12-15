----------------------------------------------------------------------
-- This file is used to modify trained CNN to FCN as the coarse net
-- with trained  and fine net
-- 
-- I: trained cnn model (nn)
-- O: casted fcn model (nn)
----------------------------------------------------------------------

require 'nn'
require 'dp'
require 'image'
require 'paths'
require 'dataset'
require 'paths'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -l,--loaddir     (default "logs")   subdirectory to load nets
   -d,--savedir     (default "logs")   subdirectory to save nets
   -n,--name        (default "mitosis.net") name of network
   -s,--save                           save trained network
   -f,--full                           use the full dataset
   -m,--norm        (default 1)        normlize 1.fixed/2.individual/3.global
   -b,--batchSize   (default 1)        batch size [it must be 1!!!]
   -t,--threads     (default 12)       number of threads
   -c,--cuda        (default "y")      use cuda? [it must be "y"]
]]

-- check input setting
if opt.cuda ~= "y" or opt.batchSize ~= 1 or (opt.norm ~= 1 and opt.norm ~= 2 and opt.norm ~= 3) then
   error('<Error> incorrect setting')
end

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.cuda == 'y' then
   -- use cuda
   print('<torch> use cuda')
   require 'cutorch'
   require 'cunn'
   torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------------------------------------------------
-- load model to modify
-- on the 2-class classification problem
--
local filename = paths.concat(opt.loaddir, opt.name)
print('<Generater> loading network from '..filename)
model = torch.load(filename)

-- keep model parameter
mp = model:parameters()

-- remove FC layers
model:remove(14)
model:remove(13)
model:remove(11)
model:remove(10)

-- put back 2 convolutional layers to replace FC layers
model:insert(nn.SpatialConvolutionMM(32, 100, 6, 6),10);
model:add(nn.SpatialConvolutionMM(100, 2, 1, 1));

-- copy back the parameters
for i = 7,10 do
   model:parameters()[i]:set(mp[i]);
end

-- geometry: width and height of input images
geometry = {2084,2084}

--add pixel-wised softmax layer
local dim = model:outside({1,3,geometry[1],geometry[2]})
model:add(nn.SpatialSoftMax())
model:add(nn.AddConstant(-0.5))
model:add(nn.ReLU())

----------------------------------------------------------------------
-- get dataset
--
if opt.full then
   nbTestingSamples = 35
else
   nbTestingSamples = 2
   print('<warning> only using 2 samples to test quickly (use flag -full to use 35 samples)')
end

-- load generating set -- timer begins
tmr = sys.clock()
ds.ascii = false
data = ds.loadDataset('database/fullimage35.t7', nbTestingSamples)
print('<Generater> data loaded')

-- normalize data set
if opt.norm == 1 then
   print('<Generater> conducting preloaded normalization')
   local infoname = paths.concat(opt.savedir, 'mitosis.info')
   st = torch.load(infoname)
   mean = st[1]
   std = st[2]
   data:normalize(mean, std)
elseif opt.norm == 2 then
   print('<Generater> conducting individual normalization')
   data:normalize(mean, std)
elseif opt.norm == 3 then
   print('<Generater> conducting global normalization')
   data:normalizeGlobal(mean, std)
end

-- timer ends
tmr = sys.clock() - tmr
print('<Generater> data loading & normalizing took ' .. tmr .. 's')

----------------------------------------------------------------------
-- define testing function
-- return type: table
--
function test(dataset)
   -- local vars
   local time = sys.clock()
   local output = torch.Tensor(dataset:size(),2,dim[3],dim[4])

   -- test over given dataset
   print('<Generater> on testing Set:')
   for t = 1,dataset:size(),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size())

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,geometry[1],geometry[2])
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size()) do
         -- load new sample
         local sample = dataset[i]
         local input = sample[1]:clone()
         inputs[k] = input
         k = k + 1
      end

      -- test samples
      output[{{t},{},{},{}}] = model:forward(inputs):clone()
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size(1)
   print('<Generater> time to generate 1 sample = ' .. (time*1000) .. 'ms')
   
   return output
end

-- set cuda again
if opt.cuda == 'y' then
   data.data = data.data:cuda()
   torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------------------------------------------------
-- and proceed!
--
model:evaluate()
map = test(data)
print('<Generater> output generated')

----------------------------------------------------------------------
-- post processing
--
print('<Generater> calculating Score Mask')

-- generate raw score mask
result = map[{{},{2},{},{}}]:clone()

-- set counter for positive
counter = result:gt(0):int():sum(3):sum(4):squeeze() -- 0.5-0.5

-- convert result to image format
result = result*510 -- 255*2

-- save output raw score mask
for index = 1,math.min(nbTestingSamples,result:size(1)) do
   image.save('output/O'..tostring(index)..'.png',result[index]:byte())
end
print('<Generater> done ')

-- save
if opt.save then
   -- create saving folder
   local filename = paths.concat(opt.savedir, 'mitosis.coarse.net')
   os.execute('mkdir -p ' .. sys.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   
   -- clear temperary info and save
   model:clearState()
   torch.save(filename, model)
end

-- print counter as reference
print(counter)
