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
require 'fnpp'
require 'fnio'
require 'fntt'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--savedir     (default "logs")   subdirectory to load/save nets
   -n,--name        (default "Nc.net") trained network name
   -s,--save                           save trained network
   -f,--full                           use the full dataset
   -p,--prenorm                        conduct prefixed normalization
   -m,--norm        (default 1)        normlize 1.fixed/2.individual/3.global
   -b,--batchSize   (default 1)        batch size [it must be 1!!!]
   -t,--threads     (default 6)        number of threads
   -c,--cuda        (default "y")      use cuda? [it must be "y"]
   -i,--index       (default 1)        input image index (1-35)
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
local filename = paths.concat(opt.savedir,'mitosis', opt.name)
print('<Generater> loading network from '..filename)
model = torch.load(filename)

-- geometry: width and height of input images
geometry = {3,2086,2086}
local dim = model:outside(geometry)

----------------------------------------------------------------------
-- load image
--
tmr = sys.clock() -- timer begins
print('<image> loading data')
torch.setdefaulttensortype('torch.FloatTensor')
data = loadImages(opt.index,geometry)
data = data:cuda()
torch.setdefaulttensortype('torch.CudaTensor')
print('<image> done')

-- normalize data set
if opt.prenorm then
   print('<normalize> conducting preloaded normalization')
   local infoname = paths.concat(opt.savedir, 'mitosis.info')
   st = torch.load(infoname)
   mean = st[1]
   std = st[2]
else
   print('<normalize> conducting individual normalization')
end
data = normalize(data, mean, std)
print('<normalize> done')

-- timer ends
tmr = sys.clock() - tmr
print('<Generater> data loading & normalizing took ' .. tmr .. 's')

-- set cuda again
if opt.cuda == 'y' then
   data = data:cuda()
   torch.setdefaulttensortype('torch.CudaTensor')
end

----------------------------------------------------------------------
-- and proceed!
--
model:evaluate()
map = singleGenerate(data,model)
print('<Generater> output generated')

----------------------------------------------------------------------
-- post processing
--
print('<Generater> calculating Score Mask')

-- generate raw score mask
result = map[{{2},{},{}}]:clone()

-- set counter for positive
counter = result:gt(0):int():sum() -- 0.5-0.5

-- convert result to image format
resultb = result:gt(0):int()*255

-- save output raw score mask
image.save('output/scoremask'..opt.index..'.png',resultb:byte())

-- get grouped mask
local lR,lC = getRClist(result:squeeze(),nil,1)
local glR,glC = detectGrouping(lR,lC,3)
local remap = torch.zeros(167,167)
for il = 1,glR:size(1) do
   remap[{{glR[il]},{glC[il]}}] = 255
end

-- save output grouped score mask
image.save('output/scoremaskg'..opt.index..'.png',remap:byte())
print('<Generater> done ')

-- print counter as reference
print(counter)
