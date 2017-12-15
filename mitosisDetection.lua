----------------------------------------------------------------------
-- This file is used to perform mitosis detection  
-- with trained coarse net and fine net
-- 
-- I: image (.bmp is not supported)
-- O: labeled image
----------------------------------------------------------------------

ttmr = sys.clock()
require 'torch'
require 'nn'
require 'dp'
require 'optim'
require 'image'
require 'paths'
require 'cutorch'
require 'cunn'
require 'loadcaffe'
require 'ccn2'
require 'xlua'
require 'dataset'
require 'math'
require 'sys'
require 'fnpp'
require 'fntt'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--savedir   (default "logs")         subdirectory to save logs
   -l,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -d,--demo                               demo with ground truth on output
   -p,--prenorm                            conduct prefixed normalization
   -t,--threshold (default 0.5)            threshold for final confidence
   -i,--index     (default 1)              input image index
   -b,--batchSize (default 32)             batch size
]]

-- fix seed
torch.manualSeed(1)

-- use cuda
print('<torch> use cuda')
torch.setdefaulttensortype('torch.CudaTensor')

----------------------------------------------------------------------
-- load models on mitosis detection
--
local filename = paths.concat(opt.loaddir, 'Nc.net')
print('<torch> loading coarse net from '..filename)
mNc = torch.load(filename)
print('<torch> done')
filename = paths.concat(opt.loaddir, 'Nf.net')
print('<torch> loading fine net from '..filename)
mNf = torch.load(filename)
mNf:remove()
mNf:add(nn.SoftMax())
print('<torch> done')

-- 2 classes
classes = {'NA','Mitosis'}

-- geometry: width and height of input images
geometry = {3,2086,2086}
local dim = mNc:outside(geometry)

----------------------------------------------------------------------
-- load image
--
data = torch.FloatTensor(geometry)

-- load generating set -- timer begins
tmr = sys.clock()
torch.setdefaulttensortype('torch.FloatTensor')
dataO = loadImages(opt.index,geometry)
torch.setdefaulttensortype('torch.CudaTensor')
data = dataO:cuda()

-- normalize data set
if opt.prenorm then
   print('<torch> conducting preloaded normalization')
   local infoname = paths.concat(opt.savedir, 'mitosis/mitosis.info')
   st = torch.load(infoname)
   mean = st[1]
   std = st[2]
else
   print('<torch> conducting individual normalization')
end
data = normalize(data ,mean, std)
print('<torch> done')

-- timer ends
tmr = sys.clock() - tmr
print('<torch> data loading & normalizing took '..(tmr*1000)..'ms')

----------------------------------------------------------------------
-- define Nc generate function
-- return type: 3D Tensor
--
function gNc(dataset)
   -- local vars
   local time = sys.clock()
   local output = torch.Tensor(dim)

   -- test over given dataset
   print('<detecter> on original image:')
   for t = 1,1 do
      -- disp progress
      xlua.progress(t, 1)

      -- create mini batch
      local input = dataset:clone()

      -- test samples
      output = mNc:forward(input):clone()
   end

   -- timing
   time = sys.clock() - time
   print('<detecter> time to generate 1 sample = '..(time*1000)..'ms')
   
   return output
end

----------------------------------------------------------------------
-- define Nf generate function
-- return type: 2D Tensor
--
function gNf(dataset,maxSample)
   -- local vars
   local time = sys.clock()
   local output = torch.Tensor(maxSample,2)

   -- test over given dataset
   print('<detecter> on sub images:')
   for t = 1,dataset:size(1),opt.batchSize do
      -- disp progress
      xlua.progress(t, dataset:size(1))

      -- create mini batch
      local inputs = torch.Tensor(opt.batchSize,3,227,227)
      local k = 1
      for i = t,math.min(t+opt.batchSize-1,dataset:size(1)) do
         -- load new sample
         local input = dataset[i]:clone()
         inputs[k] = input
         k = k + 1
      end

      -- test samples
      output[{{t,t+opt.batchSize-1},{}}] = mNf:forward(inputs):clone()
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size(1)
   print("<detecter> time to test 1 sample = "..(time*1000)..'ms')
   
   return output[{{1,dataset:size(1)},{}}]
end

----------------------------------------------------------------------
-- define scale image to fit caffeNet function
-- return type: Tensor
-- 
function getScaledImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1),3,227,227)
   
   -- take subimage and scale to fit caffeNet
   for subNum = 1,listR:size(1) do
      local subImage = datamap[{{},
                        {listR[subNum],listR[subNum]+93},
                        {listC[subNum],listC[subNum]+93}}]   
      local sImage = image.scale(subImage, 227, 227)
      subdata[subNum] = sImage:clone()
   end
   
   return subdata
end

----------------------------------------------------------------------
-- define drawing solid box function
-- return type: 2D Tensor
-- 
function getLabelImages(result,listR,listC)
   -- init vars
   local map = torch.Tensor(167,167):zero()
   
   -- draw blue boxes
   for subNum = 1,listR:size(1) do
      -- box
      local dot = {{(listR[subNum]+11)/12},
                   {(listC[subNum]+11)/12}}
      
      -- grey
      map[dot] = result[subNum]
   end
   return map

end

----------------------------------------------------------------------
-- Nc go!
--
mNc:evaluate()
map = singleGenerate(data,mNc)
print('<detecter> Nc output generated')

-- generate raw score mask
result = map[{{2},{},{}}]:clone():squeeze()
lR,lC = getRClist(result)

-- get subimages & normalize
subData = getScaledImages(dataO,lR,lC)
if opt.prenorm then
   print('<torch> conducting preloaded normalization')
   local infoname = paths.concat(opt.savedir, 'mitosis/fine.info')
   st = torch.load(infoname)
   mean = st[1]
   std = st[2]
end
subData = normalize(subData ,mean, std)

----------------------------------------------------------------------
-- Nf go!
--
-- init var, fs is to make sure size is multiple of 32 (ccn requires)
fs = math.ceil(lR:size(1)/32)*32
mNf:evaluate()
dmap = gNf(subData,fs)
print('<detecter> Nf output generated')

-- check positive positions
dresult = dmap[{{},{2}}]:clone():squeeze()
rR = lR[dresult:gt(opt.threshold)]
rC = lC[dresult:gt(opt.threshold)]
print('<detecter> '..lR:size(1)..' mitosis are detected by Nc')
if ##rR == 0 then
   print('<detecter> 0 mitosis are addressed by Nf!!!')
   ttmr = sys.clock() - ttmr
   print('<torch> total running time: '..ttmr..'s')
   os.exit()
end
print('<detecter> '..rR:size(1)..' mitosis are addressed by Nf')

-- draw boxes
grR,grC = detectGrouping(rR,rC)
print('<detecter> '..grR:size(1)..' mitosis are detected!')
if opt.demo then
   _,coordinate = loadLabels(opt.index,geometry)
   ggR,ggC = detectGrouping(coordinate[{{},{2}}]:squeeze(),
                            coordinate[{{},{1}}]:squeeze())
   grcR,grcC = grR+47,grC+47
   correctLabel,occupFlags = verifyLabels(grcR,grcC,ggR,ggC)

   DP = grR:size(1) -- detected positive
   GP = ggR:size(1) -- groundtruth positive
   TP = correctLabel:sum()

   if TP ~= 0 then
      getBoxedImages(dataO,grR[correctLabel:eq(1)],grC[correctLabel:eq(1)],{0,0.8,0})
   end
   -- false positive
   if DP ~= TP then
      getBoxedImages(dataO,grR[correctLabel:eq(0)],grC[correctLabel:eq(0)],{0,0,0.8})
   end
   -- false negative
   if GP ~= TP then
      local tgR,tgC = ggR[occupFlags:eq(0)]-47,ggC[occupFlags:eq(0)]-47
      tgR[tgR:lt(1)] = 1
      tgC[tgC:lt(1)] = 1
      tgR[tgR:gt(1993)] = 1993
      tgC[tgC:gt(1993)] = 1993
      getBoxedImages(dataO,tgR,tgC,{1,1,0})
   end
   -- calculate P/R/F1
   FP = DP-TP
   FN = GP-TP
   prcision = TP/DP
   recall = TP/GP
   F1 = 2*prcision*recall/(prcision+recall)

   -- define colors
   cg, cc = sys.COLORS.green, sys.COLORS.cyan
   cy, cw = sys.COLORS.yellow, sys.COLORS.white

   --print result
   print('<result> '..cg..TP..cw..' true positive samples')
   print('<result> '..cc..FP..cw..' false positive samples')
   print('<result> '..cy..FN..cw..' false negative samples')
   print('<result> Prcision: '..cg..string.format('%.3f%%',prcision*100))
   print('<result> Recall:   '..cg..string.format('%.3f%%',recall*100))
   print('<result> F1 score: '..cg..string.format('%.3f%%',F1*100))
else
   getBoxedImages(dataO,grR,grC,{0,0,1})
end

-- save output
image.save('output/F'..opt.index..'.png',(dataO*255):byte())

ttmr = sys.clock() - ttmr
print('<torch> total running time: '..ttmr..'s')
