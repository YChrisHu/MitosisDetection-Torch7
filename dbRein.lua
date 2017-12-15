----------------------------------------------------------------------
-- This file is used to create reinforced database
-- for reinforce training corase net OR 
-- for fine tunning the modified caffeNet
-- 
-- I: images (.bmp is not supported)
-- I: cordinate labels
-- O: labeled image
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'dp'
require 'optim'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'fntt'
require 'fnpp'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -n,--ncname    (default "Nc.net")       name of corase net to use
   -s,--savename  (default "trainRe")      save refined database
   -l,--loadname  (default "trainRe")      load database to enlarge
   -t,--thresh    (default 0)              threshold 0~0.5
   -c,--corase                             to reinforce corase net
   -e,--enlarge                            enlarge previous dataset
   -p,--prenorm                            conduct prefixed normalization
   -f,--flip                               get fliped images
   -i,--index     (default 1)              input image index (1-35)
]]

-- fix seed
torch.manualSeed(1)

-- use cuda
print('<torch> use cuda')
torch.setdefaulttensortype('torch.CudaTensor')

----------------------------------------------------------------------
-- load models on mitosis detection
--
local filename = paths.concat(opt.loaddir, opt.ncname)
print('<torch> loading corase net from '..filename)
modelNc = torch.load(filename)
print('<torch> done')

-- 2 classes
classes = {'NA','Mitosis'}

-- geometry: width and height of input images
geometry = {3,2086,2086}
local dim = modelNc:outside(geometry)

----------------------------------------------------------------------
-- load image
--
tmr = sys.clock() -- timer begins
torch.setdefaulttensortype('torch.FloatTensor')
dataImg = loadImages(opt.index,geometry)
data = dataImg:cuda()
torch.setdefaulttensortype('torch.CudaTensor')

-- normalize data set
if opt.prenorm then
   print('<normalize> conducting preloaded normalization')
   local infoname = paths.concat(opt.loaddir, 'mitosis.info')
   st = torch.load(infoname)
   mean = st[1]
   std = st[2]
else
   print('<normalize> conducting individual normalization')
end
data = normalize(data, mean, std)
print('<normalize> done')

-- load cordinates
labelmap,coordinate = loadLabels(opt.index,geometry)

-- timer ends
tmr = sys.clock() - tmr
print('<torch> data loading & normalizing took '..(tmr*1000)..'ms')

----------------------------------------------------------------------
-- define scale image to fit caffeNet function
-- return type: Tensor
-- 
function getScaledImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1),3,227,227)
   
   -- take subimage and scale to fit caffeNet
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1))
      local subImage = datamap[{{},
                                {listR[subNum],listR[subNum]+93},
                                {listC[subNum],listC[subNum]+93}}]   
      local sImage = image.scale(subImage, 227, 227)
      subdata[subNum] = sImage:clone()
   end
   
   return subdata
end

----------------------------------------------------------------------
-- define image to fit corase net function
-- return type: Tensor
-- 
function getImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1),3,94,94)
   
   -- take subimage
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1))
      local subImage = datamap[{{},
                                {listR[subNum],listR[subNum]+93},
                                {listC[subNum],listC[subNum]+93}}]
      local sImage = subImage  
      subdata[subNum] = sImage:clone()
   end
   
   return subdata
end

----------------------------------------------------------------------
-- define scaled image labels from CSV files
-- return type: Tensor
-- 
function verifyImageLabels(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1)):fill(1)
   
   -- take label to fit caffeNet
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1)) -- 39~54
      local subImage = datamap[{{listR[subNum]+31,listR[subNum]+62},
                                {listC[subNum]+31,listC[subNum]+62}}]   
      if subImage:sum() > 0 then
         subdata[subNum] = 2
      end
   end
   
   return subdata
end

----------------------------------------------------------------------
-- define drawing line function
-- return type: N/A
-- 
function getShiftedImages(listR,listC)
   -- init vars
   local k = 0
   local newListR = torch.Tensor(listR:size(1)*21):zero()
   local newListC = torch.Tensor(listR:size(1)*21):zero()
   
   -- draw blue boxes
   for subNum = 1,listR:size(1) do
      for movR = -16,16,8 do
         for movC = -16,16,8 do
            if listR[subNum]+movR > 0 and listR[subNum]+movR < 1994 and
               listC[subNum]+movC > 0 and listC[subNum]+movC < 1994 and
               movR*movC ~= 256 and movR*movC ~= -256 then
               k = k+1
               newListR[k] = listR[subNum] + movR
               newListC[k] = listC[subNum] + movC
            end
         end
      end                         
   end
   
   return newListR[{{1,k}}],newListC[{{1,k}}]
end

----------------------------------------------------------------------
-- define scale image to fit caffeNet function
-- return type: Tensor
-- 
function getFlipedImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1)*8,3,94,94)
   
   -- take subimage
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1))
      local subImage = datamap[{{},
                                {listR[subNum],listR[subNum]+93},
                                {listC[subNum],listC[subNum]+93}}]
      local subImage2 = image.vflip(subImage)
      
      subdata[(subNum-1)*8+1] = subImage:clone()
      subdata[(subNum-1)*8+5] = subImage2:clone()
      for ir = 1,3 do
         subdata[(subNum-1)*8+ir+1] = image.rotate(subImage, ir*math.pi/2)
         subdata[(subNum-1)*8+ir+5] = image.rotate(subImage2, ir*math.pi/2)
      end
   end
   
   return subdata
end

----------------------------------------------------------------------
-- define scale image to fit caffeNet function
-- return type: Tensor
-- 
function getFlipScaledImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1)*8,3,227,227)
   
   -- take subimage and scale to fit caffeNet
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1))
      local sImage = datamap[{{},
                              {listR[subNum],listR[subNum]+93},
                              {listC[subNum],listC[subNum]+93}}]   
      local subImage = image.scale(sImage, 227, 227)
      local subImage2 = image.vflip(subImage)
      
      subdata[(subNum-1)*8+1] = subImage:clone()
      subdata[(subNum-1)*8+5] = subImage2:clone()
      for ir = 1,3 do
         subdata[(subNum-1)*8+ir+1] = image.rotate(subImage, ir*math.pi/2)
         subdata[(subNum-1)*8+ir+5] = image.rotate(subImage2, ir*math.pi/2)
      end
   end
   
   return subdata
end

----------------------------------------------------------------------
-- Nc go!
--
modelNc:evaluate()
map = singleGenerate(data,modelNc)
print('<detecter> Nc output generated')

-- generate raw score mask
result = map[{{2},{},{}}]:clone():squeeze()
listRow,listColumn = getRClist(result,opt.thresh)
print('<detecter> '..listRow:size(1)..' candidate detected!')
listRow,listColumn = getShiftedImages(listRow,listColumn)
print('<detecter> '..listRow:size(1)..' candidate shifted!')

-- get raw/scaled images
scaleLabel = verifyImageLabels(labelmap,listRow,listColumn)
listRow = listRow[scaleLabel:eq(1)]
listColumn = listColumn[scaleLabel:eq(1)]
if listRow:nDimension() ~= 0 then
   if opt.corase then
      if opt.flip then
         scaleData = getFlipedImages(dataImg,listRow,listColumn)
      else
         scaleData = getImages(dataImg,listRow,listColumn)
      end
   else
      if opt.flip then
         scaleData = getFlipScaledImages(dataImg,listRow,listColumn)
      else
         scaleData = getScaledImages(dataImg,listRow,listColumn)
      end
   end

   print('<detecter> '..scaleData:size(1)..' false positive samples')

   -- prepare saving data
   torch.setdefaulttensortype('torch.FloatTensor')
   scaleData = scaleData:float()
   scaleLabel = torch.IntTensor(scaleData:size(1)):fill(1)
   if opt.enlarge then
      if opt.corase then
         dge = torch.load(opt.loadname..'C.t7')
      else
         dge = torch.load(opt.loadname..'F.t7')
      end
      print('<torch> merging dataset')
      dge[1] = torch.cat(dge[1],scaleData,1)
      dge[2] = torch.cat(dge[2],scaleLabel,1)
      print('<torch> '..dge[2]:size(1)..' samples in DB now')
   else
      dge = {}
      dge[1] = scaleData
      dge[2] = scaleLabel
      print('<torch> '..dge[2]:size(1)..' samples in DB now')
   end

   -- save output
   print('<torch> saving output dataset')
   if opt.corase then
      torch.save(opt.savename..'C.t7',dge)
   else
      torch.save(opt.savename..'F.t7',dge)
   end
else
   print('<detecter> 0 false positive samples!')
end

