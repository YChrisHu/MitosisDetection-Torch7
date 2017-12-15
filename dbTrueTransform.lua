----------------------------------------------------------------------
-- This file is used to create reinforced database
-- for reinforce training coarse net OR 
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
require 'image'
require 'math'
require 'paths'
require 'xlua'
require 'sys'
require 'fnpp'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -s,--savename  (default "trainFG")      save refined database
   -l,--loadname  (default "trainFG")      load database to enlarge
   -e,--enlarge                            enlarge previous dataset
   -i,--index     (default 1)              input image index (1-35)
]]

-- fix seed
torch.manualSeed(1)

-- geometry: width and height of input images
geometry = {3,2086,2086}

----------------------------------------------------------------------
-- define drawing line function
-- return type: N/A
-- 
function getBoxedImages(listR,listC)
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
function getSelectedImages(datamap,listR,listC)
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
-- load image
--
tmr = sys.clock() -- timer begins
torch.setdefaulttensortype('torch.FloatTensor')
data = loadImages(opt.index,geometry)

-- load cordinates
labelmap,coordinate = loadLabels(opt.index,geometry)

-- timer ends
tmr = sys.clock() - tmr
print('<torch> data loading & normalizing took '..(tmr*1000)..'ms')

----------------------------------------------------------------------
-- get ground truth mask
--
ggR,ggC = detectGrouping(coordinate[{{},{2}}]:squeeze(),coordinate[{{},{1}}]:squeeze())
ggR = torch.round(ggR - 47)
ggC = torch.round(ggC - 47)
print('<torch> '..ggR:size(1)..' mitosis in image')

-- get shifted image locations
ngR,ngC = getBoxedImages(ggR,ggC)
torch.setdefaulttensortype('torch.FloatTensor')
newData = getSelectedImages(data,ngR,ngC)

newData = newData:float()
newLabel = torch.IntTensor(ngR:size(1)*8):fill(2)

if opt.enlarge then
   dge = torch.load(opt.loadname .. 'C.t7')
   print('<torch> merging dataset')
   dge[1] = torch.cat(dge[1],newData,1)
   dge[2] = torch.cat(dge[2],newLabel,1)
   print('<torch> ' .. dge[2]:size(1) .. ' samples in DB now')
else
   dge = {}
   dge[1] = newData
   dge[2] = newLabel
   print('<torch> ' .. dge[2]:size(1) .. ' samples in DB now')
end

-- save output
print('<torch> saving output dataset')
torch.save(opt.savename .. 'C.t7',dge)
