----------------------------------------------------------------------
-- This file is used to show performance of coarse net
-- 
-- I: images (.bmp is not supported)
-- I: cordinate labels
-- O: performance detail on screen
----------------------------------------------------------------------

require 'torch'
require 'nn'
require 'dp'
require 'optim'
require 'image'
require 'paths'
require 'cutorch'
require 'cunn'
require 'xlua'
require 'sys'
require 'fntt'
require 'fnpp'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -n,--ncname    (default "Nc.net")       name of coarse net to use
   -t,--thresh    (default 0)              threshold 0~0.5
   -p,--prenorm                            conduct prefixed normalization
   --demo                                  for demostartion output
   --test                                  for test images
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
print('<torch> loading coarse net from '..filename)
modelNc = torch.load(filename)
print('<torch> done')

-- 2 classes
classes = {'NA','Mitosis'}

-- geometry: width and height of input images
geometry = {3,2086,2086}
local dim = modelNc:outside(geometry)

----------------------------------------------------------------------
-- define scaled image labels from CSV files
-- return type: Tensor
-- 
function verifyImageLabels(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1)):fill(1)
   
   -- take label to fit caffeNet
   for subNum = 1,listR:size(1) do
      xlua.progress(subNum, listR:size(1))
      local subImage = datamap[{{listR[subNum]+39,listR[subNum]+54},
                                {listC[subNum]+39,listC[subNum]+54}}]   
      if subImage:sum() > 0 then
         subdata[subNum] = 2
      end
   end
   
   return subdata
end

----------------------------------------------------------------------
-- start evaluate
--
-- init vars
aDP = 0  -- detected positive
aGP = 0  -- groundtruth positive
aTP = 0

-- train/test
if opt.test then
   lp = {36,50}
else
   lp = {1,35}
end
   
-- loop
for oindex = lp[1],lp[2] do
   ----------------------------------------------------------------------
   -- load image
   --
   tmr = sys.clock() -- timer begins
   print('<image> loading data')
   torch.setdefaulttensortype('torch.FloatTensor')
   dataO = loadImages(oindex,geometry)
   data = dataO:cuda()
   torch.setdefaulttensortype('torch.CudaTensor')
   print('<image> done')

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
   labelmap,coordinate = loadLabels(oindex,geometry)

   -- timer ends
   tmr = sys.clock() - tmr
   print('<torch> data loading & normalizing took '..(tmr*1000)..'ms')

   ----------------------------------------------------------------------
   -- Nc go!
   --
   modelNc:evaluate()
   map = singleGenerate(data,modelNc)
   print('<detecter> Nc output generated')

   -- generate raw score mask
   result = map[{{2},{},{}}]:clone():squeeze()
   listRow,listColumn = getRClist(result,opt.thresh)
   if not listRow then
      error('<detecter> 0 true samples detected!')
   end
   print('<detecter> '..listRow:size(1)..' true samples detected!')

   ----------------------------------------------------------------------
   -- get TP FP FN
   --
   glR,glC = detectGrouping(listRow,listColumn)
   ggR,ggC = detectGrouping(coordinate[{{},{2}}]:squeeze(),coordinate[{{},{1}}]:squeeze())
   glcR,glcC = glR+47,glC+47
   correctLabel,occupFlags = verifyLabels(glcR,glcC,ggR,ggC)
   
   DP = glR:size(1) -- detected positive
   GP = ggR:size(1) -- groundtruth positive
   TP = correctLabel:sum()
   
   if opt.demo then
      -- true positive
      if TP ~= 0 then
         getBoxedImages(dataO,glR[correctLabel:eq(1)],glC[correctLabel:eq(1)],{0,1,1},94,10)
      end
      -- false positive
      if DP ~= TP then
         getBoxedImages(dataO,glR[correctLabel:eq(0)],glC[correctLabel:eq(0)],{0,1,1},94,10)
      end
      -- false negative
      if GP ~= TP then
         local tgR,tgC = ggR[occupFlags:eq(0)]-47,ggC[occupFlags:eq(0)]-47
         tgR[tgR:lt(1)] = 1
         tgC[tgC:lt(1)] = 1
         tgR[tgR:gt(1993)] = 1993
         tgC[tgC:gt(1993)] = 1993
         getBoxedImages(dataO,tgR,tgC,{0,1,1},94,10)
      end
      image.save('output/C'..oindex..'.png',(dataO*255):byte())      
   end
   
   aDP = aDP + DP
   aGP = aGP + GP
   aTP = aTP + TP
end
   
   FP = aDP-aTP
   FN = aGP-aTP
   prcision = aTP/aDP
   recall = aTP/aGP
   F1 = 2*prcision*recall/(prcision+recall)

-- define colors
cg, cc = sys.COLORS.green, sys.COLORS.cyan
cy, cw = sys.COLORS.yellow, sys.COLORS.white

--print result
print('<result> '..cg..aTP..cw..' true positive samples')
print('<result> '..cc..FP..cw..' false positive samples')
print('<result> '..cy..FN..cw..' false negative samples')
print('<result> Prcision: '..cg..string.format('%.3f%%',prcision*100))
print('<result> Recall:   '..cg..string.format('%.3f%%',recall*100))
print('<result> F1 score: '..cg..string.format('%.3f%%',F1*100))

