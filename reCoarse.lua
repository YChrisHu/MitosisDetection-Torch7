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
   -i,--index     (default 9)              input image index (1-35)
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
-- varify labels
-- return type: Tensor
-- 
function verifyLabels(realR,realC,groundR,groundC)
   -- local vars
   torch.setdefaulttensortype('torch.FloatTensor')
   local labelFlags = torch.IntTensor(realR:size(1)):fill(0)
   local occupFlags = torch.IntTensor(groundR:size(1)):fill(0)
   
   -- matching
   for r_index = 1,realR:size(1) do
      -- 0 means host ungrouped
      if labelFlags[r_index] == 0 then 
         for g_index = 1,groundR:size(1) do
            -- 0 means target ungrouped
            if occupFlags[g_index] == 0 then
               -- calculate distances
               local diffR = torch.abs(realR[r_index]-groundR[g_index])
               local diffC = torch.abs(realC[r_index]-groundC[g_index])

               -- set group number
               if (diffR <= 36 and diffC <= 36) then
                  labelFlags[r_index] = 1
                  occupFlags[g_index] = 1
               end
            end
         end
      end
   end
   
   return labelFlags
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
if not listRow then
   error('<detecter> 0 true samples detected!')
end
print('<detecter> '..listRow:size(1)..' true samples detected!')

-- get ground truth mask
-- glRow,glColumn = getRClist(torch.Tensor(167,167):fill(1),opt.thresh)

----------------------------------------------------------------------
-- get TP FP FN
--
-- scaleLabel = verifyImageLabels(labelmap,listRow,listColumn)
-- gLabel = verifyImageLabels(labelmap,glRow,glColumn)
glR,glC = detectGrouping(listRow,listColumn)
ggR,ggC = detectGrouping(coordinate[{{},{2}}]:squeeze(),coordinate[{{},{1}}]:squeeze())
glcR,glcC = glR+47,glC+47
correctLabel = verifyLabels(glcR,glcC,ggR,ggC)

DP = glR:size(1)  -- detected positive
GP = ggR:size(1)  -- groundtruth positive
TP = correctLabel:sum()
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

