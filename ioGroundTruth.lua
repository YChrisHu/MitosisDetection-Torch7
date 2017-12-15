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
require 'paths'
require 'xlua'
require 'sys'
require 'fnpp'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -n,--ncname    (default "Nc.net")       name of coarse net to use
   -s,--savename  (default "trainRe")      save refined database
   -l,--loadname  (default "trainRe")      load database to enlarge
   -t,--thresh    (default 0)              threshold 0~0.5
   -c,--coarse                             to reinforce coarse net
   -p,--prenorm                            conduct prefixed normalization
   -i,--index     (default 9)              input image index (1-35)
]]

-- fix seed
torch.manualSeed(1)

-- geometry: width and height of input images
geometry = {3,2086,2086}

----------------------------------------------------------------------
-- function load image
--
function loadGround(index,geo)
   print('<image> loading image '..index)
   local dataImg = torch.zeros(geo[1],geo[2],geo[3])
   local dataI = image.load('./TGround/' .. index .. '.png')
   dataImg[{{},{2,geo[2]-1},{2,geo[3]-1}}] = dataI
   print('<image> done')
   return dataImg
end

----------------------------------------------------------------------
-- load image
--
tmr = sys.clock() -- timer begins
torch.setdefaulttensortype('torch.FloatTensor')
data = loadGround(opt.index,geometry)

-- load cordinates
labelmap,coordinate = loadLabels(opt.index,geometry)

-- timer ends
tmr = sys.clock() - tmr
print('<torch> data loading & normalizing took '..(tmr*1000)..'ms')

----------------------------------------------------------------------
-- define drawing line function
-- return type: N/A
-- 
function getBoxedImages(datamap,listR,listC)
   -- set color here: R/G/B
   local cbit = {0,0,1}

   -- draw blue boxes
   for subNum = 1,listR:size(1) do
      -- box
      local box = {}
      -- top
      box[1] = {{listR[subNum]   ,listR[subNum]+2 },
                {listC[subNum]   ,listC[subNum]+93}}
      -- bottom
      box[2] = {{listR[subNum]+91,listR[subNum]+93},
                {listC[subNum]   ,listC[subNum]+93}}
      -- left
      box[3] = {{listR[subNum]   ,listR[subNum]+93},
                {listC[subNum]   ,listC[subNum]+2 }}
      -- right
      box[4] = {{listR[subNum]   ,listR[subNum]+93},
                {listC[subNum]+91,listC[subNum]+93}}
                    
      for movR = -16,16,8 do
         for movC = -16,16,8 do
            if torch.abs(movR*movC)~=256 and listR[subNum]+movR > 0 and
               listR[subNum]+movR < 1994 and listC[subNum]+movC > 0 and
               listC[subNum]+movC < 1994 then
               local box2 = boxShift(box,movR,movC)
               for ic = 1,3 do
                  for ib = 1,4 do
                     datamap[ic][box2[ib]] = cbit[ic]
                  end
               end   
            end
         end
      end                         
   end
   
   return datamap
end

----------------------------------------------------------------------
-- define drawing line function
-- return type: N/A
-- 
function boxShift(box,modifierR,modifierC)
   local line = {{{},{}},{{},{}},{{},{}},{{},{}}}
   for ib = 1,4 do
      line[ib][1][1] = box[ib][1][1] + modifierR
      line[ib][1][2] = box[ib][1][2] + modifierR
      line[ib][2][1] = box[ib][2][1] + modifierC
      line[ib][2][2] = box[ib][2][2] + modifierC
   end
   return line
end
----------------------------------------------------------------------
-- get ground truth mask
--
ggR,ggC = detectGrouping(coordinate[{{},{2}}]:squeeze(),coordinate[{{},{1}}]:squeeze())
ggR = ggR - 47
ggC = ggC - 47
print('<torch> '..ggR:size(1)..' mitosis in image')
torch.setdefaulttensortype('torch.FloatTensor')
data = getBoxedImages(data,ggR,ggC)

-- save output
image.save('output/G'..opt.index..'.png',(data*255):byte())
