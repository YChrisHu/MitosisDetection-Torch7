----------------------------------------------------------------------
-- This file is used to define functions 
-- for input processing [loadImages] [loadLabels]
-- for output processing [getImages] [verifyImageLabels]
----------------------------------------------------------------------

require 'image'
require 'paths'
csv2tensor = require 'csv2tensor'

----------------------------------------------------------------------
-- load image
--
function loadImages(index,geo)
   print('<image> loading image '..index)
   local dataImg = torch.zeros(geo[1],geo[2],geo[3])
   local dataI = image.load('./TData/' .. index .. '.png')
   dataImg[{{},{2,geo[2]-1},{2,geo[3]-1}}] = dataI
   print('<image> done')
   return dataImg
end

----------------------------------------------------------------------
-- load cordinates
--
function loadLabels(index,geo)
   local coordinate = csv2tensor.load('./TLabel/' .. index .. '.csv')
   local labelmap = torch.IntTensor(geo[2],geo[3]):zero()
   coordinate = coordinate:view(-1,2)
   for i = 1,coordinate:size(1) do
      labelmap[{{coordinate[i][2]+1},{coordinate[i][1]+1}}] = 1
   end
   return labelmap,coordinate
end

----------------------------------------------------------------------
-- define image to fit corase net function
-- return type: Tensor
-- 
function getImages(datamap,listR,listC)
   -- local vars
   local subdata = torch.Tensor(listR:size(1),3,94,94)
   
   -- take subimage and scale to fit caffeNet
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
      xlua.progress(subNum, listR:size(1))
      local subImage = datamap[{{listR[subNum]+39,listR[subNum]+54},
                                {listC[subNum]+39,listC[subNum]+54}}]   
      if subImage:sum() > 0 then
         subdata[subNum] = 2
      end
   end
   
   return subdata
end
