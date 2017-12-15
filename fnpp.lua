----------------------------------------------------------------------
-- This file is used to define functions 
-- for preprocessing [normalize]
-- for postprocessing [getRClist] [detectGrouping]
----------------------------------------------------------------------

require 'cutorch'
require 'xlua'

----------------------------------------------------------------------
-- define normalization function
-- return type: 2 number
--
function normalize(data, mean_, std_)
   local mean = mean_ or data:mean()
   local std = std_ or data:std()
      data:add(-mean)
   if std > 0 then
      data:mul(1/std)
   end
   return data, mean, std
end

----------------------------------------------------------------------
-- define Row Column list function
-- return type: table
-- 
function getRClist(datamap,threshold_,interval_)
   -- local vars
   local threshold = threshold_ or 0
   local interval = interval_ or 12
   local dim = datamap:size()
   
   -- create row column number map
   torch.setdefaulttensortype('torch.FloatTensor')
   local nbR = torch.range(1,dim[1]):repeatTensor(dim[2],1):t()
   local nbC = torch.range(1,dim[2]):repeatTensor(dim[1],1)
   torch.setdefaulttensortype('torch.CudaTensor')
   
   -- get all row map and column map
   local mapR = torch.cmul(datamap:gt(threshold):float(),nbR)
   local mapC = torch.cmul(datamap:gt(threshold):float(),nbC)
   
   -- assign them into individual maps
   local listR = mapR[torch.gt(mapR,0)]*interval-interval+1
   local listC = mapC[torch.gt(mapC,0)]*interval-interval+1
   
   return listR:cuda(),listC:cuda()
end

----------------------------------------------------------------------
-- define function : group discreted positive points
-- return type: Tensor
-- 
function detectGrouping(listR,listC,distance_)
   -- local vars
   torch.setdefaulttensortype('torch.FloatTensor')
   local distance = distance_ or 36
   local groupFlags = torch.IntTensor(listR:size(1)):fill(0)
   local groupNum = 0
   
   -- grouping
   for h_index = 1,listR:size(1) do
      -- 0 means host ungrouped
      if groupFlags[h_index] == 0 then 
         for t_index = 1,listR:size(1) do
            -- 0 means target ungrouped
            if groupFlags[t_index] == 0 then
               -- calculate distances
               local diffR = torch.abs(listR[h_index]-listR[t_index])
               local diffC = torch.abs(listC[h_index]-listC[t_index])
               
               -- set group number
               if (diffR <= distance and diffC <= distance) then
                  groupFlags[t_index] = groupNum+1
               end
            end
         end
         -- go to next group
         groupNum = groupNum+1
      end
   end
   
   -- find group centroid
   local groupR = torch.Tensor(groupNum):fill(0)
   local groupC = torch.Tensor(groupNum):fill(0)
   for g_index = 1,groupNum do
      groupR[g_index] = listR[torch.eq(groupFlags,g_index)]:mean()
      groupC[g_index] = listC[torch.eq(groupFlags,g_index)]:mean()
   end
   
   -- recursion
   if groupNum ~= listR:size(1) then
      groupR,groupC = detectGrouping(groupR,groupC)
   end
   
   torch.setdefaulttensortype('torch.CudaTensor')
   return groupR:cuda(),groupC:cuda()
end

----------------------------------------------------------------------
-- define drawing line function
-- return type: N/A
-- 
function getBoxedImages(datamap,listR,listC,bRGB,boxsize,thickness)
   -- set color here
   local rbit,gbit,bbit = bRGB[1],bRGB[2],bRGB[3]
   local rMax = boxsize or 94    -- range Max
   local rMin = thickness or 8   -- range Min
   
   -- draw blue boxes
   for subNum = 1,listR:size(1) do
      -- box
      local topL = {{listR[subNum],listR[subNum]+rMin-1},
                    {listC[subNum],listC[subNum]+rMax-1}}
      local botL = {{listR[subNum]+rMax-rMin,listR[subNum]+rMax-1},
                    {listC[subNum],listC[subNum]+rMax-1}}
      local lefL = {{listR[subNum],listR[subNum]+rMax-1},
                    {listC[subNum],listC[subNum]+rMin-1}}
      local rigL = {{listR[subNum],listR[subNum]+rMax-1},
                    {listC[subNum]+rMax-rMin,listC[subNum]+rMax-1}}
      
      -- red
      datamap[1][topL] = rbit
      datamap[1][botL] = rbit
      datamap[1][lefL] = rbit
      datamap[1][rigL] = rbit
      -- green
      datamap[2][topL] = gbit
      datamap[2][botL] = gbit
      datamap[2][lefL] = gbit
      datamap[2][rigL] = gbit
      -- blue
      datamap[3][topL] = bbit
      datamap[3][botL] = bbit
      datamap[3][lefL] = bbit
      datamap[3][rigL] = bbit                        
   end

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
   
   return labelFlags,occupFlags
end
