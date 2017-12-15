----------------------------------------------------------------------
-- This file is used to devide DB into multiple parts
-- 
-- I: 3 DBs
-- O: 13 DBs
----------------------------------------------------------------------

require 'torch'
require 'image'

----------------------------------------------------------------------
-- load dataset
--

-- load training & testing set
print('<torch> loading false positive samples')
fpData = torch.load('./database/trainReC.t7')
print('<torch> loading true positive samples')
tpData = torch.load('./database/trainFGC.t7')
print('<torch> loading random negative samples')
rnData = torch.load('./database/trainFullC.t7')

-- merge
print('<torch> merging samples')
data = torch.cat({fpData[1],tpData[1],rnData[1]},1)
label = torch.cat({fpData[2],tpData[2],rnData[2]},1)
fpData,tpData,rnData = nil,nil,nil
collectgarbage()

-- shuffle
shuffle = torch.randperm(1064960):long()
for i = 1,13 do
   print('<torch> separating samples '..i)
   local src = data:index(1,shuffle[{{i*81920-81919,i*81920}}])
   local lrc = label:index(1,shuffle[{{i*81920-81919,i*81920}}])
   
   -- store the scaled images
   local tre = {}
   tre[1] = src
   tre[2] = lrc
   torch.save('./database/partial/'..i..'C.t7',tre)
   src,lrc,tre = nil,nil,nil
   collectgarbage()
end

