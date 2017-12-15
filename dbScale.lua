----------------------------------------------------------------------
-- This file is used to enlarge 94x DB to 227x DB
-- to fit in caffeNet
-- 
-- I: images 94x94
-- O: images 227x227
----------------------------------------------------------------------

require 'torch'
require 'image'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -b,--number        (default 14336)          samples in dataset
   -a,--ascii                                  save as ascii format (large)
   -s.--shuffle                                shuffle dataset
   -n,--name          (default "trainFG")      database name
]]

-- geometry: width and height of input images
geometry = {94,94}

----------------------------------------------------------------------
-- load dataset
--

-- load training & testing set
if opt.ascii then
   data = torch.load('./database/'..opt.name..'.t7','ascii')
else
   data = torch.load('./database/'..opt.name..'.t7')
end

-- take the image data
if opt.shuffle then
   shuffle = torch.randperm(data[1]:size(1))
   src = data[1]:index(1,shuffle[{{1,opt.number}}]:long())
   lrc = data[2]:index(1,shuffle[{{1,opt.number}}]:long())
else
   src = data[1][{{1,opt.number},{},{},{}}]
   lrc = data[2][{{1,opt.number}}]
end

----------------------------------------------------------------------
-- scale the image
--

--set default type
torch.setdefaulttensortype('torch.FloatTensor')

-- initialize storage
res = torch.zeros(opt.number,3,227,227)

-- scale images
for i = 1, opt.number do
   res[i] = image.scale(src[i], 227, 227)
end

----------------------------------------------------------------------
-- store the scaled images
--
data[1] = res
data[2] = lrc
if opt.ascii then
   data = torch.save('./database/'..opt.name..'F.t7',data,'ascii')
else
   data = torch.save('./database/'..opt.name..'F.t7',data)
end

