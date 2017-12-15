----------------------------------------------------------------------
-- This file is used to perform mitosis detection  
-- with trained coarse net and fine net
-- 
-- I: image (.bmp is not supported)
-- O: labeled image
----------------------------------------------------------------------

matio = require 'matio'
require 'torch'
require 'nn'
require 'dp'
require 'optim'
require 'image'
require 'paths'
require 'cutorch'
require 'cunn'
require 'fnpp'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--savedir   (default "logs")         subdirectory to save logs
   -d,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -p,--prenorm                            conduct prefixed normalization
   -t,--threshold (default 0.99999)        threshold for final confidence
   -n,--name      (default "4009.png")        input image file name
   -b,--batchSize (default 32)             batch size
]]

----------------------------------------------------------------------
-- load models on mitosis detection
--
local filename = paths.concat(opt.loaddir, 'Nc.net')
print('<torch> loading corase net from '..filename)
mNc = torch.load(filename)
print('<torch> done')
pNc = mNc:parameters()

----------------------------------------------------------------------
-- load image and generate feature maps
--
data = image.load('./Data/'..opt.name)
data = data:cuda()
print('<torch> conducting preloaded normalization')
local infoname = paths.concat(opt.savedir, 'mitosis.info')
st = torch.load(infoname)
mean = st[1]
std = st[2]
data = normalize(data ,mean, std)
print('<torch> done')

-- modify model to generate feature
for i = 1,7 do
   mNc:remove()
end
mNc:evaluate()

-- get feature maps
c3 = mNc:forward(data)
mNc:remove()
mNc:remove()
mNc:remove()
c2 = mNc:forward(data)
mNc:remove()
mNc:remove()
mNc:remove()
c1 = mNc:forward(data)
c1,c2,c3 = c1:float(),c2:float(),c3:float()

-- store in mat as tabel
print('<torch> saving result')
tableFeature = {C1=c1,C2=c2,C3=c3}
matio.save('featureMap.mat',tableFeature)
