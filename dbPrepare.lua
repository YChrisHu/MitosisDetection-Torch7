----------------------------------------------------------------------
-- This file is used to create full 35 images database
-- 
-- I: images (.bmp is not supported)
-- I: cordinate labels
-- O: labeled images
----------------------------------------------------------------------

require 'torch'
require 'xlua'
require 'fnio'

----------------------------------------------------------------------
-- parse command-line options
--
local opt = lapp[[
   -d,--loaddir   (default "logs/mitosis") subdirectory to load nets
   -s,--savename  (default "trainFull")    save refined database
]]

-- fix seed
torch.manualSeed(1)

-- 2 classes
classes = {'NA','Mitosis'}

-- geo: width and height of input images
geo = {3,2086,2086}
dim = {2,167,167}

-- indicate full image
nbR = torch.range(1,dim[2]):repeatTensor(dim[3],1):t():clone():view(-1)
nbC = torch.range(1,dim[3]):repeatTensor(dim[2],1):clone():view(-1)
nbR = nbR*12-11
nbC = nbC*12-11

for image_index = 1,35 do
   mapData = loadImages(image_index,geo)
   mapLabel = loadLabels(image_index,geo)
   imgData = getImages(mapData,nbR,nbC)
   imgLabel = verifyImageLabels(mapLabel,nbR,nbC)
   imgData = imgData:float()
   imgLabel = imgLabel:int()
   if image_index == 1 then
      dge = {}
      dge[1] = imgData
      dge[2] = imgLabel
      print('<torch> '..dge[2]:size(1)..' samples in DB now')
      np = dge[2]:sum()-27889
      print('<torch> '..np..' positive samples')
   else
      print('<torch> merging dataset')
      dge[1] = torch.cat(dge[1],imgData,1)
      dge[2] = torch.cat(dge[2],imgLabel,1)
      print('<torch> '..dge[2]:size(1)..' samples in DB now')
      np = dge[2]:sum()-27889*image_index
      print('<torch> '..np..' positive samples')
   end
end

torch.save(opt.savename..'C.t7',dge)
