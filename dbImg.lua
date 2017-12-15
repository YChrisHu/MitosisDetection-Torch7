----------------------------------------------------------------------
-- This file is used to reformat image to Tensor
-- 
-- I: images 94x94
-- O: file .t7 (ascii)
----------------------------------------------------------------------
require 'image'
nb = 35
data = torch.FloatTensor(nb,3,2084,2084)
for index = 1,nb do
   data[index] = image.load('./TData/'..tostring(index)..'.png')
end
genx =data
geny =torch.ones(nb):int()
dge = {}
dge[1] = genx
dge[2] = geny
torch.save('fullimage35.t7',dge)
