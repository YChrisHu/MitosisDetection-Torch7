----------------------------------------------------------------------
-- This file is defining dataset creation and noralization
-- 
-- I: images 94x94
-- I: labels
-- O: database containing images and labels
-- P: ds_dir, ds_trest, ds_teset, ds_ascii
----------------------------------------------------------------------

require 'torch'
require 'paths'

----------------------------------------------------------------------
-- inti vars
--
ds = {}

ds.path_dataset = ds_dir or './database/'
ds.path_trainset = ds_trset or paths.concat(ds.path_dataset, 'train.t7')
ds.path_testset = ds_teset or paths.concat(ds.path_dataset, 'test.t7')
ds.ascii = true

----------------------------------------------------------------------
-- define training set and testing set
--
function ds.loadTrainSet(maxLoad, geometry)
   return ds.loadDataset(ds.path_trainset, maxLoad, geometry)
end

function ds.loadTestSet(maxLoad, geometry)
   return ds.loadDataset(ds.path_testset, maxLoad, geometry)
end

----------------------------------------------------------------------
-- define data loading and normalizing
--
function ds.loadDataset(fileName, maxLoad)
   local f = nil
   if ds.ascii then
      f = torch.load(fileName, 'ascii')
   else
      f = torch.load(fileName)
   end
   local data = f[1]   --f[1] is data
   local labels = f[2]   --f[2] is label

   local nExample = f[1]:size(1)
   if maxLoad and maxLoad > 0 and maxLoad < nExample then
      nExample = maxLoad
      print('<dataset> loading only ' .. nExample .. ' examples')
   end
   data = data[{{1,nExample},{},{},{}}]
   labels = labels[{{1,nExample}}]
   print('<dataset> done')

   local dataset = {}
   dataset.data = data
   dataset.labels = labels

   -- individual normalization
   function dataset:normalize(mean_, std_)
      local mean = torch.Tensor(data:size(1))
      local std = torch.Tensor(data:size(1))
      for i=1,data:size(1) do
         local mean2 = mean_ or data[i]:mean()
         local std2 = std_ or data[i]:std()
         mean[i] = mean2
         std[i] = std2
            data[i]:add(-mean2)
         if std2 > 0 then
            data[i]:mul(1/std2)
         end
      end
      return mean, std
   end
   
   -- global normalization
   function dataset:normalizeGlobal(mean_, std_)
      local std = std_ or data:std()
      local mean = mean_ or data:mean()
      data:add(-mean)
      data:mul(1/std)
      return mean, std
   end
   
   -- used to index database
   function dataset:size()
      return nExample
   end

   local labelvector = torch.zeros(2)

   setmetatable(dataset, {__index = function(self, index)
              local input = self.data[index]
              local class = self.labels[index]
              local label = labelvector:zero()
              label[class] = 1
              local example = {input, label}
                                       return example
   end})

   return dataset
end
