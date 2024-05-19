import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
# import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import gc
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def getsequenceData(input_array, sample_per_file,
             input_steps = 20, future_steps = 5,down_sample=2,spatial_ds=1,
             batch_size = 16, cropx=[20,364],cropy=[16,240], pin_memory = False, shuffle=True):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation),
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''    
    return get_data_loader(input_array,sample_per_file,input_steps,future_steps,down_sample,cropx,cropy,spatial_ds,batch_size,pin_memory,shuffle)

def get_data_loader(input_array,sample_per_file,input_steps,future_steps,down_sample,cropx,cropy,spatial_ds,batch_size,pin_memory,shuffle):
    
    transform = torch.from_numpy
    dataset = GetClimateDataset(input_array,sample_per_file, transform, input_steps,future_steps,down_sample,cropx,cropy,spatial_ds) 

    dataloader = DataLoader(dataset,
                            batch_size = int(batch_size),
                            num_workers = 4, # TODO: make a param
                            shuffle = shuffle, 
                            sampler = None,
                            drop_last = True,
                            pin_memory = pin_memory)

    return dataloader


class GetClimateDataset(Dataset):
    '''Dataloader class for NSKT and cosmo datasets'''
    def __init__(self, input_array,sample_per_file, transform, input_steps,future_steps,down_sample,cropx,cropy,spatial_ds):
        self.input =input_array
        self.transform = transform
        self.num_snapshots =input_steps
        self.num_predict = future_steps
        self.spatial_ds = spatial_ds
        self.down_sample = down_sample
        self.seq_len = 461
        self.sample_per_file = sample_per_file
        self.file_num = input_array.shape[0]
        self.sample_num = self.file_num * self.sample_per_file
        self.data_all = np.zeros((self.file_num,self.sample_per_file,3, 376, 256, self.seq_len), dtype=np.float32)
        self.cropx = cropx
        self.cropy = cropy
        # self.device = torch.device('cuda')
        self._load_data_()

    def __len__(self):
        return (self.seq_len-self.down_sample*self.num_snapshots-self.down_sample*self.num_predict)*self.sample_num
    
    def _load_data_(self):
        for i in range(self.file_num):
            file_id = self.input[i]
            datadir = f'/pscratch/sd/d/dwlyu/new_data/{file_id}.npy'
            self.data_all[i] = np.load(datadir)
        print('finish loading sets..............')
            

    def __getitem__(self, global_idx):
        # print("batch_idx: {}".format(batch_idx))
        file_idx,sample_idx,local_idx = self.get_indices(global_idx)
        X = self.transform(self.data_all[file_idx,sample_idx,:,self.cropx[0]:self.cropx[1]:self.spatial_ds,self.cropy[0]:self.cropy[1]:self.spatial_ds, local_idx:local_idx + 
                                      self.down_sample*self.num_snapshots : self.down_sample]).permute(0,3,1,2)
        
        y = self.transform(self.data_all[file_idx,sample_idx,:,self.cropx[0]:self.cropx[1]:self.spatial_ds,self.cropy[0]:self.cropy[1]:self.spatial_ds, local_idx + self.down_sample*self.num_snapshots:local_idx + 
                                      self.down_sample*(self.num_snapshots + self.num_predict) : self.down_sample]).permute(0,3,1,2)
        gc.collect()
        return X,y


    def get_indices(self, global_idx):
        each_sample = self.seq_len-self.down_sample*self.num_snapshots-self.down_sample*self.num_predict
        all_sample_idx = int(global_idx/each_sample)  # which sample we are on
        file_idx = int(all_sample_idx/self.sample_per_file)
        sample_idx = int(all_sample_idx % self.sample_per_file)
        local_idx = int(global_idx % each_sample)
        return file_idx,sample_idx,local_idx




if __name__ == "__main__":
    # import time
    inputarray = np.load("/pscratch/sd/d/dwlyu/new_data/inputlist.npy")
    train_loader = getsequenceData(input_array=inputarray,sample_per_file=60,batch_size=1)
    for idx, (input,target) in enumerate (train_loader):
        print(idx)     
        print(input.shape)
        print(target.shape)