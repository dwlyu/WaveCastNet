import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, TensorDataset
import gc

def getsequenceData_uq(input_array,
             input_steps = 20, future_steps = 5,down_sample=2,spatial_ds=1,
             batch_size = 16, pin_memory = False, shuffle=True):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation),
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''    
    return get_data_loader(input_array,input_steps,future_steps,down_sample,spatial_ds,batch_size,pin_memory,shuffle)

def get_data_loader(input_array,input_steps,future_steps,down_sample,spatial_ds,batch_size,pin_memory,shuffle): 
    seed = 3938
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    transform = torch.from_numpy
    dataset = GetClimateDataset(input_array, transform, input_steps,future_steps,down_sample,spatial_ds) 

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
    def __init__(self, input_array, transform, input_steps,future_steps,down_sample,spatial_ds):
        self.input =input_array
        self.transform = transform
        self.num_snapshots =input_steps
        self.num_predict = future_steps
        self.spatial_ds = spatial_ds
        self.down_sample = down_sample
        self.seq_len = 461
        self.sample_num = input_array.shape[0]

    def __len__(self):
        return (self.seq_len-self.down_sample*self.num_snapshots-self.down_sample*self.num_predict)*self.sample_num
            
    def __getitem__(self, global_idx):
        sample_idx,local_idx = self.get_indices(global_idx)
        X = self.transform(self.input[sample_idx,:,20:364:self.spatial_ds,16:240:self.spatial_ds, local_idx:local_idx + 
                                      self.down_sample*self.num_snapshots : self.down_sample]).permute(0,3,1,2)
        
        y = self.transform(self.input[sample_idx,:,20:364:self.spatial_ds,16:240:self.spatial_ds, local_idx + self.down_sample*self.num_snapshots:local_idx + 
                                      self.down_sample*(self.num_snapshots + self.num_predict) : self.down_sample]).permute(0,3,1,2)
        gc.collect()
        return X,y


    def get_indices(self, global_idx):
        each_sample = self.seq_len-self.down_sample*self.num_snapshots-self.down_sample*self.num_predict
        sample_idx = int(global_idx/each_sample)  # which sample we are on
        local_idx = int(global_idx % each_sample)
        return sample_idx,local_idx




if __name__ == "__main__":
    # import time
    inputarray = np.load("/pscratch/sd/d/dwlyu/new_data/inputlist.npy")
    train_loader = getsequenceData_uq(input_array=inputarray,sample_per_file=60,batch_size=1)
    for idx, (input,target) in enumerate (train_loader):
        print(idx)     
        print(input.shape)
        print(target.shape)