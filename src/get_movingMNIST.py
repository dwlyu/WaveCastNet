import numpy as np
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getData():
    np.random.seed(5544)
    
    # Load Data as Numpy Array
    MovingMNIST = np.load('data/mnist_test_seq.npy').transpose(1, 0, 2, 3)
    
    # Shuffle Data
    np.random.shuffle(MovingMNIST)
    
    # Train, Test, Validation splits
    train_data = MovingMNIST[:8000]         
    test_data = MovingMNIST[8000:10000]       
    
    return train_data, test_data

def collate(batch):
    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(np.array(batch)).unsqueeze(1)     
    batch = batch / 255.0                        
    batch = batch.to(device)                     

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)                     
    return batch[:,:,rand-10:rand], batch[:,:,rand]     


