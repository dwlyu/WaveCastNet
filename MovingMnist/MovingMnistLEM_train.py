import numpy as np
import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader
from Seq2Seq import Seq2Seq

import argparse

parser = argparse.ArgumentParser(description='')

parser.add_argument("--lr", default=5e-4, help="")
parser.add_argument("--seed", default=1, help="")
parser.add_argument("--num", default=64, help="")

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()

learning_rate = float(args.lr)
random_seed = int(args.seed)
channel_num = int(args.num)

#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(random_seed)
# Load Data as Numpy Array
MovingMNIST = np.load('test/ConvLSTM/mnist_test_seq.npy').transpose(1, 0, 2, 3)

# Shuffle Data
np.random.shuffle(MovingMNIST)

# Train, Test, Validation splits
train_data = MovingMNIST[:8000]         
val_data = MovingMNIST[8000:9000]       
test_data = MovingMNIST[9000:10000]     

def collate(batch):

    # Add channel dim, scale pixels between 0 and 1, send to GPU
    batch = torch.tensor(batch).unsqueeze(1)     
    batch = batch / 255.0                        
    batch = batch.to(device)                     

    # Randomly pick 10 frames as input, 11th frame is target
    rand = np.random.randint(10,20)                     
    return batch[:,:,rand-10:rand], batch[:,:,rand]     


# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True, 
                        batch_size=16, collate_fn=collate)

# Validation Data Loader
val_loader = DataLoader(val_data, shuffle=True, 
                        batch_size=16, collate_fn=collate)
# Get a batch
# input, _ = next(iter(val_loader))

# Reverse process before displaying
# input = input.cpu().numpy() * 255.0    

# The input video frames are grayscale, thus single channel
model = Seq2Seq(dt=1, num_channels=1, num_kernels=channel_num, 
kernel_size=(3, 3), padding=(1, 1), activation="relu", 
frame_size=(64, 64), num_layers = 3).to(device)

#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')

# use input data to train the model
num_epochs = 100
trainloss_list = []
validloss_list = []

for epoch in range(1, num_epochs+1):
    
    if epoch % 80 == 0:
        learning_rate_1 = learning_rate/2
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate_1)

    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        output = model(input.to(device))
        output = output.clamp(0, 1)      
        output[output!=output] = 0                               
        loss = criterion(output.flatten(), target.flatten().to(device))       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()                                 
    train_loss /= len(train_loader.dataset)   
    trainloss_list.append(train_loss)   
                  

    val_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in val_loader:                          
            output = model(input.to(device)) 
            loss = criterion(output.flatten(), target.flatten())   
            val_loss += loss.item()                                
    val_loss /= len(val_loader.dataset)  
    validloss_list.append(val_loss)                        

    print("Epoch:{} Training Loss:{:.2f} Validation Loss:{:.2f}\n".format(
        epoch, train_loss, val_loss))
with open("test/ConvLSTM/ConvLEM/trainloss_res1:{:.0f}".format(channel_num), "wb") as fp:   #Pickling
        pickle.dump(trainloss_list, fp) 
with open("test/ConvLSTM/ConvLEM/validloss_res1:{:.0f}".format(channel_num), "wb") as fp:   #Pickling
        pickle.dump(validloss_list, fp)     
torch.save(model.state_dict(),'test/ConvLSTM/ConvLEM/LEM_res1:{:.0f}.pt'.format(channel_num))