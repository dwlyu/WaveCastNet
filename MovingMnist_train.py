import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pickle

from src.models_movingMNIST import Seq2SeqLEM, Seq2SeqLSTM, Seq2SeqQRNN
from src.get_movingMNIST import getData, collate

import argparse
parser = argparse.ArgumentParser(description='Learn Moving MNIST', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='LEM', choices=['LEM', 'LSTM', 'QRNN'], help='Choose architecture.')
parser.add_argument('--width', type=int, default=64, metavar='S', help='number of channels')
parser.add_argument('--num_layers', type=int, default=3, metavar='S', help='number of recurrent layers')


# Optimization options
parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--decay', type=float, default=0.0, help='Weight decay (L2 penalty).')
parser.add_argument('--gamma', type=float, default=10, help='Learning rate decay.')
parser.add_argument('--batch_size', type=int, default=16, help='Number of train data points per batch.')
parser.add_argument('--batch_size_test',  type=int, default=32, help='Number of train data points per batch.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')



# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


#==============================================================================
# Get Data
#==============================================================================
train_data, test_data = getData()

# Training Data Loader
train_loader = DataLoader(train_data, shuffle=True, 
                        batch_size=args.batch_size, collate_fn=collate)

# Validation Data Loader
test_loader = DataLoader(test_data, shuffle=True, 
                        batch_size=args.batch_size_test, collate_fn=collate)



#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True


# The input video frames are grayscale, thus single channel

if args.model == 'LEM':
    model = Seq2SeqLEM(dt=1, num_channels=1, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(64, 64), num_layers = args.num_layers).to(device)
if args.model == 'LSTM':
    model = Seq2SeqLSTM(num_channels=1, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(64, 64), num_layers = args.num_layers).to(device)
if args.model == 'QRNN':
    model = Seq2SeqQRNN(num_channels=1, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(64, 64), num_layers = args.num_layers).to(device)


#==============================================================================
# Model summary
#==============================================================================
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')


#==============================================================================
# Optimizer and Loss
#==============================================================================
optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate, 
                         weight_decay=args.decay)

# Binary Cross Entropy, target pixel values either 0 or 1
criterion = nn.BCELoss(reduction='sum')



#==============================================================================
# Start Training
#==============================================================================
# use input data to train the model
num_epochs = 100
trainloss_list = []
validloss_list = []
best_eval = np.inf

for epoch in range(1, num_epochs+1):
    
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
    

    if epoch % 150 == 0:
        learning_rate_new = args.learning_rate / args.gamma
        for param_group in optim.param_groups:
            param_group['lr'] = learning_rate_new              


    test_loss = 0                                                 
    model.eval()                                                   
    with torch.no_grad():                                          
        for input, target in test_loader:                          
            output = model(input.to(device)) 
            loss = criterion(output.flatten(), target.flatten())   
            test_loss += loss.item()                                
    test_loss /= len(test_loader.dataset)  
    validloss_list.append(test_loss)                        
            

    if (test_loss < best_eval):
        best_eval = test_loss
        # save best model
        DESTINATION_PATH = 'MovingMNIST_results/'
        OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.model}_width_{args.width}_lr_{args.learning_rate}_seed_{args.seed}')
        if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
        torch.save(model, OUT_DIR + '.pt')  

    print("Epoch: {} Training Loss: {:.2f} Validation Loss: {:.2f}\n".format(
        epoch, train_loss, test_loss))
           
    np.save(OUT_DIR + "testloss", validloss_list)
    np.save(OUT_DIR + "trainloss", trainloss_list)

    

# save final model
OUT_DIR = os.path.join(DESTINATION_PATH, f'final_arch_{args.model}_width_{args.width}_lr_{args.learning_rate}_seed_{args.seed}')
if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
torch.save(model, OUT_DIR+'.pt')          
        

