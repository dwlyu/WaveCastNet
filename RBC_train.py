import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import h5py
import Validation

from models_RBC import AEConvLEM, AEConvLSTM, AEConvQRNN, Validation
import argparse
parser = argparse.ArgumentParser(description='Learn Moving MNIST', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model', type=str, default='LEM', choices=['LEM', 'LSTM', 'QRNN'], help='Choose architecture.')
parser.add_argument('--width', type=int, default=72, metavar='S', help='number of channels')
parser.add_argument('--input_steps', type=int, default=20, metavar='S', help='input step size')
parser.add_argument('--future_steps', type=int, default=10, metavar='S', help='predicted step size')

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
def generate_dataset(input_array, input_step=5, batch_size=32, shuffle=False, future_step = 5):
    inputs, targets = [], []
    for idx in range(input_array.shape[0]-input_step-future_step):
        inputs.append(torch.from_numpy(input_array[idx:idx+input_step,:, :128, :]).permute(1,0,2,3))
        targets.append(torch.from_numpy(input_array[idx+input_step:idx+future_step+input_step,:, :128, :]).permute(1,0,2,3))
    dataset =TensorDataset(torch.stack(inputs, dim=0), torch.stack(targets, dim=0))
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    return dataloader

file = h5py.File("/scratch/data/RBC/rbc_3569_512128/rbc_3569_512128_s2.h5", 'r')
w = file['tasks']['vorticity'][()].astype('float32')
u = file['tasks']['u'][()].astype('float32')
v = file['tasks']['v'][()].astype('float32')
h5_file_vor = np.stack((w, u, v), axis=1)
# Calculate pixel-wise mean and standard deviation from training set
vor_mean = np.mean(h5_file_vor[:1600, :, :], axis=0)
vor_std = np.std(h5_file_vor[:1600, :, :], axis=0)
# Subtract the mean
h5_file_vor -= vor_mean
# Divide by the standard deviations
h5_file_vor /= vor_std
# Tranforming mean and std from numpy to tensor
tensor_mean = torch.tensor(vor_mean[:,:128,:128],dtype=torch.float32).to(device).view(1,3,1,128,128)
tensor_std = torch.tensor(vor_std[:,:128,:128],dtype=torch.float32).to(device).view(1,3,1,128,128)

train_loader = generate_dataset(h5_file_vor[:1600], input_step=args.input_steps, batch_size=args.batch_size, shuffle=True, future_step=args.future_steps)
val_loader = generate_dataset(h5_file_vor[1600:2000], input_step=args.input_steps, batch_size=args.batch_size_test, shuffle=True, future_step=args.future_steps)
#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
seed_everything(args.seed)

if args.model == 'LEM':
    model = AEConvLEM(dt=1, num_channels=3, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(32, 32)).to(device)
if args.model == 'LSTM':
    model = AEConvLSTM(num_channels=3, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(32, 32)).to(device)
if args.model == 'QRNN':
    model = AEConvQRNN(num_channels=3, num_kernels=args.width, 
    kernel_size=(3, 3), padding=(1, 1), activation="relu", 
    frame_size=(32, 32)).to(device)


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
criterion = torch.nn.MSELoss(reduction='mean')

#==============================================================================
# Start Training
#==============================================================================
# use input data to train the model
num_epochs = args.epochs
trainloss_list = []
validMSE = []
validRFNE = []
validRMSE = []
validACC = []
best_eval= np.inf

for epoch in range(1, num_epochs+1):
    if epoch % 75 == 0:
        learning_rate_new = args.learning_rate / args.gamma
        for param_group in optim.param_groups:
            param_group['lr'] = learning_rate_new 

    iteration = 0
    train_loss = 0                                                 
    model.train()                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        output = model((input.to(device)),future_seq = args.future_steps)                             
        loss = criterion((output).flatten(), target.flatten().to(device))       
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()                                           
        train_loss += loss.item()    
        iteration += 1                             
    
    # scheduler.step()
    train_loss /= iteration   
    trainloss_list.append(train_loss)   
                  

    MSEloss = 0
    Rfne = 0
    Rmse = 0  
    Acc = 0 
    iteration = 0                                            
    model.eval()                                                  
    with torch.no_grad():                                          
        for input, target in val_loader:
            output1 = model((input.to(device)),future_seq = args.future_steps)
            output = output1*tensor_std + tensor_mean 
            target = target.to(device)
            target *= tensor_std
            target += tensor_mean                       
            loss = criterion((output).flatten(), target.to(device).flatten())                                                                                        
            MSEloss += loss.item()
            rfne1 = Validation.validation_rfne((output),target.to(device))
            Rfne += rfne1
            rmse1 = Validation.validation_rmse((output),target.to(device))
            Rmse += rmse1 
            acc1 = Validation.validation_acc((output),target.to(device))
            Acc += acc1
            iteration += 1
    MSEloss /= iteration
    Rfne /= iteration
    Rmse /= iteration
    Acc /= iteration
    validMSE.append(MSEloss)
    validRFNE.append(Rfne)
    validACC.append(Acc)       
    validRMSE.append(Rmse)                
            

    if (MSEloss < best_eval):
        best_eval = MSEloss
        # save best model
        DESTINATION_PATH = 'RBC_results/'
        OUT_DIR = os.path.join(DESTINATION_PATH, f'best_arch_{args.model}_width_{args.width}_lr_{args.learning_rate}_seed_{args.seed}')
        if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
        torch.save(model, OUT_DIR + '.pt')  

    print("Epoch:{} Training Loss:{:.6f} Validationloss: MSE:{:.6f} RFNE:{:.6f} RMSE:{:.6f} ACC:{:.6f}\n".format(
        epoch, train_loss, MSEloss, Rfne, Rmse, Acc))
           
    np.save(OUT_DIR + "testloss", validMSE)
    np.save(OUT_DIR + "trainloss", trainloss_list)
    np.save(OUT_DIR + 'rfne',  validRFNE)
    np.save(OUT_DIR + 'acc',  validACC)
    np.save(OUT_DIR + 'rmse',  validRMSE)
    

# save final model
OUT_DIR = os.path.join(DESTINATION_PATH, f'final_arch_{args.model}_width_{args.width}_lr_{args.learning_rate}_seed_{args.seed}')
if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
torch.save(model, OUT_DIR+'.pt')