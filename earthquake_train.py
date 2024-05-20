import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 
import torch
from src.models_earthquake._init_ import AEConvLEM, AEConvGRU, AEConvLSTM, AEConvLEM_dense, AEConvLEM_sparse, VisionTransformer, Timesformer_eq, SwinTransformer3D_eq
from src.models_earthquake._init_ import getsequenceData
import src.models_earthquake.Validation_pixel as Validation_pixel
import gc
from torch.nn.parallel import DataParallel
import argparse
import os
    
class Huber(nn.Module):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean'):
        super(Huber, self).__init__()
        # Initialize any parameters you need for your custom loss function
        self.weight = weight
        self.size_average = size_average
        self.reduce = reduce
        self.reduction = reduction
        self.L1loss = torch.nn.L1Loss(reduction='mean')
        self.MSEloss = torch.nn.MSELoss(reduction='mean')

    def forward(self, output, target, delta=0.2): ##delta can be changed
        error = output - target
        abs_error = torch.abs(error)
        loss = torch.where(abs_error <= delta, 0.5 * error**2 /delta + 0.5 * delta, abs_error)
        return torch.mean(loss)


parser = argparse.ArgumentParser(description='')
# Loading options
parser.add_argument("--model", type=str, default='LEM_dense', 
                    choices=['LEM_dense','LEM_sparse','LEM', 'LSTM', 'GRU', 'Swin','Time-s-pyramid','Time-s-plain'], 
                    help="Choose architecture.")

parser.add_argument("--activation", type=str, default='tanh', choices=['tanh','relu'], help="Activation function")
parser.add_argument("--training_uq", type=bool, default=False, help="Uncertainty quantification")
parser.add_argument("--retrain", type=bool, default=False, help="If it's retraining")
parser.add_argument('--load_seed', type=int, default=1, help='Loading seed')
parser.add_argument('--num_kernels', type=int, default=144, metavar='S', help='Latent space dimension')
parser.add_argument('--patch_size', type=int, nargs=3, default=(1, 8, 8), help="Patch size as three integers (d, h, w)")
parser.add_argument('--window_size', type=int, nargs=3, default=(2, 6, 6), help="Window size as three integers (d, h, w)")
# parser.add_argument('--spatial_ds', type=int, default=1, help="Spatial down sample scale factor")
# parser.add_argument('--temporal_ds', type=int, default=2, help="Temporal down sample scale factor")

# Optimization options
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='Initial learning rate.')
parser.add_argument('--decay', type=float, default=0.0, help='Weight decay (L2 penalty).')
parser.add_argument('--scheduler', type=int, default=25, help='Learning rate steps.')
parser.add_argument('--gamma', type=float, default=0.5, help='Learning rate decay.')
parser.add_argument('--batch_size', type=int, default=64, help='Number of train data points per batch.')
parser.add_argument('--batch_size_test',  type=int, default=8, help='Number of train data points per batch.')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args = parser.parse_args()


#==============================================================================
# Loading Data
#==============================================================================

if args.training_uq:
    load_seed = (int)(args.load_seed)
    np.random.seed(load_seed^3 + load_seed)
    indices_to_keep = np.arange(300)
    mask = np.ones(300, dtype=bool)
    for i in range(15): # Iterate over 15 different depths
        rand = np.random.choice(np.arange(0, 20), size=4, replace=False) # Choose 20 epic centers randomly at a certain epic center depth
        for r in rand:
            file_id = (int)(r/4)
            idx = (int)(r%4)
            mask[file_id * 60 + i * 4 + idx] = False
    indices_to_keep = indices_to_keep[mask]

    inputlist = np.load("/pscratch/sd/d/dwlyu/new_data/input/all_input.npy", mmap_mode='r') # Training data path
    inputlist = inputlist[indices_to_keep]
    print(inputlist.shape)
    print('finish loading data...............................')

    testlist = np.load("/pscratch/sd/d/dwlyu/new_data/input/all_validation.npy") # Validation data path
    print(testlist.shape)
    print('finish loading data...............................')
else:
    inputlist = np.load("/pscratch/sd/d/dwlyu/new_data/input/all_input.npy")
    print(inputlist.shape)
    print('finish loading data...............................')

    testlist = np.load("/pscratch/sd/d/dwlyu/new_data/input/all_validation.npy")
    print(testlist.shape)
    print('finish loading data...............................')

if args.model == 'LEM_dense' or args.model == 'LEM_sparse':
    train_loader = getsequenceData(inputlist,input_steps=30, future_steps=30,batch_size=args.batch_size,
                                    down_sample=2, spatial_ds=1,pin_memory=True)
    val_loader = getsequenceData(testlist, input_steps=30, future_steps=195, batch_size=args.batch_size_test,
                                    down_sample=2, spatial_ds=1,pin_memory=True)
    step = 30
elif args.model == 'Time-s-plain' or args.model == 'Time-s-pyramid' or args.model == 'Swin':
    train_loader = getsequenceData(inputlist, input_steps=60, future_steps=60,batch_size=args.batch_size,
                                    cropx=[30,350], cropy=[16,240], down_sample=1, spatial_ds=4,pin_memory=True)
    val_loader = getsequenceData(testlist, input_steps=60, future_steps=390, batch_size=args.batch_size_test,
                                    cropx=[30,350], cropy=[16,240], down_sample=1, spatial_ds=4,pin_memory=True)
    step = 60
    # Ablation Study Setup
else:
    train_loader = getsequenceData(inputlist, input_steps=60, future_steps=60,batch_size=args.batch_size,
                                    cropx=[24,360], cropy=[16,240], down_sample=1, spatial_ds=4,pin_memory=True)
    val_loader = getsequenceData(testlist, input_steps=60, future_steps=390, batch_size=args.batch_size_test,
                                    cropx=[24,360], cropy=[16,240], down_sample=1, spatial_ds=4,pin_memory=True)
    step = 60

#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
seed_everything(args.seed)

#==============================================================================
# creating loss list
#==============================================================================

trainloss_list = [] 
validMSE = []
validRFNE = []
validACC = []
validRMSE =[]
patch_size = tuple(args.patch_size)
window_size = tuple(args.window_size)
        
#==============================================================================
# loading models
#==============================================================================

if args.model == 'LEM_dense':
    model = AEConvLEM_dense(dt=1, num_channels=3, num_kernels=args.num_kernels, 
    kernel_size=(3, 3), padding=(1, 1), activation=args.activation, 
    frame_size=(43,28)).to(device)

elif args.model == 'LEM_sparse':
    model = AEConvLEM_sparse(dt=1, num_channels=3, num_kernels=args.num_kernels, 
    kernel_size=(3, 3), padding=(1, 1), activation=args.activation, 
    frame_size=(43,28)).to(device)

elif args.model == 'LEM':
    model = AEConvLEM(dt=1, num_channels=3, num_kernels=args.num_kernels, 
    kernel_size=(3, 3), padding=(1, 1), activation=args.activation, 
    frame_size=(21,14)).to(device)
        
elif args.model == 'GRU':
    model = AEConvGRU(dt=1, num_channels=3, num_kernels=args.num_kernels, 
    kernel_size=(3, 3), padding=(1, 1), activation=args.activation, 
    frame_size=(21,14)).to(device)

elif args.model == 'LSTM':
    model = AEConvLSTM(num_channels=3, num_kernels=args.num_kernels, 
    kernel_size=(3, 3), padding=(1, 1), activation=args.activation, 
    frame_size=(21,14)).to(device)
    
elif args.model == 'Swin':
    model = SwinTransformer3D_eq(embed_dim=args.num_kernels,patch_size=patch_size,window_size=window_size).to(device)
    
elif args.model == 'Time-s-plain':
    model = VisionTransformer(embed_dim=args.num_kernels,patch_size=patch_size).to(device)
    
elif args.model == 'Time-s-pyramid':
    model = Timesformer_eq(embed_dim=args.num_kernels,patch_size=patch_size).to(device)
# data parallel    
if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    
print('**** Setup ****')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')

#==============================================================================
# Optimizer and Loss
#==============================================================================
    
optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.scheduler, gamma=args.gamma)
criterion = Huber(reduction='mean')
num_epochs = args.epochs

DESTINATION_PATH = 'Earthquake_results/'
if not os.path.isdir(DESTINATION_PATH):
                            os.mkdir(DESTINATION_PATH)
best_error= np.inf
checkpoint_dir = DESTINATION_PATH
if args.model == 'Swin' or args.model == 'Time-s-plain' or args.model == 'Time-s-pyramid':
    OUT_DIR = os.path.join(DESTINATION_PATH, f'arch_{args.model}_patchsize_{args.patch_size}_dim_{args.num_kernels}_lr_{args.learning_rate}_seed_{args.seed}')
else:
    OUT_DIR = os.path.join(DESTINATION_PATH, f'arch_{args.model}_dim_{args.num_kernels}_lr_{args.learning_rate}_seed_{args.seed}')
    
#==============================================================================
# Retraining Setup
#==============================================================================

if args.retrain:
    state_dict  = torch.load(f'{OUT_DIR}_final_.pt') # Loading the final model
    model.load_state_dict(state_dict=state_dict)
    
    trainloss_list = list(np.load(f'{OUT_DIR}_train.npy'))
    validMSE = list(np.load(f'{OUT_DIR}_MSE.npy'))
    validRFNE = list(np.load(f'{OUT_DIR}_RFNE.npy'))
    validRMSE = list(np.load(f'{OUT_DIR}_RMSE.npy'))
    validACC = list(np.load(f'{OUT_DIR}_ACC.npy'))

    checkpoint_path = f'{OUT_DIR}_checkpoint.pt'
    checkpoint = torch.load(checkpoint_path)

    np.random.set_state(checkpoint['random_state']['numpy_seed'])
    torch.set_rng_state(checkpoint['random_state']['torch_seed'])
    torch.cuda.set_rng_state_all(checkpoint['random_state']['torch_cuda_seed'])  
    optim.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['schedular_state_dict'])

#==============================================================================
# training and validating process
#==============================================================================

for epoch in range(1, num_epochs+1):
    iteration = 0
    train_loss = 0            
    random_state = {
        'numpy_seed': np.random.get_state(),
        'torch_seed': torch.get_rng_state(),
        'torch_cuda_seed': torch.cuda.get_rng_state_all()
    }

    # Save the model and optimizer checkpoint
    torch.save({
        'epoch': epoch,
        'optimizer_state_dict': optim.state_dict(),
        'schedular_state_dict': scheduler.state_dict(),
        'random_state': random_state
    }, f'{OUT_DIR}_checkpoint.pt')        
                                 
    model.train()
                                                  
    for batch_num, (input, target) in enumerate(train_loader, 1):  
        input = input.to(device)
        target = target.to(device)
        if args.model == 'Time-s-plain' or args.model == 'Time-s-pyramid' or args.model == 'Swin':
            output = model(input)
        else:
            output = model(input, future_seq = step)
        loss = criterion(output.flatten(),target.flatten())     
        loss.backward()                                            
        optim.step()                                               
        optim.zero_grad()     
        loss.detach()                                      
        train_loss += loss.item()    
        iteration += 1                      
    
    scheduler.step()
    train_loss /= iteration   
    trainloss_list.append(train_loss)   
                  

    MSEloss = 0
    Rfne = 0
    Rmse = 0  
    Acc = 0 
    iteration = 0      
    print('Validation')                                   
    model.eval()                                                  
    with torch.no_grad():                                          
        for input, target in val_loader:
            input = input.to(device)
            target = target.to(device)
            output = []
            for i in range(7):
                
                if i < 6:
                    if args.model == 'Time-s-plain' or args.model == 'Time-s-pyramid' or args.model == 'Swin':
                        r2 = model(input).detach()
                    else:
                        r2 = model(input, future_seq = step).detach()
                    output.append(r2)
                else:
                    if args.model == 'Time-s-plain' or args.model == 'Time-s-pyramid' or args.model == 'Swin':
                        r2 = model(input).detach()
                        r2 = r2[:,:,:step//2]
                    else:
                        r2 = model(input, future_seq = step//2).detach()
                    output.append(r2)
                input = r2
                
            output = torch.concat(output,dim=2)                
            loss = criterion((output).flatten(), target.flatten()).detach()                                                                                        
            MSEloss += loss.item()
            rfne1 = Validation_pixel.validation_rfne((output),target)
            Rfne += rfne1
            rmse1 =Validation_pixel.validation_rmse((output),target)
            Rmse += rmse1 
            acc1 = Validation_pixel.validation_acc((output),target)
            Acc += acc1
            iteration += 1
            gc.collect()
            
    MSEloss /= iteration
    Rfne /= iteration
    Rmse /= iteration
    Acc /= iteration
    validRMSE.append(Rmse)
    validMSE.append(MSEloss)
    validRFNE.append(Rfne)
    validACC.append(Acc)
    
    print("Epoch:{} Training Loss:{:.6f} Validationloss: MSE:{:.6f} RFNE:{:.6f} RMSE:{:.6f} ACC:{:.6f}\n".format(
        epoch, train_loss, MSEloss, Rfne, Rmse, Acc))
    if MSEloss < best_error:
        best_error = MSEloss
        torch.save(model.state_dict(),f'{OUT_DIR}_best_.pt') # Best model
        
    np.save(f'{OUT_DIR}_train', trainloss_list)
    np.save(f'{OUT_DIR}_MSE',  validMSE)
    np.save(f'{OUT_DIR}_RFNE',  validRFNE)
    np.save(f'{OUT_DIR}_RMSE',  validRMSE)
    np.save(f'{OUT_DIR}_ACC',  validACC)
    torch.save(model.state_dict(),f'{OUT_DIR}_final_.pt') #Final model