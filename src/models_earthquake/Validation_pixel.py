import numpy as np
import torch
from torch import nn

def validation_rfne(output,target):
    batch_size, _, seq_len, height, width = target.size()
    rfne = []
    for i in range(batch_size):
        for j in range(_):
                err_rfne = torch.norm((target[i,j,:,:,:]-output[i,j,:,:,:]))/ torch.norm(target[i,j,:,:,:])
                rfne.append(err_rfne.detach())
                    
                    
    rfne1 = torch.mean(torch.tensor(rfne)).item()
    return rfne1

def validation_rmse(output,target):
    batch_size, _, seq_len, height, width = target.size()
    criterion = torch.nn.MSELoss(reduction='mean')
    rmse = []
    for i in range(batch_size):
        for j in range(_):
                err_rmse = torch.sqrt(criterion(output[i,j,:,:,:].flatten(),target[i,j,:,:,:].flatten())).detach()
                rmse.append(err_rmse.detach())
    
    return torch.mean(torch.tensor(rmse)).item()

def validation_acc(output,target):
    batch_size, _, seq_len, height, width = target.size()
    acc = []
    for i in range(batch_size):
        for j in range(_):
            sum1 = torch.sum((output[i,j,:,:,:]*target[i,j,:,:,:]).flatten())
            sum2 = torch.sum((target[i,j,:,:,:]*target[i,j,:,:,:]).flatten())
            sum3 = torch.sum((output[i,j,:,:,:]*output[i,j,:,:,:]).flatten())
            snapshot_acc = sum1/torch.sqrt(sum2*sum3)
            acc.append(snapshot_acc.detach())
                    
                    
    return torch.mean(torch.tensor(acc)).item()

def pgv_error(output, target):
    batch_size, _, seq_len, height, width = target.size()
    PGVerror = []
    for i in range(batch_size):
        for j in range(_):
            for h in range(height):
                for w in range(width):
                    pgv_true = torch.max(torch.abs(target[i,j,:,h,w])) # my tensor dimension is [batch, channel, time_length]
                    pgv_pred = torch.max(torch.abs(output[i,j,:,h,w])) 
                    PGVerror.append(torch.abs(pgv_true - pgv_pred))
                   
    return torch.mean(torch.tensor(PGVerror)).item()

def plot_pgv(input):
    _, seq_len, height, width = input.size()
    PGV = torch.zeros(height, width)
    for h in range(height):
        for w in range(width):
            PGV[h,w] = torch.max(torch.sqrt(input[0,:,h,w]**2 + input[1,:,h,w]**2))
                   
    return PGV

def plot_pgv_1(input):
    _, seq_len, height, width = input.size()
    PGV = torch.zeros(_, height, width)
    for c in range(_):     
        for h in range(height):
            for w in range(width):
                PGV[c,h,w] = torch.max(torch.abs(input[c,:,h,w]))
                   
    return PGV

def plot_pgv_arrival(input):
    _, seq_len, height, width = input.size()
    PGV = torch.zeros(height, width)
    for h in range(height):
        for w in range(width):
                pgv_value = torch.max(torch.sqrt(input[0,:,h,w]**2 + input[1,:,h,w]**2))
                condition = (torch.sqrt(input[0,:,h,w]**2 + input[1,:,h,w]**2) == pgv_value)
                indtmp = torch.where(condition)[0]
                if len(indtmp)>=1:
                    indtmp = indtmp[0]
                PGV[h,w] = indtmp               
    return PGV

def arrivetime_error(output, target):
    batch_size, _, seq_len, height, width = target.size()
    arrivetime_error = []
    for i in range(batch_size):
        for j in range(_):
            for h in range(height):
                for w in range(width):
                    pgv_true = torch.max(torch.abs(target[i,j,:,h,w])) # my tensor dimension is [batch, channel, time_length]
                    pgv_pred = torch.max(torch.abs(output[i,j,:,h,w])) 
                    condition = (torch.abs(target[i,j,:,h,w]) == pgv_true )
                    indtmp = torch.where(condition)[0]
                    condition = (torch.abs(output[i,j,:,h,w]) == pgv_pred )
                    indpred = torch.where(condition)[0]
                    arrivetime_error.append(torch.abs(indtmp-indpred))
                    
    arrivetime_error_tensor = torch.tensor(arrivetime_error, dtype=torch.float32)
    return torch.mean(arrivetime_error_tensor).item()

def spatial_error(output, target):
    batch_size, _, seq_len, height, width = target.size()
    spatial_error = []
    for i in range(batch_size):
        for j in range(_):
            for t in range(seq_len):
                target_cpu = target[i, j, t, :, :].cpu().detach().numpy()
                output_cpu = output[i, j, t, :, :].cpu().detach().numpy()
                
                if np.max(target_cpu)>1 and np.max(output_cpu)>1:
                    indtmp = np.unravel_index(np.argmax(target_cpu), target_cpu.shape)
                    indpred = np.unravel_index(np.argmax(output_cpu), output_cpu.shape)
                
                    indtmp_tensor = torch.tensor(indtmp, dtype=torch.float32)
                    indpred_tensor = torch.tensor(indpred, dtype=torch.float32)
                
                    spatial_error.append(torch.norm(indtmp_tensor- indpred_tensor).item())

                
    spatial_error_tensor = torch.tensor(spatial_error, dtype=torch.float32)
    return torch.mean(spatial_error_tensor).item()