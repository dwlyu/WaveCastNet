import numpy as np
import torch
from torch import nn
def validation_rfne(output,target):
    batch_size, _, seq_len, height, width = target.size() # [b,c,t,h,w]
    rfne = []
    for i in range(batch_size):
        for j in range(_):
            for t in range(seq_len):
                err_rfne = torch.norm((target[i,j,t]-output[i,j,t])) / torch.norm(target[i,j,t])
                rfne.append(err_rfne)
    rfne1 = torch.mean(torch.tensor(rfne)).item()
    return rfne1

def validation_rmse(output,target):
    batch_size, _, seq_len, height, width = target.size()
    criterion = torch.nn.MSELoss(reduction='mean')
    rmse = []
    for i in range(batch_size):
        for j in range(_):
            for t in range(seq_len):
                err_rfne = torch.sqrt(criterion(output[i,j,t].flatten(),target[i,j,t].flatten()))
                rmse.append(err_rfne)
    return torch.mean(torch.tensor(rmse)).item()

def validation_acc(output,target):
    batch_size, _, seq_len, height, width = target.size()
    acc = []
    for i in range(batch_size):
        for j in range(_):
            for t in range(seq_len):
                sum1 = torch.sum((output[i,j,t]*target[i,j,t]).flatten())
                sum2 = torch.sum((target[i,j,t]*target[i,j,t]).flatten())
                sum3 = torch.sum((output[i,j,t]*output[i,j,t]).flatten())
                snapshot_acc = sum1/torch.sqrt(sum2*sum3)
                acc.append(snapshot_acc)
    return torch.mean(torch.tensor(acc)).item()
    