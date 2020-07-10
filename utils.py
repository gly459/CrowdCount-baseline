import h5py
import torch
import shutil
from torch import nn
import torch.nn.functional as F

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='checkpoint.pth.tar'):
    save_path = './saved_models/'
    torch.save(state, save_path+task_id+filename)
    if is_best:
        shutil.copyfile(save_path+task_id+filename, save_path+task_id+'model_best.pth.tar')            
