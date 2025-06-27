import torch
import torch.nn as nn
from functools import partial

### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"


# SYSTEM
d_x = 4
d_z = 2
d_u = 1
# 'cartpole-custom' or 'cartpole-gym'
system = 'cartpole-gym'


# DATA
dataset = 'gym-500k.pkl'
num_traj = int(50e4)
traj_len = 10
num_test_traj = int(10e4)


# MODEL
load_name = 'gym-500k.pth'
save_name = load_name
enc_layers = [[9, 4]] + [[6, 9]] + [[6, 6]] + [[2, 6]]
dec_layers = [[6, 2]] + [[6, 6]] + [[9, 6]] + [[4, 9]]
fdyn_layers = [[10, 3]] + 10 *[[10, 10]] + [[2, 10]]
act = torch.tanh
vae = False
batch_norm = False
residual_fdyn = True
residual_ae = True
res_bias = True
final_act = True
init_fdyn = nn.init.xavier_normal_ #nn.init.eye_
init_ae = nn.init.xavier_normal_


# TRAINING 
'''
m = 9
lr = 5e-4
batch_size = 512
test_inc = 1
epochs = 100
'''
#weight_decay=0.0001
#ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
#fdyn_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)

clip_gradient = True
with_replacement = True

m = 9
lr = 10
batch_size = 128 # 2048
test_inc = 1
epochs = 10 # int(10e4)
ae_opt = torch.optim.NAdam
fdyn_opt = torch.optim.NAdam
