import torch
import torch.nn as nn
from functools import partial

### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"


# SYSTEM
DT = 0.02
d_x = 4
d_z = 2
d_u = 1
# 'cartpole-custom' or 'cartpole-gym'
system = 'cartpole-gym'
enc_true_dyn = False


# DATA
dataset = 'gym-500k.pkl'
num_traj = int(50e4)
traj_len = 10
num_test_traj = int(10e4)


# MODEL
load_name = 'gym-500k.pth'
save_name = load_name
d_h = 32 # dim of hidden layers
n_h = 3 # num of hidden layers
enc_layers = [[d_h, 4]] + n_h * [[d_h, d_h]] + [[2, d_h]]
dec_layers = [[d_h, 2]] + n_h * [[d_h, d_h]] + [[4, d_h]]
fdyn_layers = [[d_h, 3]] + n_h * [[d_h, d_h]] +  [[2, d_h]]
act = torch.tanh
vae = False
batch_norm = False
residual_fdyn = False
residual_ae = False
res_bias = False
final_act = False
init_fdyn = nn.init.kaiming_uniform_ #nn.init.eye_
init_ae = nn.init.kaiming_uniform_
ode = True
ode_method = 'dopri5'


# TRAINING 
'''
m = 9
lr = 5e-4
batch_size = 512
test_inc = 1
epochs = 100
'''
weight_decay=0.0001
ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
fdyn_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)

clip_gradient = True
with_replacement = False

m = 5
lr = 5e-4
batch_size = 32768
test_inc = 1
epochs = 2 # int(10e4)
#ae_opt = torch.optim.NAdam
#fdyn_opt = torch.optim.NAdam
