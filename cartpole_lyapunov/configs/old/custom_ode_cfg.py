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
system = 'cartpole-custom'
enc_true_dyn = False


# DATA
dataset = 'custom-500k.pkl'
num_traj = int(50e4)
traj_len = 10
num_test_traj = int(10e4)


# MODEL
load_name = 'custom-ode-500k.pth'
save_name = load_name
enc_layers = [[9, 4]] + [[6, 9]] + [[6, 6]] + [[2, 6]]
dec_layers = [[6, 2]] + [[6, 6]] + [[9, 6]] + [[4, 9]]
fdyn_layers = [[6, 3]] + 2*[[6, 6]] + [[2, 6]]
act = torch.tanh
vae = False
batch_norm = False
residual_ae = True
residual_fdyn = True
res_bias = True
final_act = False
init_ae = nn.init.xavier_uniform_
init_fdyn = nn.init.xavier_uniform_
ode = True
ode_method = 'rk4'


# TRAINING
m = 8
lr = 1e-3 #5e-4
batch_size = 32768
test_inc = 1
epochs = 100
weight_decay = 0.00#0.01
ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
fdyn_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
clip_gradient = False
with_replacement = False




