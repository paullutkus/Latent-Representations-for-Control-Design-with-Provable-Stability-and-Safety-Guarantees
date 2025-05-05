import torch
import torch.nn as nn
from functools import partial

from configs.base_cfg import *

# system
system = 'cartpole-custom'

# data
dataset = 'custom-T13-500k.pkl'
num_traj = int(5e3)
traj_len = 13
num_test_traj = int(1e3)
dset_size = 5

# model
load_name = 'ode_test.pth'
save_name = load_name
enc_layers = [[9, 4]] + [[6, 9]] + [[6, 6]] + [[2, 6]]
dec_layers = [[6, 2]] + [[6, 6]] + [[9, 6]] + [[4, 9]]
fdyn_layers = [[3, 3]] + [[2, 3]]
act = torch.tanh
vae = False
batch_norm = False
residual_ae = False
residual_fdyn = False
res_bias = False
final_act = False
init_ae = nn.init.xavier_uniform_
init_fdyn = nn.init.xavier_uniform_

ode = True
ode_method = 'dopri8'

# training
m = 4
lr = 1e-3
epochs=1000
test_inc = 5
ae_opt = torch.optim.AdamW
fdyn_opt = torch.optim.AdamW

# loss
single_example_loss = True

