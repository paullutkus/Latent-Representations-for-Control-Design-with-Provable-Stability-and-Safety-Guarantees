import torch
import torch.nn as nn
from functools import partial

### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"

from configs.base_cfg import *

# SYSTEM
DT = 0.02
d_x = 4
d_z = 3
d_u = 1
# 'cartpole-custom' or 'cartpole-gym'
system = 'cartpole-custom'
enc_true_dyn = False
control_affine = True
symbols = (0, 1)


# DATA
dataset = 'custom-500k.pkl'
num_traj = int(50e4)
traj_len = 10
num_test_traj = int(10e4)


# MODEL
load_name = 'custom_control_affine.pth'
save_name = load_name
d_h = 4
enc_layers = [[d_h, d_x]] + [[d_z, d_h]]
dec_layers = [[d_h, d_z]] + [[d_x, d_h]]
fdyn_drift_layers = [[d_h, d_z]] + [[d_z, d_h]]
fdyn_cntrl_layers = [[d_h, d_z]] + [[d_z*d_u, d_h]]
act = torch.tanh
vae = False
batch_norm = False
residual_ae = True
residual_fdyn_drift = False
residual_fdyn_cntrl = False
res_bias = False
final_act = False
init_ae = nn.init.xavier_uniform_
init_fdyn_drift = nn.init.xavier_uniform_
init_fdyn_cntrl = nn.init.xavier_uniform_
ode = False
ode_method = 'rk4'


# TRAINING
m = 9
lr = 5e-4
batch_size = 512
test_inc = 1
epochs = 100 #400
weight_decay=0.01
ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
fdyn_drift_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
fdyn_cntrl_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
clip_gradient = False
with_replacement = False





