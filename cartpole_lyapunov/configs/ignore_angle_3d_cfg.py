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
control_affine = False
symbols = (0, 1, 3)
video_idx = 3
video_end_epoch = 10
learn_drift = False


# DATA
dataset = 'custom-500k-w-drift.pkl'
num_traj = int(50e4)
traj_len = 10
drift_traj_len = 3
num_test_traj = int(10e4)


# MODEL
load_name = 'ignore_cart_pos.pth'
save_name = load_name
d_h = 64
n_hidden = 0
enc_layers = [[d_h, d_x]] + n_hidden * [[d_h, d_h]] + [[d_z, d_h]]
dec_layers = [[d_h, d_z]] + n_hidden * [[d_h, d_h]] + [[d_x, d_h]]
fdyn_layers = [[d_h, d_z+d_u]] + n_hidden * [[d_h, d_h]] + [[d_z, d_h]]
fdyn_drift_layers = [[d_h, d_z]] + n_hidden * [[d_h, d_h]] + [[d_z, d_h]]
fdyn_cntrl_layers = [[d_h, d_z]] + n_hidden * [[d_h, d_h]] + [[d_z*d_u, d_h]]
act = torch.tanh
vae = False
batch_norm = False
residual_ae = False #True
residual_fdyn = False
residual_fdyn_drift = True
residual_fdyn_cntrl = False
res_bias_ae = True
res_bias_fdyn = True
final_act = False
init_ae = nn.init.xavier_uniform_
init_fdyn = nn.init.xavier_uniform_
init_fdyn_drift = nn.init.xavier_uniform_
init_fdyn_cntrl = nn.init.xavier_uniform_
init_fdyn_drift_bias = nn.init.zeros_ #nn.init.normal_ #slowing convergence
init_fdyn_cntrl_bias = nn.init.zeros_ #nn.init.normal_ #slowing convergence


ode = False
ode_method = 'rk4'


# LOSS
rec_batch_reduc = torch.sum
mstep_batch_reduc = torch.sum


# TRAINING
m = 9
lr = 5e-4 #5e-4
batch_size = 512#16384 #512
test_inc = 1
epochs = 10 #400
weight_decay=0.01
#ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
ae_opt = torch.optim.AdamW
fdyn_opt = torch.optim.AdamW
fdyn_drift_opt = torch.optim.AdamW
fdyn_cntrl_opt = torch.optim.AdamW
#fdyn_drift_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
#fdyn_cntrl_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
clip_gradient = False
with_replacement = False





