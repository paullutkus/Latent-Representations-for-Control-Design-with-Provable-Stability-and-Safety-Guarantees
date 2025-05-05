import torch
import torch.nn as nn
import torch.nn.functional as F
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
linear_state_space = True
learn_drift = True
neural_dynamics = False
learn_residual = False
symbols = (0, 1, 3)
make_video = True
video_idx = 5
video_end_epoch = 10
plot_eigenvalues = True
plot_jac_norm = True
plot_jac_err = False
run_tests = True


# DATA
dataset = 'custom-drift_match-500k.pkl' #'gym-1step-5million.pkl'
num_traj = int(5e5)
traj_len = 10
num_test_traj = int(1e5)


# MODEL
load_name = 'ltv.pth'
save_name = load_name
d_h = 64
n_hidden_ae = 0 
n_hidden_fdyn = 0
enc_layers = [[d_h, d_x]] + n_hidden_ae * [[d_h, d_h]] + [[d_z, d_h]]
dec_layers = [[d_h, d_z]] + n_hidden_ae * [[d_h, d_h]] + [[d_x, d_h]]
fdyn_layers = [[d_h, d_z+d_u]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z, d_h]]
fdyn_drift_layers = [[d_h, d_z]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z*d_z, d_h]]
fdyn_cntrl_layers = [[d_h, d_z]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z*d_u, d_h]]
act = torch.tanh # try silu and gelu
vae = False
batch_norm = False
residual_ae = False #True
residual_fdyn = False
residual_fdyn_drift = False
residual_fdyn_cntrl = False
res_bias_ae = True
res_bias_fdyn = True
final_act = False
init_ae = nn.init.xavier_uniform_ # try kaiming 
init_ae_bias = None
init_fdyn = nn.init.xavier_uniform_ # try kaiming
init_fdyn_bias = None
init_fdyn_drift = nn.init.xavier_uniform_
init_fdyn_cntrl = nn.init.xavier_uniform_
init_fdyn_drift_bias = nn.init.normal_
init_fdyn_cntrl_bias = nn.init.normal_
ode = False
ode_method = 'rk4'


# TRAINING
m = 9
lr = 5e-4 #5e-4 
batch_size = 512 #16384 #32768
test_inc = 1
epochs = 20 #400
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
stop_loss = 0.0


# LOSS
rec_batch_reduc = torch.sum
mstep_batch_reduc = torch.sum
jac_batch_reduc = torch.mean
drift_batch_reduc = torch.sum


# JACOBIAN RECONSTRUCTION
rec_jac = False
ptb_jac = False
ptb_x = torch.tensor([0., 0., 0., 0])
ptb_u = torch.tensor([0.])
ptb_eps_x = 1e-7
ptb_eps_u = 1e-9



