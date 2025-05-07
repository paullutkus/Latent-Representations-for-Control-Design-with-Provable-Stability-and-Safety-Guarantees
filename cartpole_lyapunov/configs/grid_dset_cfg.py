import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"

from configs.base_cfg import *


# DEBUG 
debug = False
run_tests = True
tqdm_each_epoch = False


# SYSTEM
DT = 0.02
d_x = 4
d_z = 2
d_u = 1
# 'cartpole-custom' or 'cartpole-gym'
system = 'cartpole-custom'
enc_true_dyn = False
control_affine = False
linear_state_space = True ### TURN BACK ON AFTER TEST REVERSE AE ###
linear_state_space_offset = False
neural_dynamics = False
reverse_ae = False
adversarial_training = False
contrastive_encoder_loss = False
learn_residual = False
learn_drift = True
active_learning = False

#predict_mstep_schedule = [True, False, True]
predict_mstep = True

#penalize_rec_schedule = [True, True, True]
penalize_rec = True

#penalize_reproj_schedule = [False, True, True]
penalize_reproj = True

#penalize_encoder_diagram_mstep_schedule = [False, True, True]
penalize_encoder_diagram_mstep = True

penalize_latent_origin_norm = True
penalize_isotropic_latent = True
penalize_encoder_diagram = False
penalize_decoder_diagram = False
penalize_ae_jac_norm = False
penalize_fdyn_jac_norm = False

make_video = False
video_idx = 5
video_end_epoch = 10
plot_eigenvalues = False
plot_jac_norm = False
plot_jac_err = False
latent_space_training_video = False #True
symbols = (0, 1, 3)#(0, 1, 3)
ignored = (2,)


# DATA
#dataset = 'grid_dset_2.pkl'
dataset = 'grid_dset_T16.pkl'

#dataset = 'custom-drift_match-500k.pkl'
#dataset = 'u30-T16-5e5.pkl' #'gym-1step-5million.pkl'
#dataset = 'main-T4-u4-1e6.pkl'
#dataset = 'u8-T25-5e4.pkl'
#dataset = 'u10-T28-5e4.pkl'
grid_x_ranges = d_x * [0.5]
grid_u_ranges = d_u * [3]
grid_n_per_axis = 15
num_traj = int(5e4)#int(1e6)
traj_len = 16 #7
num_test_traj = int(5e3)#int(2e5)
x_range=0.
x_range_active=torch.pi/6
u_range=3.#10.

dataset_finetune = 'lqr-dataset.pkl'
num_traj_finetune = int(1e4)
traj_len_finetune = 10
num_test_traj_finetune = int(0.2e4)




# MODEL
load_name = 'ltv_2d.pth'
save_name = load_name
d_h = 64 #64 #128
n_hidden_ae = 0#0 
n_hidden_fdyn = 0#0
enc_layers = [[d_h, d_x]] + n_hidden_ae * [[d_h, d_h]] + [[d_z, d_h]]
dec_layers = [[d_h, d_z]] + n_hidden_ae * [[d_h, d_h]] + [[d_x, d_h]]
fdyn_layers = [[d_h, d_z+d_u]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z, d_h]]
fdyn_drift_layers = [[d_h, d_z]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z*d_z, d_h]]
fdyn_offset_layers = [[d_h, d_z]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z, d_h]]
fdyn_cntrl_layers = [[d_h, d_z]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z*d_u, d_h]]
act = torch.tanh # try silu and gelu
vae = False
batch_norm = False
residual_ae = False #True
residual_fdyn = False
residual_fdyn_drift = False
residual_fdyn_offset = False
residual_fdyn_cntrl = False
res_bias_ae = True
res_bias_fdyn = True
final_act = False
init_ae = nn.init.xavier_uniform_ # try kaiming 
init_ae_bias = None
init_fdyn = nn.init.xavier_uniform_ # try kaiming
init_fdyn_bias = None
init_fdyn_drift = nn.init.xavier_uniform_ #nn.init.xavier_normal_
init_fdyn_offset = nn.init.xavier_uniform_
init_fdyn_cntrl = nn.init.xavier_uniform_ #nn.init.xavier_normal_
init_fdyn_drift_bias = nn.init.uniform_ #nn.init.normal_
init_fdyn_offset_bias = nn.init.zeros_
init_fdyn_cntrl_bias = nn.init.zeros_
ode = False
ode_method = 'rk4'


# TRAINING
m = traj_len - 1
#m_schedule = [10, 30, 50, -1] #50
m_schedule = m*[0]

lr = 1e-3 #5e-4, 1e-3 if mean loss
batch_size = 512 #512 #16384 #32768
store_model_freq = 2 #3 #15
epochs = 100 #300 #1500
test_inc = epochs-1
weight_decay=0.00
ae_opt = torch.optim.AdamW
fdyn_opt = torch.optim.AdamW
fdyn_drift_opt = torch.optim.AdamW
fdyn_offset_opt = torch.optim.AdamW
fdyn_cntrl_opt = torch.optim.AdamW
clip_gradient = False
with_replacement = False
stop_loss = 0.0


# LOSS
lam_mstep_backwards = 1
lam_drift_backwards = 1
lam_mstep_forwards = 2
lam_drift_forwards = 2

rec_batch_reduc = torch.mean
reproj_batch_reduc = torch.mean
mstep_batch_reduc = torch.mean
drift_batch_reduc = torch.mean
enc_diagram_batch_reduc = torch.mean
active_batch_reduc = torch.mean

contrastive_rec_batch_reduc = torch.mean
dec_diagram_batch_reduc = torch.mean
jac_norm_ae_batch_reduc = torch.mean
jac_norm_fdyn_batch_reduc = torch.mean
jac_batch_reduc = torch.mean


# JACOBIAN RECONSTRUCTION
rec_jac = False
ptb_jac = False
ptb_x = torch.tensor([0., 0., 0., 0])
ptb_u = torch.tensor([0.])
ptb_eps_x = 1e-7
ptb_eps_u = 1e-9

# ACTIVE LEARNING 
#mix_start = epochs//8
#mix_end = epochs//5
mix_len = 50
