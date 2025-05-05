import torch
import torch.nn as nn
from functools import partial

### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"


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
symbols = (0, 1, 2, 3)
enc_true_dyn = False
control_affine = False
linear_state_space = False
linear_state_space_offset = False
neural_dynamics = False
reverse_ae = False
predict_mstep = True
learn_drift = False
learn_residual = False
adversarial_training = False
contrastive_encoder_loss = False
active_learning = False

penalize_rec = True
penalize_reproj = False
penalize_encoder_diagram = False
penalize_encoder_diagram_mstep = False
penalize_decoder_diagram = False
penalize_latent_origin_norm = False
penalize_isotropic_latent = False
penalize_ae_jac_norm = False
penalize_fdyn_jac_norm = False

make_video = False
video_idx = 5
video_end_epoch = 10
plot_eigenvalues = False
plot_jac_norm = False
plot_jac_err = False
latent_space_training_video = False

# DATA
dataset = 'custom-500k.pkl'
num_traj = int(50e4)
traj_len = 10
drift_traj_len = 3
num_test_traj = int(10e4)
dset_size = None
u_range = 25.
x_range_active = torch.pi/3
x_range = 0.
# for grid dset
grid_x_ranges = d_x * [0.5]
grid_u_ranges = d_u * [3]
grid_n_per_axis = 15


dataset_finetune = 'lqr-dataset.pkl'
num_traj_finetune = int(1e4)
traj_len_finetune = 10
num_test_traj_finetune = int(0.2e4)


# MODEL
load_name = 'custom-500k.pth'
save_name = load_name
d_h = 64
n_hidden_ae = 0
n_hidden_fdyn = 0
enc_layers = [[d_h, d_x]] + n_hidden_ae * [[d_h, d_h]] + [[d_z, d_h]]
dec_layers = [[d_h, d_z]] + n_hidden_ae * [[d_h, d_h]] + [[d_x, d_h]]
fdyn_layers = [[d_h, d_z+d_u]] + n_hidden_fdyn * [[d_h, d_h]] + [[d_z, d_h]]
act = torch.tanh
vae = False
batch_norm = False
residual_ae = False
residual_fdyn = False
residual_fdyn_drift = False
residual_fdyn_cntrl = False
res_bias_ae = False
res_bias_fdyn = False
final_act = False
init_ae = nn.init.xavier_uniform_
init_ae_bias = None
init_fdyn = nn.init.xavier_uniform_
init_fdyn_bias = None
init_fdyn_drift_bias = None
init_fdyn_cntrl_bias = None
ode = False
ode_method = 'rk4'


# TRAINING
m = 9
m_schedule = None
lr = 5e-4
batch_size = 512
test_inc = 1
epochs = 50 #400
store_model_freq = 1
weight_decay=0.01
ae_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
fdyn_opt = partial(torch.optim.NAdam, weight_decay=weight_decay)
clip_gradient = False
with_replacement = False
stop_loss = 0.0


# LOSS
lam_rec = 1
lam_reproj = 1
lam_mstep_backwards = 1
lam_drift_backwards = 1
lam_mstep_forwards = 1
lam_drift_forwards = 1
lam_origin = 1
lam_isotropic_latent = 1
lam_active = 1
single_example_loss = False
rec_batch_reduc = torch.sum
reproj_batch_reduc = torch.sum
contrastive_rec_batch_reduc = torch.sum
mstep_batch_reduc = torch.sum
jac_batch_reduc = torch.sum
drift_batch_reduc = torch.sum
jac_norm_ae_batch_reduc = torch.sum
jac_norm_fdyn_batch_reduc = torch.sum
enc_diagram_batch_reduc = torch.sum
dec_diagram_batch_reduc = torch.sum
active_batch_reduc = torch.sum

#drift_loss = False


# JACOBIAN RECONSTRUCTION
rec_jac = False
ptb_jac = False
ptb_x = torch.tensor([0., 0., 0., 0])
ptb_u = torch.tensor([0.])
ptb_eps = 1e-3


