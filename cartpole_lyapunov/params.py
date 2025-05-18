### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"

#cfg = 'latent_ltv_2d'
#cfg = 'large'
#cfg = 'expert_controller'
#cfg = 'expert_small_dset'
#cfg = 'small_pert_random'
#cfg = 'extra_small_pert_random'
#cfg = 'xxs_pert'
#cfg = 'new_main'
#cfg = 'grid_dset'
#cfg = 'grid_dset2'
cfg = 'invariant_set'


cfg = cfg + "_cfg"
exec ("from" + " configs." + cfg + " import *")

