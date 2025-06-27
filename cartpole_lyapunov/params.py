### LOGIC FOR CONFIGS
# create <name>.config
# in this params.py use, "from <name> import *"

# chose config from 'configs' directory
cfg = 'invariant_set'

# import all variables from cfg into this file
cfg = cfg + "_cfg"
exec ("from" + " configs." + cfg + " import *")

