#
# run_save_alg_settings.r
# Loads the script `experiments/<experiment_tag>/script/save_alg_settings.r`
# for a specific experiment, which defines the functions `get_emulator_settings()`,
# `get_mcmc_settings()` and `get_design_settings()`. These functions are 
# called to obtain the lists defining the emulator models, MCMC algorithms,
# and design criteria that will be used throughout the experiment. These 
# lists are saved to file in `experiments/<experiment_tag>/output/alg_settings`.
# This script is intended to be run a single time at the beginning of an 
# experiment, though one can append new settings to the saved lists at a later
# time. The important thing is not to change the labels stored in the list 
# which serve as unique identifiers for the various algorithms.
#

# ------------------------------------------------------------------------------
# Settings.
# ------------------------------------------------------------------------------

experiment_tag <- "banana"
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")


# ------------------------------------------------------------------------------
# Set up filepaths.
# ------------------------------------------------------------------------------

experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
alg_definition_script <- file.path(experiment_dir, "scripts", "save_alg_settings.r")
out_dir <- file.path(experiment_dir, "output", "alg_settings")

dir.create(out_dir, recursive=TRUE)

# ------------------------------------------------------------------------------
# Load settings and write to file.
# ------------------------------------------------------------------------------

# Source script with settings functions.
source(alg_definition_script)

# Load settings lists.
init_design_settings <- get_init_design_settings()
em_settings <- get_emulator_settings()
mcmc_settings <- get_mcmc_settings()
acq_settings <- get_acq_settings()

# Save to file.
saveRDS(init_design_settings, file.path(out_dir, "init_design_settings.rds"))
saveRDS(em_settings, file.path(out_dir, "em_settings.rds"))
saveRDS(mcmc_settings, file.path(out_dir, "mcmc_settings.rds"))
saveRDS(acq_settings, file.path(out_dir, "acq_settings.rds"))



