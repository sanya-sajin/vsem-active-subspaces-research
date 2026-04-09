#
# init_emulator.r
# This script is intended to be called by `init_emulator.sh`, which is in turn
# intended to be called by the runner script `run_init_emulator.r`. Currently
# this workflow is set up so that each (design tag, emulator tag) combination
# is handled by a separate remote node on the cluster. This means that all 
# design replicates are fit for the (design tag, emulator tag) combination 
# on a single node. Currently this is fine, but if the model fitting process
# becomes more expensive, then it will be more efficient to batch out the 
# design IDs across different nodes as well. Note that this is the initial 
# emulator fitting round, so this file is only used for "round 1".
#
# Fit emulators are saved to 
# experiments/<experiment_tag>/round1/em/<em_tag>/<design_tag>/design_<design_id>/<em_id>/em_llik.rds
# Recall that emulator IDs are defined to be unique within each emulator tag.
# The `id_map.csv` is saved to experiments/<experiment_tag>/round1/em/
# contains the columns "em_tag", "em_id", "design_tag", "design_id", "seed". 
# This provides information on which design was used to fit a specific emulator, 
# as well as the random seed that was used in the fitting.
#
# It is assumed that the `id_map.csv` file has been saved before this script
# is run. This file is used to identify the design IDs to run, 
# and the em_ids determine where the outputs are saved. Note that `id_map.csv`
# is saved by the runner script `run_init_emulator.r`.
#
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)
library(docopt)

# -----------------------------------------------------------------------------
# docopt string for parsing command line arguments.  
# -----------------------------------------------------------------------------

"Usage:
  init_emulator.r [options]
  init_emulator.r (-h | --help)

Options:
  -h --help                                 Show this screen.
  --experiment_tag=<experiment_tag>         The experiment tag.
  --em_tag=<em_tag>                         The emulator tag.
  --design_tag=<design_tag>                 The initial design tag.
" -> doc

# Read command line arguments.
cmd_args <- docopt(doc)
experiment_tag <- cmd_args$experiment_tag
em_tag <- cmd_args$em_tag
design_tag <- cmd_args$design_tag

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

print("--------------------Running `init_emulator.r` --------------------")
print(paste0("Experiment tag: ", experiment_tag))
print(paste0("Emulator tag: ", em_tag))
print(paste0("Design tag: ", design_tag))

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
pecan_dir <- file.path(base_dir, "..", "sipnet_calibration", "src")
src_dir <- file.path(code_dir, "src")
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
setup_dir <- file.path(experiment_dir, "output", "inv_prob_setup")
design_dir <- file.path(experiment_dir, "output", "round1", "design", design_tag)
base_out_dir <- file.path(experiment_dir, "output", "round1", "em", em_tag, design_tag)
em_settings_path <- file.path(experiment_dir, "output", "alg_settings", "em_settings.rds")
em_ids_path <- file.path(experiment_dir, "output", "round1", "em", "id_map.csv")

print(paste0("Using emulator settings: ", em_settings_path))
em_settings <- readRDS(em_settings_path)

print(paste0("Using em ids: ", em_ids_path))
em_ids <- fread(em_ids_path)

print(paste0("Creating output directory: ", base_out_dir))
dir.create(base_out_dir, recursive=TRUE)

# Source required files.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "inv_prob_test_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "gp_helper_functions.r"))
source(file.path(src_dir, "gpWrapper.r"))
source(file.path(src_dir, "llikEmulator.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(src_dir, "gp_mcmc_functions.r"))
source(file.path(src_dir, "basis_function_emulation.r"))
source(file.path(pecan_dir, "prob_dists.r"))

# Load inverse problem setup information.
inv_prob <- readRDS(file.path(setup_dir, "inv_prob_list.rds"))

# ------------------------------------------------------------------------------
# Select design IDs that will be used to fit emulators.
# ------------------------------------------------------------------------------

d_tag <- design_tag
e_tag <- em_tag
em_ids <- em_ids[(design_tag==d_tag) & (em_tag==e_tag)]

if(anyDuplicated(em_ids$design_id)) {
  stop("Found duplicate design IDs. Design ID should be unique within the design tag.")
}


# ------------------------------------------------------------------------------
# Fit emulators, and save results to file.
# ------------------------------------------------------------------------------

print("-------------- Fitting emulators --------------")

for(i in 1:nrow(em_ids)) {
  
  # Locate the correct design.
  design_id <- em_ids[i, design_id]
  em_id <- em_ids[i, em_id]
  seed <- em_ids[i, seed]
  print(paste0("-----> Design ID: ", design_id))
  print(paste0("Seed: ", seed))
  design_name <- paste0("design_", design_id)
  design_path <- file.path(design_dir, design_name, "design_info.rds")
  print(paste0("Loading design: ", design_path))
  design_info <- readRDS(design_path)
  
  # Create output directory.
  save_dir <- file.path(base_out_dir, design_name, paste0("em_", em_id))
  dir.create(save_dir, recursive=TRUE)
  print(paste0("Output dir: ", save_dir))
  
  # Fit and save emulator.
  set.seed(seed)
  em_llik <- em_settings[[em_tag]]$fit_em(design_info, inv_prob)
  saveRDS(em_llik, file=file.path(save_dir, "em_llik.rds"))
}

