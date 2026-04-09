#
# postprocess_approx_mcmc.r
# This script is intended to be run after `run_approx_mcmc.r` to identify 
# potential issues with MCMC runs, as well as conduct MCMC post-processing.
# This includes computation of diagnostics (R-hat), determiming burn-ins,
# and computing chain weights for nonmixing chains. This script should be
# run before `run_analyze_mcmc.r`.
#
# The main inputs to this script are an experiment tag and round. It then
# processes all MCMC output found in:
# experiments/<experiment_tag>/round_<round>/mcmc
# 
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)
library(tictoc)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Paths
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Experiment and round.
experiment_tag <- "vsem"
round <- 1L

# Settings for defining what constitutes a valid MCMC run. The R-hat threshold
# is for the maximum R-hat over all parameters with an MCMC run (or within
# a single MCMC chain for within-chain diagnostics used for nonmixing chains).
# The min itr threshold is the minimum number of samples a chain is allowed
# to have.
rhat_threshold <- 1.05
min_itr_threshold <- 500L

# `overwrite = TRUE` will re-compute post-processing results for runs already
# found in "group_info.csv" and "mcmc_summary.csv". Otherwise, will only 
# post-process new runs and append to these existing files.
overwrite <- FALSE

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

round_name <- paste0("round", round)

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
mcmc_dir <- file.path(experiment_dir, "output", round_name, "mcmc")

# Paths.
mcmc_ids_path <- file.path(mcmc_dir, "id_map.csv")

# Source required scripts.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(base_dir, "scripts", "helper", "sim_study_functions.r"))

# ------------------------------------------------------------------------------
# Identify MCMC runs to process.
# ------------------------------------------------------------------------------

# Load MCMC ID map. Unique by columns "mcmc_id", "mcmc_tag".
# TODO: I messed up and rows are actually unique by "mcmc_id", "mcmc_tag",
# "em_tag". Need to fix this.
#
# Should also update `process_mcmc_round` to save to file periodically.
mcmc_ids <- fread(mcmc_ids_path)[, .(mcmc_tag, em_tag, em_id, mcmc_id)]
print(paste0("Preparing to process ", nrow(mcmc_ids), " MCMC runs."))


# ------------------------------------------------------------------------------
# Run processing code.
# ------------------------------------------------------------------------------

start_time <- tic()
results <- process_mcmc_round(experiment_dir, round, mcmc_ids, 
                              rhat_threshold=rhat_threshold,
                              min_itr_threshold=min_itr_threshold, 
                              write_files=TRUE)
end_time <- toc()





