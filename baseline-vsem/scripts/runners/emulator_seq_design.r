#
# emulator_seq_design.r
#
# This script runs one round of sequential acquisition of design points. It 
# therefore should only be run for rounds >= 2 (i.e., not the initial design
# round). The user specifies:
#  (i.) the experiment ID.
#  (i.) the round.
#  (iii.) An MCMC tag and ID.
#  (iv.) Sequential design settings.
# 
# The round is an integer; e.g., `2` for the first round of sequential 
# acquisition after the initial design round (round 1). In the round 2 example.
# all outputs from this run will be stored in 
# `<experiment_dir>/round2/design/<acq_id>/<mcmc_tag>/<mcmc_id>`. The MCMC 
# tag/ID and acquisition ID are discussed below.
#
# The MCMC tag/ID correspond to an MCMC run from the previous round. Recall that
# MCMC IDs are unique within tags, so both the tag and ID are required to specify
# a unique run from the previous round. This will be used to identify the 
# current emulator that will be updated. It will also be used to read MCMC
# output from the current posterior approximation, which can be used to specify
# candidate points for discrete optimization, and weights used in some
# acquisition criteria that integrate over the design space.
#
# Perhaps the most important sequential design setting is the number of points
# to acquire, which is typically called `n_batch` in the code. Currently,
# this is passed as a command line argument to this script. While it is not
# explicitly enforced, the code is set up so that all sequential design runs 
# within a single round of sequential design will use the same `n_batch` value.
# The other major portion of the settings is the acquisition function settings.
# This is specified by passing and acquisition function ID "acq_id" via a 
# command line argument. The acquisition function ID corresponds to the index
# of an element of the list, 
# `<experiment_dir>/acq_settings_list.rds`, which must
# be saved in advance of running this script.
#
# In all rounds beyond round 1, the design tags in the "design" directory
# correspond to the acquisition function IDs. We make the assumption that 
# the acquisition IDs complete encode the acquisition algorithm, so the 
# only variation across methods is captured by this ID alone. This is similar
# to the convention for the MCMC IDs. Both the MCMC and acquisition settings
# are saved in lists in the base experiment directory and should not be 
# modified throughout the experiment (except perhaps adding new IDs). The
# order of the elements in these lists are used to define the MCMC and 
# acquisition method IDs that are used across all rounds.
#
# Andrew Roberts
#

# TODO:
# - For IEVAR criteria, check the "plugin" argument. I might have to set this to TRUE always.
#      -> Conclusion: only relevant for forward model emulator, so don't worry about it for now.
# - Implement batch heuristics.
# - Implement log-normal adjustments:
#      -> How to do this for IEVAR?
# - Make sure I am passing the "rectified" adjustment to all of the algorithms.
# - Add prior to IEVAR so it targets log posterior density, not just llik. Currently
#   doesn't matter since I'm using uniform priors.
# - Use Stein or support points when sub-sampling.

library(ggplot2)
library(data.table)
library(assertthat)
library(docopt)
library(tictoc)

# ------------------------------------------------------------------------------
# docopt string for parsing command line arguments.  
# ------------------------------------------------------------------------------

"Usage:
  test_docopt.r [options]
  test_docopt.r (-h | --help)

Options:
  -h --help                                 Show this screen.
  --experiment_tag=<experiment_tag>         Used to locate the base output directory.
  --round=<round>                           Integer specifying the round.
  --mcmc_tag=<mcmc_tag>                     MCMC tag from previous round.
  --mcmc_id=<mcmc_id>                       MCMC ID from previous round.
  --n_batch=<n_batch>                       Integer number of points to acquire.
  --acq_id=<acq_id>                         Integer acquisition function ID.
" -> doc

# Read command line arguments.
cmd_args <- docopt(doc)
experiment_tag <- cmd_args$experiment_tag     # Example: experiment_tag <- "vsem"
round <- as.integer(cmd_args$round)           # Example: round <- 2L
mcmc_tag <- cmd_args$mcmc_tag                 # Example: mcmc_tag <- "mcwmh-joint-rect"
mcmc_id <- cmd_args$mcmc_id                   # Example: mcmc_id <- "1008787650"
n_batch <- as.integer(cmd_args$n_batch)       # Example: n_batch <- 4L
acq_id <- as.integer(cmd_args$acq_id)         # Example: acq_id <- 1L

print(paste0("experiment_tag: ", experiment_tag))
print(paste0("round: ", round))
print(paste0("mcmc_tag: ", mcmc_tag))
print(paste0("mcmc_id: ", mcmc_id))
print(paste0("n_batch: ", n_batch))
print(paste0("acq_id: ", acq_id))

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

seed <- sample.int(.Machine$integer.max, size=1L)
print(paste0("Setting seed: ", seed))
set.seed(seed)

# Base directory: all paths are relative to this directory.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Directory to source code.
src_dir <- file.path(base_dir, "src")

# Define directories.
prev_round <- round - 1L
assert_that(prev_round > 0)
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
inv_prob_dir <- file.path(experiment_dir, "inv_prob_setup")
round_dir <- file.path(experiment_dir, paste0("round", round))
design_dir <- file.path(round_dir, "design")
acq_dir <- file.path(design_dir, paste0("acq_", acq_id))
mcmc_base_input_dir <- file.path(experiment_dir, paste0("round", prev_round),
                                 "mcmc", mcmc_tag)
mcmc_input_dir <- file.path(mcmc_base_input_dir, mcmc_id)
out_dir <- file.path(acq_dir, mcmc_tag, mcmc_id) # Outputs from this file saved here.

# File path to acquisition settings.
acq_settings_path <- file.path(experiment_dir, "acq_settings_list.rds")

print(paste0("MCMC input dir (from previous round): ", mcmc_input_dir))
print(paste0("Output directory: ", out_dir))
print(paste0("Acq settings path: ", acq_settings_path))

if(!file.exists(acq_settings_path)) {
  stop("Acquisition settings path does not exist.")
}

# Source required scripts.
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
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "seq_design_gp.r"))
source(file.path(src_dir, "seq_design_for_post_approx.r"))
source(file.path(src_dir, "sim_study_functions.r"))

# Create output directory.
dir.create(out_dir, recursive=TRUE)

# ------------------------------------------------------------------------------
# Load inverse problem setup and exact MCMC samples.
#    Exact prior and posterior samples are used as validation points for 
#    emulators within the sequential design algorithm. They are not used for 
#    candidate/grid points.
# ------------------------------------------------------------------------------

inv_prob <- readRDS(file.path(inv_prob_dir, "inv_prob_list.rds"))
test_info_prior <- readRDS(file.path(inv_prob_dir, "test_info_prior.rds"))
test_info_post <- readRDS(file.path(inv_prob_dir, "test_info_post.rds"))
llik_func <- inv_prob$llik_obj$get_llik_func()

# ------------------------------------------------------------------------------
# Load acquisition function settings and extract specific acq function.
# ------------------------------------------------------------------------------

# The acquisition ID is the index in the acquisition settings list.
acq_settings <- readRDS(acq_settings_path)[[acq_id]]

print("Using the following acquisition function:")
print(paste0("Acq ID: ", acq_id))
print(acq_settings)

# ------------------------------------------------------------------------------
# Load llikEmulator object.
#   This is the current llikEmulator that we seek to augment with new design
#   points. The MCMC tag and ID map uniquely to the emulator ID, which is then 
#   used to load the emulator. Also loads the design tag/ID corresponding to
#   the initial design used to train the emulator.
# ------------------------------------------------------------------------------

# Load the MCMC ID mapping file for the specified MCMC tag. Extract the 
# emulator tag/ID.
print("-----> Loading emulator tag and ID")
id <- mcmc_id
mcmc_id_map <- fread(file.path(mcmc_base_input_dir, "id_map.csv"))
mcmc_id_map <- mcmc_id_map[mcmc_id == id]
if(nrow(mcmc_id_map) != 1L) {
  stop("After subsetting ID map using mcmc_id ", mcmc_id, " the resulting table", 
       " has ", nrow(mcmc_id_map), " rows. Should have exactly one.")
}

em_tag <- mcmc_id_map$em_tag
em_id <- mcmc_id_map$em_id
print(paste0("Emulator tag: ", em_tag))
print(paste0("Emulator ID: ", em_id))
 
# Load the emulator.
base_em_dir <- file.path(experiment_dir, paste0("round", prev_round), "em", em_tag)
em_path <- file.path(base_em_dir, em_id, "em_llik.rds")
print(paste0("Loading emulator from path: ", em_path))
em_llik <- readRDS(em_path)


# ------------------------------------------------------------------------------
# Set up metrics to track within sequential design loop.
#   Allows computing evaluations metrics within the sequential design loop
#   every `interval` iterations, as specified below.
# ------------------------------------------------------------------------------

tracking_settings <- list(interval=10L, func_list=list())

tracking_settings$func_list$pw_prior <- function(model) {
  pred <- model$emulator_model$calc_pred_multi_func(list(crps="crps", mse="mse"),
                                                    type="pw", X_new=test_info_prior$input, 
                                                    Y_new=matrix(test_info_prior$llik, ncol=1L))
  pred[, .(mean=mean(y1)), by=func]
}

tracking_settings$func_list$pw_post <- function(model) {
  pred <- model$emulator_model$calc_pred_multi_func(list(crps="crps", mse="mse"),
                                                    type="pw", X_new=test_info_post$input, 
                                                    Y_new=matrix(test_info_post$llik, ncol=1L))
  pred[, .(mean=mean(y1)), by=func]
}

tracking_settings$func_list$agg_prior <- function(model) {
  model$emulator_model$calc_pred_func("log_score", type="agg", 
                                      X_new=test_info_prior$input, 
                                      Y_new=matrix(test_info_prior$llik, ncol=1L),
                                      return_cov=TRUE)
}

tracking_settings$func_list$agg_post<- function(model) {
  model$emulator_model$calc_pred_func("log_score", type="agg", 
                                      X_new=test_info_post$input, 
                                      Y_new=matrix(test_info_post$llik, ncol=1L),
                                      return_cov=TRUE)
}


# ------------------------------------------------------------------------------
# Candidate and Grid Points
#
#   TODO: ignoring below text for now and simply re-generating the point batches
#         every time this file is called. Should talk to Jonathan about best
#         approach here.
#   
#   Candidate points is a discrete set of points from which the new batch of
#   design points are selected. Grid points is a set of points used for 
#   numerical integration over the design points for integrated uncertainty 
#   type acquisition functions. We impose the requirement that both of these
#   sets of points should be the same within a unique combination of 
#   {round, mcmc_tag, mcmc_id}. i.e., across all acquisition methods that are
#   updating the same current emulator model. This is to prevent deviations
#   in results stemming only from good or bad luck in the random selection of
#   candidate/grid points sets. This restriction is implemented below by
#   only creating new candidate/grid point sets if current sets have not 
#   already been created and saved to file. These files are saved in a separate
#   directory called "grid_points". Note that the grid points sets also need
#   to vary by mcmc_id given that they may be constructed by sub-sampling
#   the previous MCMC output.
# ------------------------------------------------------------------------------

par_names <- inv_prob$par_names

# Candidate grid: finite set of points that will be optimized over. Not 
# necessarily required by all acquisition functions. 
candidate_grid <- NULL
candidate_settings <- acq_settings$candidate_settings

if(!is.null(candidate_settings)) {
  
  # Prior samples.
  if(isTRUE(candidate_settings$n_prior > 0)) {
    candidate_grid <- sample_prior(inv_prob$par_prior, 
                                   n=candidate_settings$n_prior)[,par_names]
  }
  
  # Previous posterior samples.
  if(isTRUE(candidate_settings$n_mcmc > 0)) {
    candidate_grid_mcmc <- load_samp_mat(experiment_dir, prev_round, mcmc_tag, 
                                         mcmc_id, only_valid=TRUE, 
                                         n_subsamp=candidate_settings$n_mcmc)[,par_names]
    
    if(is.null(candidate_grid)) candidate_grid <- candidate_grid_mcmc
    else candidate_grid <- rbind(candidate_grid, candidate_grid_mcmc)
  }
}

if(!is.null(candidate_grid)) {
  candidate_grid_path <- file.path(out_dir, "candidate_grid.rds")
  print(paste0("Writing candidate grid to file: ", candidate_grid_path))
  saveRDS(candidate_grid, candidate_grid_path)
}


# Integration grid points: only used for acquisition functions that require
# approximation of an integral over the design space.
weight_grid <- NULL
int_grid_settings <- acq_settings$int_grid_settings

if(!is.null(int_grid_settings)) {
  # Prior samples.
  if(isTRUE(int_grid_settings$n_prior > 0)) {
    weight_grid <- sample_prior(inv_prob$par_prior, 
                                int_grid_settings$n_prior)[,par_names]
  }
  
  # Previous posterior samples.
  if(isTRUE(int_grid_settings$n_mcmc > 0)) {
    weight_grid_mcmc <- load_samp_mat(experiment_dir, prev_round, mcmc_tag, 
                                      mcmc_id, only_valid=TRUE, 
                                      n_subsamp=int_grid_settings$n_mcmc)[,par_names]
    
    if(is.null(weight_grid)) weight_grid <- weight_grid_mcmc
    else weight_grid <- rbind(weight_grid, weight_grid_mcmc)
  }
}

if(!is.null(weight_grid)) {
  weight_grid_path <- file.path(out_dir, "weight_grid.rds")
  print(paste0("Writing weight grid to file: ", weight_grid_path))
  saveRDS(weight_grid, weight_grid_path)
}


# ------------------------------------------------------------------------------
# Run Sequential Design
# ------------------------------------------------------------------------------

# Arguments for `run_seq_design`.
args <- list(model=em_llik, n_batch=n_batch, opt_method="grid", 
             true_func=llik_func, reoptimize_hyperpar=FALSE,
             candidate_grid=candidate_grid, tracking_settings=tracking_settings,
             grid_points=weight_grid)

# Append acquisition function settings to function arguments list.
args$acq_func_name <- acq_settings$name
args <- c(args, acq_settings$acq_args)
args$response_heuristic <- acq_settings$response_heuristic
if(is.na(args$response_heuristic)) args$response_heuristic <- NULL

# Run sequential design loop.
tic()
acq_results <- do.call(run_seq_design, args)
tictoc_info <- toc()

# Store additional information to save.
acq_results$runtime <- tictoc_info$toc - tictoc_info$tic
acq_results$acq_settings <- acq_settings
acq_results$seed <- seed

# Save results to file. Files are named by their acquisition ID.
out_path <- file.path(out_dir, "acq_results.rds")
print(paste0("Saving: ", out_path))
saveRDS(acq_results, out_path)



  
  