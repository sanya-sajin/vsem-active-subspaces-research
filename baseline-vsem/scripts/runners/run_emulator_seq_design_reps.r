# run_emulator_seq_design_reps.r
# 
# Controls the dispatching of jobs for sequential design runs. Executes
# `run_emulator_seq_design.sh` once per run, which in turn runs
# `emulator_seq_design.r`. The experiment and round are fixed at the beginning
# of the script, so everything discussed here is with respect to the 
# selected experiment and round. Each run is then defined by a unique 
# combination of acquisition ID, MCMC tag, and MCMC ID, where the latter two 
# uniquely define an MCMC run from the previous round. Note that the MCMC ID
# maps to a unique emulator ID from the previous round as well. The set of 
# runs executed by this script is determined indirectly by specifying the
# a design tag, emulator tag, a set of MCMC tags, and set of acquisition IDs. 
# The design tag, emulator tag, and MCMC tags (which are tags from the previous 
# round) are used to select the set of MCMC IDs from the previous round. 
# Each one of these MCMC runs is then run using each of the specified 
# acquisitions. The acquisition IDs are stored in a list of lists, where the
# sub-lists are used in the job batching; acq IDs in the same sub-list will 
# be run on the same jobs.
# TODO: this actually isn't implemented yet; need to implement batching of jobs.
#
# In addition to the above settings, the user must specify the number of 
# points to acquire in the round of sequential design, which will be held
# constant for all runs.

library(data.table)
library(assertthat)

# Settings determining which runs will be executed.
experiment_tag <- "vsem"
round <- 2L
design_tag_prev <- "LHS_200"
em_tag_prev <- "llik_quad_mean"
mcmc_tags_prev <- c("mcwmh-joint-rect", "marginal-rect", "mean-rect")
acq_ids <- c(12L) # list(c(11L, 12L), 1L, 2L, 3L)
n_batch <- 100L


# ------------------------------------------------------------------------------
# Set up directories/filepaths.
# ------------------------------------------------------------------------------

# Base directory: all paths are relative to this directory.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Path to bash file to run.
bash_path <- file.path(base_dir, "scripts", "gp_post_approx_paper", "sim_study",
                       "bash_files", "run_emulator_seq_design.sh")
assert_that(file.exists(bash_path))

# Define directories.
prev_round <- round - 1L
assert_that(prev_round > 0)
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)
prev_round_dir <- file.path(experiment_dir, paste0("round", prev_round))
mcmc_dir_prev <- file.path(prev_round_dir, "mcmc")
em_dir_prev <- file.path(prev_round_dir, "em")

# ------------------------------------------------------------------------------
# Extract set of MCMC Tag/ID combinations.
# ------------------------------------------------------------------------------

id_map <- get_mcmc_rep_ids(experiment_dir, round, mcmc_tags_prev, 
                           design_tag_prev, em_tag_prev)

print(paste0("Number of MCMC runs selected: ", nrow(id_map)))
print(paste0("Number of acquisitions specified: ", length(unlist(acq_ids))))
print(paste0("Total number of job submissions required: ", 
             nrow(id_map) * length(unlist(acq_ids))))


# ------------------------------------------------------------------------------
# Submit jobs.
# ------------------------------------------------------------------------------

base_cmd <- paste("qsub", bash_path, experiment_tag, round)
print("Starting jobs")
  
for(acq in unlist(acq_ids)) {
  print(paste0("Acq ID: ", acq))
  
  for(i in 1:nrow(id_map)) {
    mcmc_tag <- id_map[i, mcmc_tag]
    mcmc_id <- id_map[i, mcmc_id]
    
    cmd <- paste(base_cmd, mcmc_tag, mcmc_id, n_batch, acq)
    system(cmd)
  }
}

















