#
# run_init_design.r
# Part 2 in the inverse problem simulation study workflow, to be run after
# `run_inv_prob_setup.r`. This script:
# (1) loads inverse problem data saved to file by `run_inv_prob_setup.r`. 
# (2) Generates multiple random replications of an initial design to train 
#     log-likelihood and forward model emulators.
# (3) The outputs are saved to 
#     `experiments/<experiment_tag>/output/round1/design/design_tag/.
#     Each replicate design is saved to a subdirectory of the design tag directory,
#     with directory name set to the design ID. Each subdirectory will contain
#     a file `design_info.rds` storing the design information. Within the 
#     design tag subdirectory a file called `design_info.rds` is also saved
#     storing metadata for the design. An additional file `id_map.csv` is
#     also saved in this file which provides a full list of the design IDs
#     within that design tag.
#
# Note that this file is specifically for saving initial design information
# (i.e., round 1 design). It is not used for sequential design steps. Also,
# note that the design IDs are unique within a design tag. Thus, a design tag/
# design ID combination is what uniquely identifies a specific initial
# design instance. Note that this script only saves replicate designs
# for a single design tag (i.e., a single initial design method). This is to
# do with the fact that my emphasis is not on evaluating different initial
# design methods. However, this could be generalized to be handled in a similar
# way as the emulator/MCMC/seq design algorithms if one wants to compare
# different initial design methods. Since the design replicates are all 
# created in this file, a single random seed is specified, that is associated
# with the design tag. This seed can thus be used to reproduce all of the
# replicates. To have more fine-grained control, an additional seed is defined
# for each replicate, and stored in the `id_map.csv` file.
#
# Andrew Roberts
#

library(data.table)
library(assertthat)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

design_seed <- 645324
set.seed(design_seed)

experiment_tag <- "banana"
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Design methods settings.
n_design <- 96
design_method <- "lhs_extrap" 

# Initial design tag.
design_tag <- paste(design_method, n_design, sep="_")

# Number of design replications.
n_rep <- 100L

print("--------------------Running `init_design.r` --------------------")
print(paste0("Experiment tag: ", experiment_tag))
print(paste0("Design tag: ", design_tag))
print(paste0("n_design: ", n_design))
print(paste0("design_method: ", design_method))
print(paste0("n_rep: ", n_rep))

settings <- list(experiment_tag=experiment_tag,
                 design_tag=design_tag,
                 n_design=n_design,
                 design_method=design_method,
                 n_rep=n_rep,
                 design_seed=design_seed)

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Filepath definitions.
src_dir <- file.path(code_dir, "src")
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
setup_dir <- file.path(experiment_dir, "output", "inv_prob_setup")
design_dir <- file.path(experiment_dir, "output", "round1", "design")
out_dir <- file.path(design_dir, design_tag)
alg_settings_dir <- file.path(experiment_dir, "output", "alg_settings")

# Create output directories.
print(paste0("Output directory: ", out_dir))
dir.create(out_dir, recursive=TRUE)

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

# Load inverse problem setup information.
inv_prob <- readRDS(file.path(setup_dir, "inv_prob_list.rds"))

# Read initial design algorithm settings.
alg_settings_path <- file.path(alg_settings_dir, "init_design_settings.rds")
alg_settings <- readRDS(alg_settings_path)

if(!(design_method %in% names(alg_settings))) {
  stop("Design method `", design_method, "` not found in algorithm settings: ",
       alg_settings_path)
}

design_func <- alg_settings[[design_method]]

# Save design settings.
print("Saving design settings.")
design_settings_path <- file.path(out_dir, "design_settings.rds")
if(file.exists(design_settings_path)) {
  stop("Design settings already exists: ", design_settings_path)
}

saveRDS(settings, design_settings_path)

# ------------------------------------------------------------------------------
# Define function that creates a single replicate initial design.
# ------------------------------------------------------------------------------

save_init_design <- function(seed, i) {
  print("-------------- Initiating new design --------------")
  set.seed(seed)
  print(paste0("Design: ", i, " -----  Seed: ", seed))
  
  design_info <- design_func(inv_prob, n_design)
  out_dir_design <- file.path(out_dir, paste0("design_", i))
  dir.create(out_dir_design)
  saveRDS(design_info, file=file.path(out_dir_design, "design_info.rds"))
}

# ------------------------------------------------------------------------------
# Create replicate designs
# ------------------------------------------------------------------------------

print("-------------- Generating replicate designs --------------")

# Seeds for each replicate design.
seeds <- sample.int(n=.Machine$integer.max, size=settings$n_rep)

# Save ID map file.
dt <- data.table(design_tag=design_tag, design_id=seq_len(settings$n_rep),
                 seed=seeds)
id_map_path <- file.path(file.path(design_dir, "id_map.csv"))

if(file.exists(id_map_path)) {
  print(paste0("Appending to current ID map: ", id_map_path))
  id_map <- fread(id_map_path)
  id_map <- rbindlist(list(id_map, dt), use.names=TRUE)
} else {
  print(paste0("Saving new ID map: ", id_map_path))
  id_map <- copy(dt)
}

fwrite(id_map, id_map_path)

# Generate replicate designs.
for(i in seq_along(seeds)) {
  print(i)
  seed <- seeds[i]
  save_init_design(seed, i)
}


