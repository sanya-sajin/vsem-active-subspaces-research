#
# run_init_emulator.r
#
# The runner script for the initial emulator fitting step (fitting to the 
# initial designs). This script is thus run once per the experiment, and writes
# output to the `experiments/<experiment_tag>/output/round1/em` directory.
#
# This script works by specifying an experiment tag, and lists of design
# tags and emulator tags. One emulator will be fit per emulator tag per
# design ID within the specified design tag. Currently, jobs are batched
# by unique (emulator tag, design tag) combinations, using the helper files
# `scripts/helper/init_emulator.r` and `scripts/bash/init_emulator.sh`. 
# Emulator IDs (which are unique within each round/emulator tag) are created
# in this script and saved to 
# `experiments/<experiment_tag>/output/round1/em//id_map.csv`. These
# ID map files provide the information to determine which design was used
# to fit each emulator. They are read in by the helper files mentioned above
# when fitting the emulators.
# 
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Paths
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Experiment
experiment_tag <- "banana"

# Determine emulators and designs to use.
em_tags <- c("em_lpost_twostage")
design_tags <- c("lhs_extrap_6", "lhs_extrap_12", "lhs_extrap_24", 
                 "lhs_extrap_48", "lhs_extrap_96")


# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Directories.
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
src_dir <- file.path(code_dir, "src")
design_dir <- file.path(experiment_dir, "output", "round1", "design")
em_dir <- file.path(experiment_dir, "output", "round1", "em")
bash_path <- file.path(base_dir, "scripts", "bash", "init_emulator.sh")

dir.create(em_dir, recursive=TRUE)

# Required input files.
em_settings_path <- file.path(experiment_dir, "output", "alg_settings", "em_settings.rds")

# Check emulator tags.
print(paste0("Using emulator settings: ", em_settings_path))
em_settings <- readRDS(em_settings_path)
valid_em_tags <- sapply(em_settings, function(x) x$em_label)
invalid_tags <- setdiff(em_tags, valid_em_tags)

if(length(invalid_tags) > 0L) {
  stop("Emulator tags not found in emulator settings: ", paste0(invalid_tags, collapse=", "))
}

# Read design ID map.
design_id_map_path <- file.path(design_dir, "id_map.csv")
design_id_map <- fread(design_id_map_path)

# Check if current em ID map exists. Ensure duplicate IDs are not created.
em_id_start <- 1L
id_map_curr <- NULL
id_map_path <- file.path(em_dir, "id_map.csv")
if(file.exists(id_map_path)) {
  id_map_curr <- fread(id_map_path)
  em_id_start <- id_map_curr[, max(em_id)] + 1L
} 

# ------------------------------------------------------------------------------
# Create ID map file and write to file. 
# ------------------------------------------------------------------------------

id_map <- data.table(design_id=integer(), design_tag=character(),
                     em_tag=character(), em_id=integer(), seed=integer())

for(em_tag in em_tags) {
  print(paste0("-----> Emulator tag: ", em_tag))
  em_id_map <- data.table(design_id=integer(), design_tag=character())
  
  for(design_tag in design_tags) {
    print(paste0("    Design tag: ", design_tag))
    d_tag <- design_tag
    design_ids <- design_id_map[design_tag==d_tag, .(design_tag, design_id)]
    em_id_map <- rbindlist(list(em_id_map, design_ids), use.names=TRUE)
  }
  
  # Set em_ids here so that they are unique within each em_tag.
  seeds <- sample.int(n=.Machine$integer.max, size=nrow(em_id_map))
  em_id_map[, `:=`(em_tag=em_tag, em_id=seq(em_id_start, em_id_start + .N - 1L), seed=seeds)]
  id_map <- rbindlist(list(id_map, em_id_map), use.names=TRUE)
}

# Save id_map to file.
if(!is.null(id_map_curr)) {
  id_map <- rbindlist(list(id_map_curr, id_map), use.names=TRUE)
  
  if(any(duplicated(id_map[, .(em_tag, em_id)]))) {
    stop("Duplicate (em_tag, em_id) values found in ID map. Not saving file.")
  }
}

print(paste0("Saving ID map: ", id_map_path))
fwrite(id_map, id_map_path)

# ------------------------------------------------------------------------------
# Create qsub jobs to fit emulators. 
# ------------------------------------------------------------------------------

runs <- unique(id_map[(em_tag %in% em_tags) & (design_tag %in% design_tags), 
                      .(em_tag, design_tag)])

# Execute one job per design ID.
print("-----> Starting jobs:")
base_cmd <- paste("qsub", bash_path, experiment_tag)

for(i in 1:nrow(runs)) {
  em_tag <- runs[i, em_tag]
  design_tag <- runs[i, design_tag]
  print(paste(em_tag, design_tag, sep=" --- "))
  
  cmd <- paste(base_cmd, em_tag, design_tag)
  system(cmd)
}

