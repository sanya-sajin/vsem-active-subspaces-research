#!/bin/bash -l
#$ -N approx_mcmc
#$ -l h_rt=12:00:00             # Hard time limit.
#$ -P dietzelab                 # Specify project. 
#$ -l buyin                     # Request buyin node. 
#$ -j y                         # Merge the error and output streams into a single file.

# Runs the R script `run_postprocess_mcmc.r`.

# NOTE: 
# The R library path is not set explicitly in this file. Therefore, if using an 
# R project, one should ensure that renv::load() is called within the R script 
# being run so that the correct library path is set. 

# Settings (currently only used for creating log file path - should actually pass these to the R script).
EXPERIMENT_TAG="banana"
ROUND="1"

# For logging: log file will be moved from its default location.
LOG_FILENAME="${JOB_NAME}.o${JOB_ID}"
echo "Log file: ${LOG_FILENAME}"

# Main output directory for this run.
OUT_DIR="/projectnb/dietzelab/arober/bip-surrogates-paper/experiments/${EXPERIMENT_TAG}/output/round${ROUND}/mcmc/log"
echo "Creating output directory: ${OUT_DIR}"
mkdir -p ${OUT_DIR};

# Path to R script to execute.
R_SCRIPT_PATH="/projectnb/dietzelab/arober/bip-surrogates-paper/scripts/runners/run_postprocess_mcmc.r"

# Load modules, including specification of the R version. 
module load gcc/8.3.0
module load R/4.3.1

# Required for the `Rscript` command to be found. 
export PATH="$PATH:/share/pkg.8/r/4.3.1/install/lib64/R"

# Execute R script.
Rscript ${R_SCRIPT_PATH}

# Move output file to the output directory (if it exists). 
if [ -d ${OUT_DIR} ]; then
mv "${LOG_FILENAME}" "${OUT_DIR}/${LOG_FILENAME}"
fi






