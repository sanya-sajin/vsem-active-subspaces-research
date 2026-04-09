#!/bin/bash -l
#$ -N init_emulator
#$ -l h_rt=12:00:00             # Hard time limit.
#$ -P dietzelab                 # Specify project. 
#$ -l buyin                     # Request buyin node. 
#$ -j y                         # Merge the error and output streams into a single file.

# Runs the R script `init_emulator.r`. Intended to be called from 
# `run_init_emulator.r`. See these two files for details.

# NOTE: 
# The R library path is not set explicitly in this file. Therefore, if using an 
# R project, one should ensure that renv::load() is called within the R script 
# being run so that the correct library path is set. 

# Read commandline arguments.
EXPERIMENT_TAG=$1
EM_TAG=$2
DESIGN_TAG=$3

# For logging: log file will be moved from its default location.
LOG_FILENAME="${JOB_NAME}.o${JOB_ID}"
echo "Log file: ${LOG_FILENAME}"

# Main output directory for this run.
OUT_DIR="/projectnb/dietzelab/arober/bip-surrogates-paper/experiments/${EXPERIMENT_TAG}/output/round1/em/${EM_TAG}/log"
echo "Creating output directory: ${OUT_DIR}"
mkdir -p ${OUT_DIR};

# Path to R script to execute.
R_SCRIPT_PATH="/projectnb/dietzelab/arober/bip-surrogates-paper/scripts/helper/init_emulator.r"

# Load modules, including specification of the R version. 
module load gcc/8.3.0
module load R/4.3.1

# Required for the `Rscript` command to be found. 
export PATH="$PATH:/share/pkg.8/r/4.3.1/install/lib64/R"

# Execute R script.
Rscript ${R_SCRIPT_PATH} --experiment_tag=${EXPERIMENT_TAG} \
    --em_tag=${EM_TAG} \
    --design_tag=${DESIGN_TAG}

# Move output file to the output directory (if it exists). 
if [ -d ${OUT_DIR} ]; then
mv "${LOG_FILENAME}" "${OUT_DIR}/${EM_TAG}_${DESIGN_TAG}_${LOG_FILENAME}"
fi






