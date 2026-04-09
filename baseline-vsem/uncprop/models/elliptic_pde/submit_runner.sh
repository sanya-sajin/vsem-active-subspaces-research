#!/bin/bash
#$ -N pde_experiment
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=12G
#$ -P gpsurr
#$ -pe omp 1
#$ -t 25-30 -tc 1    # cap number run concurrently to reduce chance of XLA compilation issues
#
# Array job for executing PDE experiment. It is the users responsibility
# for setting the task IDs (e.g., `-t 1-30`) and the batch size 
# (e.g., `--rep-chunk-size 10`) in a compatible manner.
#
# Example:
# For 3 different design settings, with 100 reps per setting. Batch size of
# 10 implies 30 jobs, so the task IDs should be specified as `-t 1-30`.
#
# Issues with JAX running on CPU clusters:
# https://github.com/jax-ml/jax/issues/1539
# https://github.com/jax-ml/jax/issues/16215

source /projectnb/dietzelab/arober/Bayesian-inference-with-surrogates/.venv/bin/activate

# Debug and version info
echo "JOB_ID=$JOB_ID SGE_TASK_ID=$SGE_TASK_ID"
hostname
python --version
echo "Git commit: $(git rev-parse HEAD)"

# Scratch space for XLA
export TMPDIR=/scratch/$USER/$JOB_ID.$SGE_TASK_ID
mkdir -p $TMPDIR
trap "rm -rf $TMPDIR" EXIT

# encourage single-threaded computation
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUM_INTER_THREADS=1
export NUM_INTRA_THREADS=1
export NPROC=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1"
export TF_CPP_MIN_LOG_LEVEL=2

# per job JAX cache
export JAX_CACHE_DIR=/scratch/$USER/jax_cache_$JOB_ID.$SGE_TASK_ID
mkdir -p $JAX_CACHE_DIR
trap "rm -rf $TMPDIR $JAX_CACHE_DIR" EXIT

# convert to 0-based Python indexing
TASK_ID=$((SGE_TASK_ID - 1))

# sleep between 0-30 seconds - trying to avoid JAX compilation issues due to concurrency
sleep $((RANDOM % 30))

exec python -u /projectnb/dietzelab/arober/Bayesian-inference-with-surrogates/uncprop/models/elliptic_pde/runner.py \
    --task-id ${TASK_ID} \
    --rep-chunk-size 10 \
    --experiment-name pde_experiment