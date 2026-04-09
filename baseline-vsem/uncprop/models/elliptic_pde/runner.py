from jax import config
config.update('jax_enable_x64', True)

import sys
import argparse
from pathlib import Path
from typing import Any
import math

import jax.random as jr
import jax.numpy as jnp

from uncprop.models.elliptic_pde.experiment import PDEReplicate
from uncprop.utils.experiment import Experiment


base_dir = Path('/projectnb/dietzelab/arober/Bayesian-inference-with-surrogates')
base_out_dir = base_dir / 'out'

# -----------------------------------------------------------------------------
# Experiment settings 
#    Settings here are considered fixed global state. Settings that may vary
#    across job submissions are explicitly passed.
# -----------------------------------------------------------------------------

key = jr.key(85455)

# top level experiment settings; None values filled in per job
# Note that `num_reps` is the total number of replicates for the 
# experiment, not the number executed for any particular job
base_experiment_settings = {
    'name': None,
    'base_out_dir': None,
    'num_reps': 100,
    'base_key': key,
    'Replicate': PDEReplicate,
}

# Different experimental setups
design_sizes = [10, 20, 30]

# replicate setup settings
noise_sd = 1e-3
obs_locations = jnp.array([10, 30, 60, 75])
base_setup_kwargs = {
    'n_design': None, # filled per job
    'n_kl_modes': 6,
    'num_rff': 1000,
    'obs_locations': obs_locations,
    'noise_cov': noise_sd**2 * jnp.identity(len(obs_locations)),
}

def make_subdir_name(setup_kwargs, run_kwargs):
    n = setup_kwargs['n_design']
    return f'n_design_{n}'


# replicate run settings
run_kwargs = {
    'mcmc_settings': {'n_samples': 5_000, 'n_burnin': 50_000, 
                      'thin_window': 1, 'adapt_kwargs': {'gamma_exponent': 0.5}},
    'mcwmh_settings': {'n_chains': 100, 'n_samp_per_chain': 10, 
                       'n_burnin': 10_000, 'thin_window': 500,
                       'adapt_kwargs': {'gamma_exponent': 0.5}}
}

# reps that satisfy this condition will be skipped (if overwrite is False) 
def rep_skip_fn(rep_subdir, rep_idx):
    samp_path = rep_subdir / 'samples.npz'
    return samp_path.exists()


# -----------------------------------------------------------------------------
# Batching utilities for cluster array job
# -----------------------------------------------------------------------------

def get_batch(design_idx, chunk_idx, num_reps, rep_chunk_size):
    """Return (n_design, rep_idx) for a given batch."""

    rep_start = chunk_idx * rep_chunk_size
    rep_stop = min(num_reps, (chunk_idx + 1) * rep_chunk_size)

    if rep_start >= num_reps:
        return None, []

    return design_sizes[design_idx], list(range(rep_start, rep_stop))


def task_id_to_indices(task_id, num_reps, rep_chunk_size):
    """ Map array job task ID to a chunk of experiment runs

    Let C := ceil(num_reps / rep_chunk_size). Then the total number of
    jobs is J = C * len(design_sizes). We consider the task ID in
    {0, 1, ..., J-1} and define the mapping from task ID to the chunk via:
        design_idx := task_id // C
        chunk_idx := task_id % C
    """

    n_chunks = math.ceil(num_reps / rep_chunk_size)

    design_idx = task_id // n_chunks
    chunk_idx = task_id % n_chunks

    if design_idx >= len(design_sizes):
        raise ValueError("task_id out of range")

    return design_idx, chunk_idx


# -----------------------------------------------------------------------------
# Main execution
# -----------------------------------------------------------------------------

def run_task(experiment_name, task_id, num_reps, rep_chunk_size):
    design_idx, chunk_idx = task_id_to_indices(task_id, num_reps, rep_chunk_size)
    n_design, rep_idx = get_batch(design_idx, chunk_idx, num_reps, rep_chunk_size)

    if not rep_idx:
        print(f"[task {task_id}] No replicates to run — exiting.")
        return

    setup_kwargs = dict(base_setup_kwargs)
    setup_kwargs['n_design'] = n_design

    experiment_settings = dict(base_experiment_settings)
    experiment_settings['name'] = experiment_name
    experiment_settings['base_out_dir'] = base_out_dir / experiment_name

    print(
        f'[task {task_id}] '
        f'experiment={experiment_name}, '
        f'num_reps={experiment_settings['num_reps']}, '
        f'design_idx={design_idx}, n_design={n_design}, '
        f'chunk_idx={chunk_idx}, reps={rep_idx}'
    )

    experiment = Experiment(
        subdir_name_fn=make_subdir_name,
        **experiment_settings,
    )

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        write_to_log_file=True,
        rep_skip_fn=rep_skip_fn,
    )


def main():
    """For running through a cluster array job"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--task-id', type=int, required=True)
    parser.add_argument('--rep-chunk-size', type=int, default=10)

    args = parser.parse_args()

    run_task(experiment_name=args.experiment_name, 
             task_id=args.task_id, 
             num_reps=base_experiment_settings['num_reps'],
             rep_chunk_size=args.rep_chunk_size)


def main_manual(experiment_name, n_design, rep_idx, write_to_log_file, overwrite=False):
    """Manually specify n_design and the replicate indices to run"""

    experiment_settings = dict(base_experiment_settings)
    experiment_settings['name'] = experiment_name
    experiment_settings['base_out_dir'] = base_out_dir / experiment_name

    setup_kwargs = dict(base_setup_kwargs)
    setup_kwargs['n_design'] = n_design

    print(
        f'[manual run] '
        f'experiment={experiment_name}, '
        f'num_reps={experiment_settings['num_reps']}, '
        f'n_design={n_design}, '
        f'reps={rep_idx}'
    )

    experiment = Experiment(
        subdir_name_fn=make_subdir_name,
        **experiment_settings,
    )

    results, failed_reps, skipped_reps = experiment(
        rep_idx=rep_idx,
        setup_kwargs=setup_kwargs,
        run_kwargs=run_kwargs,
        overwrite=overwrite,
        write_to_log_file=write_to_log_file,
        rep_skip_fn=rep_skip_fn,
    )


if __name__ == "__main__":
    main()