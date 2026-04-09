from jax import config
config.update('jax_enable_x64', True)
from pathlib import Path
from typing import Any

import jax.random as jr
from uncprop.models.vsem.experiment import VSEMReplicate, VSEMExperiment

base_dir = Path('.')

# -----------------------------------------------------------------------------
# Experiment settings 
# -----------------------------------------------------------------------------

key = jr.key(9768565)

# Different experimental setups
gp_tags = ['gp', 'clip_gp']
design_settings = [(4, 1e-4), (8, 1e-3), (16, 1e-2)] # (n_design, jitter)
setups = [(tag, n, jitter) for tag in gp_tags for n, jitter in design_settings]

setup_kwargs: dict[str, Any] = {'n_grid': 50, 
                                'n_design': None,
                                'noise_sd': 1.0, 
                                'verbose': False,
                                'jitter': None,
                                'surrogate_tag': None}
run_kwargs: dict[str, Any] = {}

num_reps = 100
backup_frequency = 10
experiment_name = 'vsem'
out_dir = base_dir / 'out' / experiment_name

def _make_subdir_name(setup_kwargs, run_kwargs):
    return f'{setup_kwargs['surrogate_tag']}_N{setup_kwargs['n_design']}'


# -----------------------------------------------------------------------------
# Run experiment 
# -----------------------------------------------------------------------------

def run_vsem_experiment():
    experiment = VSEMExperiment(name=experiment_name,
                                num_reps=num_reps,
                                base_out_dir=out_dir,
                                base_key=key,
                                Replicate=VSEMReplicate,
                                subdir_name_fn=_make_subdir_name)
    all_results = []

    for tag, n, jitter in setups:
        setup_kwargs['n_design'] = n
        setup_kwargs['jitter'] = jitter
        setup_kwargs['surrogate_tag'] = tag

        results = experiment(run_kwargs=run_kwargs, 
                             setup_kwargs=setup_kwargs)
        all_results.append(results)

    return all_results


if __name__ == '__main__':
    run_vsem_experiment()