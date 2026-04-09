#!/projectnb/dietzelab/arober/Bayesian-inference-with-surrogates/.venv/bin/python -u
#$ -N w2_30
#$ -P gpsurr
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=12G
#$ -pe omp 2
#
# Helper script to add run rkpcn sampler and saved results for existing experiment.

from jax import config
config.update('jax_enable_x64', True)

import os
import sys
from pathlib import Path
from datetime import datetime

import jax.numpy as jnp
import jax.random as jr

from uncprop.models.elliptic_pde.experiment import (
    load_rep,
    read_samp,
    sample_rkpcn,
)

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUM_INTER_THREADS'] = '1'
os.environ['NUM_INTRA_THREADS'] = '1'
os.environ['NPROC'] = '1'
os.environ['XLA_FLAGS'] = '--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1 inter_op_parallelism_threads=1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

timestamp = datetime.now()

print(f'Executable: {sys.executable}', flush=True)
print('Timestamp:', timestamp.strftime("%Y-%m-%d %H:%M:%S"))

experiment_name = 'pde_experiment'
base_dir = Path('/projectnb/dietzelab/arober/Bayesian-inference-with-surrogates')
base_out_dir = base_dir / 'out'
# experiment_dir = base_out_dir / experiment_name
# key = jr.key(523342)

# rho_vals = [0.0, 0.9, 0.95, 0.99]

# print(f'PRNG key: {key}')

# for n_design in [20]:
#     design_dir = experiment_dir / f'n_design_{n_design}'

#     for rep_idx in range(58, 100):
#         try:
#             print(f'n_design = {n_design} --- rep = {rep_idx}')
#             rep_dir = design_dir / f'rep{rep_idx}'
            
#             rep = load_rep(base_out_dir, experiment_name, n_design, rep_idx)
#             samp_eup = read_samp(base_out_dir, experiment_name, n_design, rep_idx)['eup']

#             posterior = rep.posterior
#             posterior_surrogate = rep.posterior_surrogate

#             key, key_init_pos, key_rkpcn = jr.split(key, 3)
#             initial_position = posterior.prior.sample(key_init_pos).squeeze()

#             prop_cov = jnp.cov(samp_eup, rowvar=False)

#             rkpcn_output = {}

#             for rho in rho_vals:
#                 print(f'\trho = {rho}')
#                 key, key_rkpcn = jr.split(key)

#                 samp_rkpcn = sample_rkpcn(key=key_rkpcn,
#                                           posterior=posterior,
#                                           surrogate_post=posterior_surrogate,
#                                           initial_position=initial_position,
#                                           prop_cov=prop_cov,
#                                           rho=rho,
#                                           n_samples=5_000,
#                                           n_burnin=50_000,
#                                           thin_window=5)
#                 tag = f'rkpcn{int(rho*100)}'                          
#                 rkpcn_output[tag] = samp_rkpcn

#             jnp.savez(rep_dir / 'rkpcn_samples.npz', **rkpcn_output)
#         except Exception as e:
#             print(f'n_design {n_design}, rep_idx {rep_idx} failed with error:')
#             print(e)


output_dir = base_dir / 'out' / 'final'
key = jr.key(12343)

from uncprop.models.elliptic_pde.experiment import (
    summarize_wasserstein_design_reps,
)

key, key_w2 = jr.split(key)
n_design = 30

print('n_design: ', n_design)
w2_results_n, eps_n = summarize_wasserstein_design_reps(key_w2, 
                                                        base_out_dir, 
                                                        experiment_name, 
                                                        n_design=n_design, 
                                                        rep_idcs=range(100),
                                                        output_dir=output_dir)
jnp.savez(output_dir / f'w2_ndesign_{n_design}.npz', **w2_results_n)
print('epsilon:', eps_n)