# uncprop/models/vsem/experiment.py
from pathlib import Path
from typing import Any, NamedTuple
from types import MappingProxyType
from collections.abc import Mapping

import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.utils.grid import Grid, DensityComparisonGrid
from uncprop.core.inverse_problem import Posterior
from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep

from uncprop.models.vsem.surrogate import (
    fit_vsem_surrogate, 
    VSEMPosteriorSurrogate, 
    LogDensClippedGPSurrogate,
)
from uncprop.core.samplers import (
    sample_distribution, 
    mcmc_loop, 
    init_rkpcn_kernel, 
    _f_update_pcn_proposal,
)

class VSEMReplicate(Replicate):
    """
    Replicate in VSEM experiment:
        - Randomly generate synthetic data/ground truth for specified inverse problem structure
        - Randomly samples design points and fits a log-posterior surrogate
        - Computes true posterior and surrogate-based approximations on 2d grid
    """

    _DEFAULT_INVERSE_PROBLEM_SETTINGS = MappingProxyType({
        'par_names': ('av', 'veg_init'),
        'n_windows': 12,
        'noise_cov_tril': jnp.identity(12),
        'n_days_per_window': 30,
        'observed_variable': 'lai',
    })

    _DEFAULT_SURROGATE_SETTINGS = MappingProxyType({
        'design_method': 'lhc',
        'jitter' : 0.0,
        'verbose': True,
    })

    def __init__(self, 
                 key: PRNGKey,
                 out_dir: Path,
                 n_design: int,
                 surrogate_tag: str, 
                 n_grid: int = 50,
                 inverse_problem_settings: Mapping[str, Any] | None = None,
                 surrogate_settings: Mapping[str, Any] | None = None,
                 **kwargs):
        
        key_inv_prob, key_surrogate = jr.split(key, 2)

        # Per-instance copies
        self.inverse_problem_settings = {
            **self._DEFAULT_INVERSE_PROBLEM_SETTINGS,
            **(inverse_problem_settings or {}),
        }

        self.surrogate_settings = {
            **self._DEFAULT_SURROGATE_SETTINGS,
            **(surrogate_settings or {}),
            'n_design': n_design,
            'surrogate_tag': surrogate_tag,
        }
        
        # exact posterior
        posterior = generate_vsem_inv_prob_rep(key=key_inv_prob,
                                               **self.inverse_problem_settings)
        vsem_params_dict = posterior.likelihood.model.vsem_params
        vsem_params = jnp.array(list(vsem_params_dict.values()))
        vsem_param_names = list(vsem_params_dict.keys())
        
        # surrogate posterior
        surrogate_post, fit_info = fit_vsem_surrogate(key=key_surrogate, 
                                                      posterior=posterior,
                                                      **self.surrogate_settings)
        
        # grid points for grid-based metrics
        grid = Grid(low=posterior.support[0],
                    high=posterior.support[1],
                    n_points_per_dim=[n_grid, n_grid],
                    dim_names=posterior.prior.par_names)
        
        self.key = key
        self.posterior = posterior
        self.posterior_surrogate = surrogate_post
        self.vsem_params = vsem_params
        self.vsem_param_names = vsem_param_names
        self.design = self.posterior_surrogate.surrogate.design
        self.fit_info = fit_info
        self.grid = grid
        self.surrogate_pred = self.posterior_surrogate.surrogate(grid.flat_grid)


    def __call__(self, 
                 key: PRNGKey, 
                 out_dir: PRNGKey,
                 rkpcn_rho_vals: dict[str, float],
                 mcmc_settings: dict[str, Any] | None = None,
                 rkpcn_settings: dict[str, Any] | None = None,
                 **kwargs):
        
        key, key_ep, key_init_mcmc, key_seed_mcmc, key_seed_rkpcn = jr.split(key, 5)

        if mcmc_settings is None:
            mcmc_settings = {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5}
        if rkpcn_settings is None:
            rkpcn_settings = {'n_samples': 1000, 'n_burnin': 50_000, 'thin_window': 5}

        # exact and approximate posteriors
        surr = self.posterior_surrogate
        dists = {
            'exact': self.posterior,
            'mean': surr.expected_surrogate_approx(),
            'eup': surr.expected_density_approx(),
            'ep': surr.expected_normalized_density_approx(key_ep, grid=self.grid)
        }
        density_comparison = DensityComparisonGrid(grid=self.grid, distributions=dists)

        # run samplers
        initial_position = self.posterior.prior.sample(key_init_mcmc)
        mcmc_keys = jr.split(key_seed_mcmc, len(dists))

        mcmc_samp = {}
        mcmc_info = {}
        print('\tRunning samplers')
        for key, (dist_name, dist) in zip(mcmc_keys, dists.items()):
            mcmc_results = sample_distribution(
                key=key,
                dist=dist,
                initial_position=initial_position,
                **mcmc_settings
            )

            mcmc_samp[dist_name] = mcmc_results['positions'].squeeze(1)
            mcmc_info[dist_name] = mcmc_results

        rkpcn_keys = jr.split(key_seed_rkpcn, len(rkpcn_rho_vals))
        rkpcn_samp = {}
        
        for i, alg_name in enumerate(rkpcn_rho_vals.keys()):
            mcmc_samp[alg_name] = _run_mcmc_rkpcn(key=rkpcn_keys[i], 
                                                  rho=rkpcn_rho_vals[alg_name],
                                                  posterior=self.posterior,
                                                  surrogate_post=surr,
                                                  initial_position=initial_position,
                                                  **rkpcn_settings)


        self.density_comparison = density_comparison
        self.coverage_results = self.density_comparison.calc_coverage(baseline='exact')
        self.mcmc_results = mcmc_results

        return self
    

class VSEMExperiment(Experiment):

    def collect_results(self, results, failed_reps, *args, **kwargs):
        """
        Note that in the below comments `n_reps` is the number of non-failed replicates.
        """

        # Only format non-failed iterations
        results = [rep for i, rep in enumerate(results) if i not in failed_reps]
        if len(results) == 0:
            return None
        
        # inverse problem data realization
        vsem_params = jnp.vstack([rep.vsem_params for rep in results])
        driver = jnp.stack([rep.posterior.likelihood.model.driver for rep in results], axis=0)
        vsem_output = jnp.stack([rep.posterior.likelihood.data.vsem_output for rep in results], axis=0)
        observable = jnp.stack([rep.posterior.likelihood.data.observable for rep in results], axis=0)
        observation = jnp.stack([rep.posterior.likelihood.data.observation for rep in results], axis=0)
        
        # design points
        design_x = jnp.stack([rep.design.X for rep in results], axis=0) # (n_reps, n_design, d)
        design_y = jnp.vstack([rep.design.y.ravel() for rep in results]) # (n_reps, n_design)

        # surrogate mean/variance predictions at grid points; each (n_reps, n_grid)
        pred_mean = jnp.vstack([rep.surrogate_pred.mean.ravel() for rep in results])
        pred_var = jnp.vstack([rep.surrogate_pred.variance.ravel() for rep in results])

        # unnormalized log-densities of each distribution over grid points; dict of (n_reps, n_grid)
        names = list(results[0].density_comparison.distributions.keys())
        log_dens_approx = {}
        for nm in names:
            log_dens_approx[nm] = jnp.vstack([
                rep.density_comparison.log_dens_grid[nm] for rep in results
            ])

        # coverage results
        log_coverage = jnp.stack(
            [rep.coverage_results[0] for rep in results], 
            axis=0
        )

        probs = results[0].coverage_results[1]
        dist_names = results[0].coverage_results[3]

        info = {'vsem_params': vsem_params,
                'driver': driver,
                'vsem_output': vsem_output,
                'observable': observable,
                'observation': observation,
                'pred_mean': pred_mean,
                'pred_var': pred_var,
                'design_x': design_x,
                'design_y': design_y,
                'log_coverage': log_coverage,
                'probs': probs,
                'dist_names': dist_names,
                'vsem_param_names': results[0].vsem_param_names}

        return info, log_dens_approx


    def save_results(self, 
                     subdir: Path,
                     results: list, 
                     failed_reps: list, 
                     *args, **kwargs):
        results_dict, log_dens_dict = self.collect_results(results, failed_reps)

        if results_dict is None:
            print('Results list is length 0. Not saving')
            return None
        
        # save grid info (constant across reps)
        grid = results[0].grid
        jnp.savez(subdir / 'grid_info.npz', 
                  low=grid.low,
                  high=grid.high,
                  n_points_per_dim=grid.n_points_per_dim,
                  dim_names=grid.dim_names)

        jnp.savez(subdir / 'results.npz', **results_dict)
        jnp.savez(subdir / 'log_dens.npz', **log_dens_dict)
        jnp.savez(subdir / 'logging_info.npz', failed_reps=failed_reps, key=jr.key_data(self.base_key))


# -----------------------------------------------------------------------------
# Helper functions for running experiment
# -----------------------------------------------------------------------------

def _run_mcmc_exact(key: PRNGKey, posterior: Posterior, n_samples: int):
    key_init_pos, key_samp = jr.split(key)
    initial_position = posterior.prior.sample(key_init_pos).squeeze()

    results = sample_distribution(key=key,
                                  dist=posterior,
                                  initial_position=initial_position,
                                  n_samples=n_samples,
                                  n_warmup=50_000,
                                  thin_window=3)
    
    samp = results[0]
    prop_cov = results[3]

    return samp, prop_cov


def _run_mcmc_rkpcn(key: PRNGKey,
                    posterior: Posterior,
                    surrogate_post: VSEMPosteriorSurrogate,
                    initial_position: Array,
                    prop_cov: Array,
                    rho: float,
                    n_samples: int,
                    n_burnin: int = 50_000,
                    thin_window: int = 5):
    
    key_ker, key_samp = jr.split(key)    

    # correctly enforce clipping operation and prior support
    low, high = surrogate_post.support
    upper_bound = lambda u: posterior.prior.log_density(u) + posterior.likelihood.log_density_upper_bound(u)

    if isinstance(surrogate_post, LogDensClippedGPSurrogate):
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            upper = upper_bound(u)
            lp = jnp.clip(f, max=upper)
            lp = jnp.where(jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()
    else:
        def log_density(f, u):
            u = jnp.atleast_2d(u)
            lp = f
            lp = jnp.where(jnp.all((u >= low) & (u <= high), axis=1), lp, -jnp.inf)
            return lp.squeeze()

    # underlying GP model
    gp = surrogate_post.surrogate

    # settings for f update in sampler
    class UpdateInfo(NamedTuple):
        rho: float
    f_update_info = UpdateInfo(rho=rho)

    initial_state, kernel = init_rkpcn_kernel(key=key_ker,
                                              log_density=log_density,
                                              gp=gp,
                                              initial_position=initial_position,
                                              u_prop_cov=prop_cov,
                                              f_update_fn=_f_update_pcn_proposal,
                                              f_update_info=f_update_info)

    # run sampler
    n_samples_total = n_burnin + thin_window * n_samples
    out = mcmc_loop(key=key_samp,
                    kernel=kernel,
                    initial_state=initial_state,
                    num_samples=n_samples_total)

    samp = out.position[n_burnin:]

    return samp[::thin_window]

# -----------------------------------------------------------------------------
# Helper functions for analysis/plotting
# -----------------------------------------------------------------------------

def load_results(out_dir: str | Path, subdir_names: list[str]):
    out_dir = Path(out_dir)
    results = {}

    for nm in subdir_names:
        subdir = out_dir / nm
        results[nm] = {}
        res = results[nm]

        res['logging_info'] = jnp.load(subdir / 'logging_info.npz')
        res['grid_info'] = jnp.load(subdir / 'grid_info.npz')
        res['results'] = jnp.load(subdir / 'results.npz')
        res['log_dens'] = jnp.load(subdir / 'log_dens.npz')

    return results


def summarize_rep(out_dir: str | Path, subdir_name: str, rep_idx: int, n_reps: int):
    subdir = Path(out_dir) / subdir_name

    # check if rep failed
    info = jnp.load(subdir / 'logging_info.npz')
    failed_reps = info['failed_reps']
    if rep_idx in failed_reps:
        print(f'Replicate {rep_idx} failed.')
        return None
    
    # adjust rep index (necessary if some reps failed)
    if len(failed_reps) > 0:
        all_rep_idx = [idx for idx in range(n_reps)]
        rep_idx = all_rep_idx.index(rep_idx)
    
    # Load grid for plots
    grid_info = jnp.load(subdir / 'grid_info.npz')

    grid = Grid(low=grid_info['low'],
                high=grid_info['high'],
                n_points_per_dim=grid_info['n_points_per_dim'],
                dim_names=grid_info['dim_names'])
    
    # load results
    results = jnp.load(subdir / 'results.npz')
    log_dens = jnp.load(subdir / 'log_dens.npz')

    # Produce plots
    gp_plot = _summarize_rep_gp(log_dens, results, grid, rep_idx)
    post_approx_plots = _summarize_rep_post_approx(log_dens, results, grid, rep_idx)
    plots = [gp_plot] + post_approx_plots

    return grid, results, log_dens, plots


def _summarize_rep_gp(log_dens, results, grid, rep_idx):
    exact = log_dens['exact'][rep_idx]
    pred_mean = results['pred_mean'][rep_idx]
    pred_sd = jnp.sqrt(results['pred_var'])[rep_idx]
    design_x = results['design_x'][rep_idx]

    fig, ax = grid.plot(z=[exact, pred_mean, pred_sd],
                        titles=['exact', 'mean', 'sd'],
                        points=design_x,  max_cols=3)
    
    return (fig, ax)


def _summarize_rep_post_approx(log_dens, results, grid, rep_idx):
    log_dens_rep = {nm: arr[rep_idx] for nm, arr in log_dens.items()}
    post_approx_grid = DensityComparisonGrid(grid=grid, log_dens_grid=log_dens_rep)
    design_x = results['design_x'][rep_idx]

    log_dens_comparison_plot = post_approx_grid.plot(normalized=True, log_scale=True, 
                                                     max_cols=4, points=design_x)
    dens_comparison_plot = post_approx_grid.plot(normalized=True, log_scale=False, 
                                                 max_cols=4, points=design_x)
    coverage_grid = post_approx_grid.plot_coverage(baseline='exact', probs=results['probs'])

    return [log_dens_comparison_plot, dens_comparison_plot, coverage_grid]