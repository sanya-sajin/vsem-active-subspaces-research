# uncprop/models/elliptic_pde/experiment.py
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import solve_triangular

from ott.geometry import pointcloud
from ott.solvers.linear import sinkhorn
from ott.problems.linear import linear_problem

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from uncprop.custom_types import PRNGKey, Array
from uncprop.utils.experiment import Replicate, Experiment
from uncprop.core.inverse_problem import Posterior
from uncprop.core.distribution import DistributionFromDensity
from uncprop.core.samplers import (
    _f_update_pcn_proposal,
    sample_distribution, 
    init_rkpcn_kernel, 
    mcmc_loop,
)
from uncprop.models.elliptic_pde.surrogate import fit_pde_surrogate, PDEFwdModelGaussianSurrogate
from uncprop.models.elliptic_pde.inverse_problem import (
    generate_pde_inv_prob_rep,
    PDESettings,
)

        
class PDEReplicate(Replicate):
    """
    Replicate in PDE experiment:
        - Randomly generate synthetic data/ground truth for specified inverse problem structure
        - Randomly samples design points and fits a forward model surrogate
        - Runs MCMC for true posterior and surrogate-based approximations
    """

    def __init__(self, 
                 key: PRNGKey,
                 out_dir: Path,
                 n_design: int,
                 num_rff: int,
                 design_method: str = 'lhc',
                 write_to_file: bool = True,
                 **kwargs):
        
        jax.clear_caches()
        if write_to_file:
            jnp.savez(out_dir / 'init_settings.npz', 
                      key_init=jr.key_data(key),
                      out_dir=out_dir,
                      n_design=n_design,
                      num_rff=num_rff,
                      design_method=design_method,
                      write_to_file=write_to_file)

        key_inv_prob, key_surrogate, key_rff = jr.split(key, 3)

        # default settings
        noise_sd = 1e-2
        n_kl_modes = 6
        obs_locations = jnp.array([10, 30, 60, 75])

        defaults = {
            'noise_cov' : noise_sd**2 * jnp.identity(len(obs_locations)),
            'n_kl_modes': n_kl_modes,
            'obs_locations': obs_locations,
            'settings': PDESettings()
        }

        # override defaults
        settings = {k: kwargs.get(k,v) for k, v in defaults.items()}
  
        # exact posterior
        inv_prob_info = generate_pde_inv_prob_rep(key=key_inv_prob, **settings)
        posterior = inv_prob_info[0]
        ground_truth = inv_prob_info[3]

        # fit surrogate
        print('\tFitting surrogate')
        design, surrogate, batchgp, opt_history = fit_pde_surrogate(key=key_surrogate,
                                                                    posterior=posterior,
                                                                    n_design=n_design,
                                                                    design_method=design_method)
        
        # surrogate-based posterior approximation
        posterior_surrogate = PDEFwdModelGaussianSurrogate(gp=surrogate,
                                                           batchgp=batchgp,
                                                           posterior=posterior,
                                                           num_rff=num_rff,
                                                           key_rff=key_rff,
                                                           log_prior=posterior.prior.log_density,
                                                           y=posterior.likelihood.observation,
                                                           noise_cov_tril=posterior.likelihood.noise_cov_tril,
                                                           support=posterior.prior.truncated_support)

        self.keys = {'init': key, 'inv_prob': key_inv_prob, 'surrogate': key_surrogate, 'rff': key_rff}
        self.posterior = posterior
        self.posterior_surrogate = posterior_surrogate
        self.ground_truth = ground_truth
        self.design = design
        self.opt_history = opt_history

        # save to disk
        if write_to_file:
            jnp.savez(out_dir / 'design.npz', X=self.design.X, y=self.design.y)


    def __call__(self, 
                 key: PRNGKey,
                 out_dir: Path,
                 mcmc_settings: dict[str, Any] | None = None,
                 mcwmh_settings: dict[str, Any] | None = None,  
                 **kwargs):
        
        key, key_init_mcmc, key_seed_mcmc, key_mcwmh = jr.split(key, 4)

        if mcmc_settings is None:
            mcmc_settings = {'n_samples': 5000, 'n_burnin': 10_000, 'thin_window': 5}
        if mcwmh_settings is None:
            mcwmh_settings = {'n_chains': 100, 'n_samp_per_chain': 10, 'n_burnin': 10_000, 'thin_window': 100}

        # sampling distributions
        dists = {
            'exact': self.posterior,
            'mean': self.posterior_surrogate.expected_surrogate_approx(),
            'eup': self.posterior_surrogate.expected_density_approx()
        }

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
                prop_cov=0.3**2 * jnp.eye(self.posterior.dim),
                **mcmc_settings
            )

            mcmc_samp[dist_name] = mcmc_results['positions'].squeeze(1)
            mcmc_info[dist_name] = mcmc_results

        # expected posterior: MCwMH
        samp_mcwmh = sample_mcwmh(key=key_mcwmh,
                                  posterior_surrogate=self.posterior_surrogate,
                                  prop_cov_init=mcmc_info['mean']['prop_cov'][0],
                                  **mcwmh_settings)
        mcmc_samp['ep_mcwmh'] = samp_mcwmh.reshape(-1, self.posterior.dim)

        # write results
        jnp.savez(out_dir / 'samples.npz', **mcmc_samp)
        jnp.savez(out_dir / 'keys.npz', **{nm: jr.key_data(k) for nm, k in self.keys.items()})

        return None


def sample_mcwmh(key: PRNGKey,
                 posterior_surrogate: PDEFwdModelGaussianSurrogate,
                 n_chains: int,
                 n_samp_per_chain: int,
                 n_burnin: int,
                 thin_window: int,
                 prop_cov_init: Array | None = None,
                 adapt_kwargs=None):
    """Monte Carlo within Metropolis-Hastings approximation of the expected posterior"""

    key_seed_trajectory, key_seed_mcmc, key_init_positions = jr.split(key, 3)

    trajectory_keys = jr.split(key_seed_trajectory, n_chains)
    mcmc_keys = jr.split(key_seed_mcmc, n_chains)

    samp = jnp.zeros((n_chains, n_samp_per_chain, posterior_surrogate.dim))
    initial_positions = posterior_surrogate.posterior.prior.sample(key_init_positions, n_chains)

    for i in range(n_chains):
        post_traj = posterior_surrogate.sample_trajectory(trajectory_keys[i])

        results = sample_distribution(
            key=mcmc_keys[i],
            dist=post_traj,
            initial_position=initial_positions[i:i+1],
            n_samples=n_samp_per_chain,
            n_burnin=n_burnin,
            thin_window=thin_window,
            prop_cov=prop_cov_init,
            adapt_kwargs=adapt_kwargs
        )

        samp = samp.at[i].set(
            results['positions'].squeeze(1)
        )

    return samp


def sample_rkpcn(key: PRNGKey,
                 posterior: Posterior,
                 surrogate_post: PDEFwdModelGaussianSurrogate,
                 initial_position: Array,
                 prop_cov: Array,
                 rho: float,
                 n_samples: int,
                 n_burnin: int = 50_000,
                 thin_window: int = 5):
    """rk-pcn algorithm for approximate EP inference"""
    
    key_ker, key_init_state, key_samp = jr.split(key, 3)    

    # log-density as a function of target function output
    observable_to_logdensity = posterior.likelihood.observable_to_logdensity
    truncated_log_prior = DistributionFromDensity(log_dens=posterior.prior.log_density,
                                                  dim=posterior.dim, support=surrogate_post.support)
    truncated_log_prior_density = truncated_log_prior.log_density

    def log_density(f, u):
        return observable_to_logdensity(f).squeeze() + truncated_log_prior_density(u).squeeze()

    # underlying GP model
    gp = surrogate_post.surrogate

    # settings for f update in sampler
    class UpdateInfo(NamedTuple):
        rho: float
    f_update_info = UpdateInfo(rho=rho)

    init_fn, kernel = init_rkpcn_kernel(key=key_ker,
                                        log_density=log_density,
                                        gp=gp,
                                        f_update_fn=_f_update_pcn_proposal,
                                        f_update_info=f_update_info)
    initial_state = init_fn(key=key_init_state,
                            initial_position=initial_position,
                            prop_cov=prop_cov)                                         
    
    # run sampler
    n_samples_total = n_burnin + thin_window * n_samples
    out = mcmc_loop(key=key_samp,
                    kernel=kernel,
                    initial_state=initial_state,
                    num_samples=n_samples_total)

    samp = out.position[n_burnin:]

    return samp[::thin_window]


# -----------------------------------------------------------------------------
# Helper functions for post-run analysis/plotting
# -----------------------------------------------------------------------------

def summarize_status(base_out_dir, experiment_name, n_design, 
                     required_files=('samples.npz', 'rkpcn_samples.npz')):
    if isinstance(n_design, int):
        n_design = [n_design]

    for n in n_design:
        out_dir = base_out_dir / experiment_name / f'n_design_{n}'
        subdirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith('rep')]
        completed = [
            all((p / file).exists() for file in required_files)
            for p in subdirs
        ]
        print(f'n_design = {n}: {sum(completed)} of {len(completed)} completed.')

def read_samp(base_out_dir, experiment_name, n_design, rep_idx):
    rep_out_dir = base_out_dir / experiment_name / f'n_design_{n_design}' / f'rep{rep_idx}'
    samp = dict(jnp.load(rep_out_dir / 'samples.npz'))
    return samp

def load_rep(base_out_dir, experiment_name, n_design, rep_idx):
    rep_out_dir = base_out_dir / experiment_name / f'n_design_{n_design}' / f'rep{rep_idx}'
    init_settings = jnp.load(rep_out_dir / 'init_settings.npz')
    key_init = jr.wrap_key_data(init_settings['key_init'])

    rep = PDEReplicate(key=key_init,
                       out_dir=rep_out_dir,
                       n_design=n_design,
                       num_rff=init_settings['num_rff'].item(),
                       design_method=init_settings['design_method'].item(),
                       write_to_file=False)
    
    return rep

def samp_trace(base_out_dir, experiment_name, n_design, rep_idx):
    samp = read_samp(base_out_dir, experiment_name, n_design, rep_idx)
    for nm, vals in samp.items():
        for i in range(vals.shape[1]):
            plt.plot(vals[:,i])
        plt.title(nm)
        plt.show()

def samp_pair_plot(base_out_dir, experiment_name, n_design, rep_idx, dist_names=None, key=jr.key(0)):
    samp = read_samp(base_out_dir, experiment_name, n_design, rep_idx)
    samp['prior'] = jr.normal(key, samp['exact'].shape) # N(0, I) prior

    if dist_names is None:
        dist_names = ['prior', 'ep_mcwmh', 'eup', 'mean', 'exact']
    par_names = [f'u{i}' for i in range(1, samp['exact'].shape[1]+1)]

    df_list = []
    for name in dist_names:
        df = pd.DataFrame(samp[name], columns=par_names)
        df['dist'] = name
        df_list.append(df)
    samp_df = pd.concat(df_list, ignore_index=True)

    sns.pairplot(samp_df, hue='dist', diag_kind='kde')

def plot_surrogate_pred(base_out_dir, 
                        experiment_name, 
                        n_design, 
                        rep_idx,
                        key=jr.key(0), 
                        n_test=500):

    rep = load_rep(base_out_dir, experiment_name, n_design, rep_idx)
    test_inputs = rep.posterior.prior.sample_lhc(key, n_test)

    posterior = rep.posterior
    posterior_surrogate = rep.posterior_surrogate

    # ground truth values at test points
    true_forward = posterior.likelihood.forward(test_inputs)
    true_log_post = posterior.likelihood.observable_to_logdensity(true_forward) + posterior.prior.log_density(test_inputs)

    # surrogate predictions at test points
    pred = posterior_surrogate.surrogate(test_inputs)
    mean_forward = pred.mean.T
    sd_forward = pred.stdev.T

    # plug-in mean and marginal predictions
    mean_approx = posterior_surrogate.expected_surrogate_approx()
    eup_approx = posterior_surrogate.expected_density_approx()
    mean_approx_pred = mean_approx.log_density(test_inputs)
    eup_approx_pred = eup_approx.log_density(test_inputs)

    # plot surrogate forward model predictions
    for i in range(true_forward.shape[1]):
        sns.scatterplot(x=true_forward[:,i], y=mean_forward[:,i], color='blue', label='Predictions')
        plt.errorbar(true_forward[:,i], mean_forward[:,i], yerr=2*sd_forward[:,i], fmt='none', ecolor='blue', alpha=0.5)
        plt.plot(true_forward[:,i], true_forward[:,i], 'r--', label='Ground truth')
        plt.xlabel('True Value')
        plt.ylabel('Predicted Mean')
        plt.legend()
        plt.title(f'output {i}')
        plt.show()

    # plug-in mean log-posterior predictions
    sns.scatterplot(x=true_log_post, y=mean_approx_pred, color='blue')
    plt.plot(true_log_post, true_log_post, 'r--')
    plt.xlabel('True log post')
    plt.ylabel('Plug-in mean')
    plt.title('Plug-In Mean Predictions')
    plt.show()


def estimate_mahalanobis_coverage(
    samples: dict[str, Array],
    baseline: str,
    probs: Array,
    jitter: float = 1e-8
) -> dict[str, Array]:
    """
    Estimate joint (ellipsoidal) coverage using Mahalanobis distance.

    This function evaluates how well each approximating distribution in
    `samples` captures the joint uncertainty of a designated baseline
    distribution. Coverage regions are defined as empirical ellipsoids
    induced by the mean and covariance of each approximating distribution.

    For each distribution X and each probability level p in `probs`,
    coverage is computed as follows:

    1. Let X be samples from an approximating distribution with shape
       (n_x, d). Compute the sample mean μ_X and covariance Σ_X.

    2. For each sample x in X, compute the squared Mahalanobis distance:
           r_X^2(x) = (x - μ_X)^T Σ_X^{-1} (x - μ_X)

    3. Define the ellipsoidal p-coverage region as:
           R_p(X) = {x : r_X^2(x) ≤ q_p}
       where q_p is the empirical p-quantile of r_X^2 evaluated on X.

    4. Evaluate coverage with respect to the baseline samples B by
       computing the fraction of baseline points that lie inside R_p(X):
           coverage(p) = mean_{b in B}[ r_X^2(b) ≤ q_p ]

    Args:
        samples:
            Dictionary mapping distribution names to sample arrays.
            Each array must have shape `(n_i, d)`, where `d` is the common
            dimensionality across all distributions.
        baseline:
            Key in `samples` identifying which distribution serves as the
            baseline (ground-truth) distribution.
        probs:
            JAX array of shape `(m,)` containing probability levels in `(0, 1]`.
            Each value defines the nominal coverage level of the ellipsoid.

    Returns:
        A dictionary mapping distribution names to coverage arrays.
        For each key `k != baseline`, the value is a JAX array of shape `(m,)`,
        where entry `i` is the empirical ellipsoidal coverage at level
        `probs[i]`, evaluated against the baseline samples.

        The baseline distribution itself is excluded from the output.

    Example:
        >>> samples = {
        ...     "truth": jnp.random.normal(size=(2000, 6)),
        ...     "approx": jnp.random.normal(size=(1000, 6)),
        ... }
        >>> probs = jnp.array([0.5, 0.9])
        >>> cov = estimate_mahalanobis_coverage(samples, "truth", probs)
        >>> cov["approx"].shape
        (2,)
    """
    baseline_samples = samples[baseline]
    d = baseline_samples.shape[1]

    def mahalanobis_sq(x, mean, chol):
        """
        x: (..., d); mean: (d,); chol: lower Cholesky, (d, d)
        """
        diff = x - mean
        y = solve_triangular(chol, diff.T, lower=True).T
        return jnp.sum(y**2, axis=-1)

    results = {}

    for name, x in samples.items():
        if name == baseline:
            continue

        # Mean and covariance of approximating distribution
        mean_x = jnp.mean(x, axis=0)
        cov_x = jnp.cov(x, rowvar=False) + jitter * jnp.eye(d)
        chol_x = jnp.linalg.cholesky(cov_x, upper=False)

        # Squared Mahalanobis distances
        r2_x = mahalanobis_sq(x, mean_x, chol_x)
        r2_baseline = mahalanobis_sq(baseline_samples, mean_x, chol_x)

        # Ellipsoidal coverage for each probability level
        thresholds = jnp.quantile(r2_x, probs)
        coverage = jnp.mean(r2_baseline[:, None] <= thresholds[None, :], axis=0)

        results[name] = coverage

    return results


def assemble_coverage_reps(base_out_dir, experiment_name, n_design, probs, 
                           approx_dist_names, baseline='exact'):
    """Returns (n_reps, n_approx_dists, n_probs) array of coverage values"""

    dist_order = [baseline] + approx_dist_names
    coverage_list = []

    out_dir = base_out_dir / experiment_name / f'n_design_{n_design}'
    subdirs = [p for p in out_dir.iterdir() if p.is_dir() and p.name.startswith('rep')]
    rep_idcs = [int(p.name.replace('rep', '')) for p in subdirs if (p / 'samples.npz').exists()]
    
    for rep_idx in rep_idcs:
        samples = read_samp(base_out_dir, experiment_name, n_design, rep_idx)
        samples = {nm: samples[nm] for nm in dist_order}
        coverage = estimate_mahalanobis_coverage(samples=samples, 
                                                 baseline=baseline,
                                                 probs=probs)
        coverage_list.append(jnp.stack(list(coverage.values())))

    return jnp.stack(coverage_list)


def wasserstein2_sinkhorn(
    x_ref: jnp.ndarray,
    x_approx: jnp.ndarray,
    epsilon: float = 0.05,
    **kwargs
):
    """
    Compute entropic-regularized W2 distance between two empirical distributions.

    Args:
        x_ref:     (N, d) reference samples
        x_approx:  (M, d) approximating samples
        epsilon:   Sinkhorn regularization strength
        **kwargs:  forwarded to Sinkhorn()

    Returns:
        Scalar W2 distance
    """
    geom = pointcloud.PointCloud(x_ref, x_approx, epsilon=epsilon)
    prob = linear_problem.LinearProblem(geom)
    solver = sinkhorn.Sinkhorn(**kwargs)
    out = solver(prob)
    return jnp.sqrt(out.reg_ot_cost)


def compute_wasserstein_comparison(
    samples: dict,
    reference_key: str,
    subsample: int | None = None,
    key: jax.random.PRNGKey = jax.random.PRNGKey(0),
    epsilon: float | None = None,
    sinkhorn_kwargs: dict | None = None
):
    sinkhorn_kwargs = sinkhorn_kwargs or {}
    ref_samples = samples[reference_key]
    n, d = ref_samples.shape
    
    # whitening matrix (Mahalanobis): Cov[(X-mu) @ W] = I
    mu_ref = jnp.mean(ref_samples, axis=0)
    cov_ref = jnp.cov(ref_samples, rowvar=False) + 1e-8 * jnp.eye(d)
    L = jax.scipy.linalg.cholesky(cov_ref, lower=True)
    W = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(d), lower=False)

    def whiten(samples):
        centered = samples - mu_ref
        return jnp.dot(centered, W)

    # Transform all chains using the reference's geometry
    samples = {
        k: whiten(v) for k, v in samples.items()
    }

    # choose regularization level using reference geometry
    if epsilon is None:
        ref_geom = pointcloud.PointCloud(
            samples[reference_key], 
            samples[reference_key], 
            epsilon=None
        )
        fixed_epsilon = ref_geom.epsilon
    else:
        fixed_epsilon = epsilon

    # optional subsampling
    if subsample is not None:
        for k, v in samples.items():
            key_choice, key, = jr.split(key)
            idx = jr.choice(key_choice, v.shape[0], (subsample,), replace=False)
            samples[k] = v[idx]

    results = {}

    for name, x in samples.items():
        if name == reference_key:
            continue

        w2 = wasserstein2_sinkhorn(samples[reference_key], x, epsilon=fixed_epsilon, **sinkhorn_kwargs)
        results[name] = w2

    return results, fixed_epsilon


def summarize_wasserstein_design_reps(key, base_out_dir, 
                                      experiment_name, n_design, 
                                      rep_idcs, output_dir=None):
    """
    Wasserstein distance to EP for all reps for a certain design size. The same regularization
    parameter epsilon is used across all reps/approximating distributions for consistency.
    """
    w2_keys = jr.split(key, len(rep_idcs))
    design_dir = base_out_dir / experiment_name / f'n_design_{n_design}'
    results = []
    eps = None

    if output_dir is not None:
        output_path = output_dir / f'w2_ndesign_{n_design}.npz'
        write_to_file = True
    else:
        write_to_file = False

    # combine results into single dictionary
    def _combine_results(res):
        keys = res[0].keys()
        results = {k: jnp.stack([rep_result[k] for rep_result in res]) for k in keys}
        return results
    
    for i, rep_idx in enumerate(rep_idcs):
        try:
            print('Rep ', rep_idx)
            rep_dir = design_dir / f'rep{rep_idx}'
            samp = dict(jnp.load(rep_dir / 'samples.npz'))
            rkpcn_samp = dict(jnp.load(rep_dir / 'rkpcn_samples.npz'))
            samp = samp | rkpcn_samp

            rep_results, eps = compute_wasserstein_comparison(
                samples=samp,
                reference_key='ep_mcwmh',
                subsample=1000,
                key=w2_keys[i],
                epsilon=eps,
                sinkhorn_kwargs={'threshold': 1e-6, 'max_iterations': 5000, 'lse_mode': True}
            )
            results.append(rep_results)

            if i > 0 and i % 20 == 0:
                if write_to_file:
                    jnp.savez(output_path, **_combine_results(results))
                jax.clear_caches()
        except Exception as e:
            print('Failed rep:', rep_idx)
            print(e)
    
    results = _combine_results(results)
    return results, eps

