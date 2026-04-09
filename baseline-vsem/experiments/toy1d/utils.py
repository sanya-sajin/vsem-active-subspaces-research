# utils.py
# Utility and helper functions for toy 1d example.

from jax import config
config.update('jax_enable_x64', True)
from pathlib import Path
from collections.abc import Callable, Sequence, Mapping
from typing import Any
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jr
import gpjax as gpx
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

from jax.scipy.stats import norm
from numpy.random import default_rng
from jax.scipy.special import logsumexp

from numpyro.distributions import LogNormal
from jax.scipy.stats import norm

from uncprop.custom_types import Array, PRNGKey
from uncprop.utils.grid import Grid, DensityComparisonGrid, normalize_density_over_grid
from uncprop.core.inverse_problem import Prior, Posterior
from uncprop.core.distribution import Distribution, DistributionFromDensity
from uncprop.utils.gpjax_models import construct_gp
from uncprop.utils.distribution import _gaussian_log_density_tril, clipped_gaussian_mean
from uncprop.utils.other import _numpy_rng_seed_from_jax_key
from uncprop.utils.gpjax_multioutput import BatchedRBF
from uncprop.models.vsem.surrogate import _estimate_ep_grid

from uncprop.core.surrogate import (
    SurrogateDistribution,
    construct_design, 
    GPJaxSurrogate, 
    FwdModelGaussianSurrogate,
    LogDensGPSurrogate,
    LogDensClippedGPSurrogate,
)
from uncprop.utils.plot import (
    set_plot_theme, 
    smart_subplots,
    plot_gp_1d,
    plot_marginal_pred_1d,
)


# -----------------------------------------------------------------------------
# Core Surrogate Containers
# -----------------------------------------------------------------------------

@dataclass
class SurrogatePost1d:
    """
    Wrapper arround a SurrogateDistribution with added functionality for 
    producing the plots for the 1d toy example.
    """
    key: PRNGKey
    post_em: SurrogateDistribution
    post_true: Posterior
    filename_label: str
    target_label: str
    grid: Grid
    n_mc: int
    f_target: Callable
    plot_surrogate_fn: Callable | None
    plot_log_dens_surrogate_fn: Callable | None
    plot_dens_surrogate_fn: Callable | None
    surrogate_pred: Distribution | None = None
    
    def __post_init__(self):
        key_samp, key_ep = jr.split(self.key)
        self.surrogate_pred = self.post_em.surrogate(self.grid.flat_grid)
        self.lpost_samp = self.post_em.sample_lpost(key_samp, self.grid.flat_grid, self.n_mc)
        self.density_grid = get_density_grid(key_ep, self.post_em, self.grid, self.post_true)

    def plot_surrogate(self, ax=None, **kwargs):
        if self.plot_surrogate_fn is None:
            return None
        
        fig, ax = self.plot_surrogate_fn(post_em_1d=self, ax=ax, **kwargs)
        ax.set_ylabel(self.target_label)

        return (fig, ax)

    def plot_log_dens_surrogate(self, ax=None, **kwargs):
        if self.plot_log_dens_surrogate_fn is None:
            return None
        
        fig, ax = self.plot_log_dens_surrogate_fn(post_em_1d=self, ax=ax, **kwargs)
        ax.set_ylabel(r'$\log \tilde{\pi}(u)$')
        return fig, ax
    
    def plot_dens_surrogate(self, ax=None, **kwargs):
        if self.plot_dens_surrogate_fn is None:
            return None

        fig, ax = self.plot_dens_surrogate_fn(post_em_1d=self, ax=ax, **kwargs)
        ax.set_ylabel(r'$\tilde{\pi}(u)$')
        return fig, ax
    
    def plot_norm_dens_surrogate(self,
                                 interval_prob: float = 0.95,
                                 gp_colors: dict[str,str] | None = None, 
                                 ax=None, **kwargs):
        grid = self.grid

        # compute mean and quantiles of normalized density trajectories
        post_samp, _ = normalize_density_over_grid(self.lpost_samp, 
                                                   cell_area=grid.cell_area,
                                                   return_log=False)
        stats = calc_dist_from_samples(post_samp, interval_prob=interval_prob)

        # ground truth normalized density
        post_true, _ = normalize_density_over_grid(self.post_true.log_density(grid.flat_grid),
                                                   cell_area=grid.cell_area, return_log=False)

        fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                        mean=stats['mean'],
                                        lower=stats['lower'],
                                        upper=stats['upper'],
                                        points=self.post_em.surrogate.design.X,
                                        true_y=post_true.ravel(),
                                        colors=gp_colors,
                                        ax=ax)
        ax.set_ylabel(r'$\pi(u)$')

        return fig, ax
    
    def plot_post_approx(self,
                         post_ylim: tuple[float,float] | None = None, 
                         post_colors: dict[str,str] | None = None,
                         ax=None, **kwargs):
        # posterior approximation plot
        fig, ax = self.density_grid.plot(normalized=True, 
                                         log_scale=False, 
                                         points=self.post_em.surrogate.design.X,
                                         colors=post_colors,
                                         ax=ax)
        
        ax.set_ylabel(r'$\pi(u)$')
        if post_ylim is not None:
            ax.set_ylim(post_ylim)

        return fig, ax


def wrap_gp(gp, design, jitter):
    """Wrap single output gpjax GP as batch GPJaxSurrogate object"""
    ker = gp.prior.kernel
    batched_kernel = BatchedRBF(batch_dim=1, input_dim=ker.n_dims,
                                lengthscale=ker.lengthscale,
                                variance=ker.variance)
    batched_gp_prior = gpx.gps.Prior(mean_function=gp.prior.mean_function,
                                     kernel=batched_kernel)
    batched_gp_post = batched_gp_prior * gp.likelihood
    return GPJaxSurrogate(gp=batched_gp_post, design=design, jitter=jitter)


def get_density_grid(key: PRNGKey,
                     post_em: SurrogateDistribution,
                     grid: Grid, 
                     post_true: Posterior):
    dists = {
        'exact': post_true,
        'mean': post_em.expected_surrogate_approx(),
        'eup': post_em.expected_density_approx(),
        'ep': post_em.expected_normalized_density_approx(key, grid=grid)
    }

    return DensityComparisonGrid(grid=grid, distributions=dists)


# -------------------------------------------------------------------------
# Helpers for Forward Model Surrogate
# -------------------------------------------------------------------------

class FwdModelGaussianSurrogateGrid(FwdModelGaussianSurrogate):
    """Posterior surrogate induced by forward model surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):

        cell_area = grid.cell_area
        input_dim = self.surrogate.input_dim
        y = self.y
        noise_cov_tril = self.noise_cov_tril

        def log_dens(x):
            log_post_samp = self.sample_lpost(key, x, n=n_mc)
            return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

        return DistributionFromDensity(log_dens=log_dens, dim=input_dim)

    def log_dens_from_target(self, x: Array, f_pred: Array) -> Array:
        y = self.y
        noise_cov_tril = self.noise_cov_tril
        log_prior_dens = self.log_prior(x)

        log_lik_vals = jnp.zeros(f_pred.shape)
        for i in range(f_pred.shape[1]):
            l = _gaussian_log_density_tril(y, m=f_pred[:,i].reshape(-1,1), L=noise_cov_tril)
            log_lik_vals = log_lik_vals.at[:,i].set(l)

        log_post_samp = log_prior_dens + log_lik_vals
        return log_post_samp

    def sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        fwd_samp = self.surrogate(x).sample(key, n=n) # (n, n_x)
        return self.log_dens_from_target(x, fwd_samp) 
    

def plot_log_dens_surrogate_fwd(post_em_1d: SurrogatePost1d,
                                interval_prob: float = 0.95,
                                gp_colors: dict[str,str] | None = None,
                                ax=None, **kwargs):
    assert isinstance(post_em_1d.post_em, FwdModelGaussianSurrogate)
    post_em = post_em_1d.post_em
    post_true = post_em_1d.post_true
    grid = post_em_1d.grid
    pred = post_em_1d.surrogate_pred

    # Empirical statistics from samples
    stats = calc_dist_from_samples(lpost_samp=post_em_1d.lpost_samp,
                                   interval_prob=interval_prob)

    # Can compute the mean analytically
    sigma = post_em.noise_cov_tril.item()
    log_dens_mean = norm.logpdf(post_em.y, loc=pred.mean, scale=sigma) - 0.5 * pred.variance / (sigma**2)

    # Ground truth
    lpost_true = post_true.log_density(grid.flat_grid).ravel()

    fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                    mean=log_dens_mean.ravel(),
                                    lower=stats['lower'],
                                    upper=stats['upper'],
                                    points=post_em.surrogate.design.X,
                                    true_y=lpost_true,
                                    colors=gp_colors,
                                    ax=ax)
    
    return fig, ax



def plot_dens_surrogate_fwd(post_em_1d: SurrogatePost1d,
                            interval_prob: float = 0.95,
                            gp_colors: dict[str,str] | None = None,
                            ax=None, **kwargs):
    assert isinstance(post_em_1d.post_em, FwdModelGaussianSurrogate)
    post_em = post_em_1d.post_em
    post_true = post_em_1d.post_true
    grid = post_em_1d.grid

    # Empirical statistics from samples
    stats = calc_dist_from_samples(lpost_samp=post_em_1d.lpost_samp,
                                   transform='exp',
                                   interval_prob=interval_prob)

    # Ground truth
    lpost_true = post_true.log_density(grid.flat_grid).ravel()

    fig, ax = plot_marginal_pred_1d(x=grid.flat_grid.ravel(),
                                    mean=jnp.exp(post_em_1d.density_grid.log_dens_grid['eup']),
                                    lower=jnp.exp(stats['lower']),
                                    upper=jnp.exp(stats['upper']),
                                    points=post_em.surrogate.design.X,
                                    true_y=jnp.exp(lpost_true),
                                    colors=gp_colors,
                                    ax=ax, **kwargs)
    
    return fig, ax


# -------------------------------------------------------------------------
# Helpers for GP Log-Density Surrogate
# -------------------------------------------------------------------------

class LogDensGPSurrogateGrid(LogDensGPSurrogate):
    """Posterior surrogate induced by log-density surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):
        
            cell_area = grid.cell_area
            input_dim = self.surrogate.input_dim

            def log_dens(x):
                log_post_samp = self.surrogate(x).sample(key, n=n_mc) # (n_mc, n_x)
                return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

            return DistributionFromDensity(log_dens=log_dens, dim=input_dim)
    
    def log_dens_from_target(self, x: Array, f_pred: Array) -> Array:
        return f_pred

    def sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        return self.surrogate(x).sample(key, n=n) # (n, n_x)
    

def plot_lognorm_surrogate(post_em_1d: SurrogatePost1d,
                           interval_prob: float = 0.95,
                           gp_colors: dict[str,str] | None = None,
                           ax=None, **kwargs):
    x = post_em_1d.grid.flat_grid
    surrogate_pred = post_em_1d.surrogate_pred
    design = post_em_1d.post_em.surrogate.design

    return plot_lognorm_1d(x=x.ravel(),
                           mean=surrogate_pred.mean,
                           sd=surrogate_pred.stdev,
                           colors=gp_colors,
                           points=design.X,
                           true_y=post_em_1d.f_target(x),
                           interval_prob=interval_prob,
                           ax=ax, **kwargs)


# -------------------------------------------------------------------------
# Helpers for Clipped GP Log-Density Surrogate
# -------------------------------------------------------------------------

class LogDensClippedGPSurrogateGrid(LogDensClippedGPSurrogate):
    """Posterior surrogate induced by clipped GP log-density surrogate
    
    Defines method to approximate EP over a grid.
    """
    def expected_normalized_density_approx(self,
                                           key,
                                           *,
                                           grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs):
        
            cell_area = grid.cell_area
            input_dim = self.surrogate.input_dim

            def log_dens(x):
                log_post_samp = self.sample_lpost(key, x, n=n_mc) # (n_mc, n_x)
                return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

            return DistributionFromDensity(log_dens=log_dens, dim=input_dim)
    
    def log_dens_from_target(self, x: Array, f_pred: Array) -> Array:
        return f_pred

    def sample_lpost(self, key, x, n=1):
        """Sample realizations of unnormalized log-posterior surrogate at finite set of points"""
        return self.sample_surrogate_pred(key, x, n=n) # (n, n_x)


def plot_clipped_gp_surrogate(post_em_1d: SurrogatePost1d,
                              interval_prob: float = 0.95,
                              gp_colors: dict[str,str] | None = None,
                              ax=None, **kwargs):
    x = post_em_1d.grid.flat_grid
    gp_pred = post_em_1d.surrogate_pred
    design = post_em_1d.post_em.surrogate.design
    upper_bound_fn = post_em_1d.post_em._log_dens_upper_bound
    lpost_true = post_em_1d.post_true.log_density(x)

    stats = calc_clipped_gaussian_stats(m=gp_pred.mean, 
                                        sd=gp_pred.stdev, 
                                        b=upper_bound_fn(x), 
                                        interval_prob=interval_prob)
    return plot_marginal_pred_1d(x=x.ravel(),
                                 mean=stats['mean'],
                                 lower=stats['lower'],
                                 upper=stats['upper'],
                                 points=design.X,
                                 true_y=lpost_true,
                                 colors=gp_colors,
                                 ax=ax)


def plot_clipped_lnp_surrogate(post_em_1d: SurrogatePost1d,
                               interval_prob: float = 0.95,
                               gp_colors: dict[str,str] | None = None,
                               ax=None, **kwargs):
    x = post_em_1d.grid.flat_grid
    gp_pred = post_em_1d.surrogate_pred
    design = post_em_1d.post_em.surrogate.design
    upper_bound_fn = post_em_1d.post_em._log_dens_upper_bound
    lpost_true = post_em_1d.post_true.log_density(x)

    stats = calc_clipped_lognormal_stats(m=gp_pred.mean, 
                                         sd=gp_pred.stdev, 
                                         b=upper_bound_fn(x), 
                                         interval_prob=interval_prob)
    return plot_marginal_pred_1d(x=x.ravel(),
                                 mean=stats['mean'],
                                 lower=stats['lower'],
                                 upper=stats['upper'],
                                 points=design.X,
                                 true_y=jnp.exp(lpost_true),
                                 colors=gp_colors,
                                 ax=ax)


def calc_clipped_gaussian_stats(m, sd, b, interval_prob=0.95):
    """
    Computes mean and confidence interval for the censored random 
    variable Y = min(X, b) where X ~ N(m, sd^2).
    
    Args:
        m: Vector of means for X.
        sd: Vector of standard deviations for X.
        b: Vector of censoring limits (upper bound).
        interval_prob: The probability mass for the credible interval (default 0.95).
        
    Returns:
        Dictionary with 'mean', 'lower', and 'upper'.
    """

    # compute mean
    alpha = (b - m) / sd
    phi_alpha = norm.cdf(alpha)
    pdf_alpha = norm.pdf(alpha)
    mean_val = (m - b) * phi_alpha - sd * pdf_alpha + b

    # compute quantiles
    alpha_level = (1.0 - interval_prob) / 2.0
    p_lower = alpha_level
    p_upper = 1.0 - alpha_level
    q_x_lower = m + sd * norm.ppf(p_lower)
    q_x_upper = m + sd * norm.ppf(p_upper)
    
    # Apply the censoring (min(X, b))
    lower_val = jnp.minimum(q_x_lower, b)
    upper_val = jnp.minimum(q_x_upper, b)
    
    return {
        'mean': mean_val,
        'lower': lower_val,
        'upper': upper_val
    }


def calc_clipped_lognormal_stats(m, sd, b, interval_prob=0.95):
    """
    Computes mean and confidence interval for the censored random 
    variable Y = min(exp(X), exp(b)) where X ~ N(m, sd^2).
    
    Args:
        m: Vector of means for the underlying Gaussian X.
        sd: Vector of standard deviations for the underlying Gaussian X.
        b: Vector of censoring limits (in the Gaussian domain). 
           The actual value limit is exp(b).
        interval_prob: The probability mass for the credible interval (default 0.95).
        
    Returns:
        Dictionary with 'mean', 'lower', and 'upper'.
    """
 
    # compute mean
    alpha = (b - m) / sd    
    term1 = jnp.exp(m + 0.5 * sd**2) * norm.cdf(alpha - sd)
    term2 = jnp.exp(b) * (1.0 - norm.cdf(alpha)) # or norm.cdf(-alpha)
    mean_val = term1 + term2

    # compute quantiles    
    alpha_level = (1.0 - interval_prob) / 2.0
    p_lower = alpha_level
    p_upper = 1.0 - alpha_level
    q_x_lower = m + sd * norm.ppf(p_lower)
    q_x_upper = m + sd * norm.ppf(p_upper)
    lower_val = jnp.exp(jnp.minimum(q_x_lower, b))
    upper_val = jnp.exp(jnp.minimum(q_x_upper, b))
    
    return {
        'mean': mean_val,
        'lower': lower_val,
        'upper': upper_val
    }


# -------------------------------------------------------------------------
# General Plotting helpers
# -------------------------------------------------------------------------

def calc_dist_from_samples(lpost_samp: Array,
                           transform: str | None = None,
                           interval_prob: float = 0.95):
    """
    By default, computes mean and lower/upper quantiles for a set of 
    unnormalized log-density samples. Optionally transforms samples prior
    to computing these quantities. The transform arg accepts None or 'exp'.
     
    The statistics are always returned on the log scale for consistency.
    """
    samp = lpost_samp
    q_lower = 0.5 * (1 - interval_prob)

    if transform is None:
        mean = jnp.mean(samp, axis=0)
    elif transform == 'exp':
        mean = logsumexp(samp, axis=0) - jnp.log(samp.shape[0])
    else:
        raise ValueError(f'Invalid transform: {transform}')

    # Exponential transform simply exponentiates lower/upper
    lower = jnp.quantile(samp, q=q_lower, axis=0)
    upper = jnp.quantile(samp, q=1-q_lower, axis=0)
        
    return {'mean': mean, 'lower': lower, 'upper': upper}


def plot_gp_surrogate(post_em_1d: SurrogatePost1d, 
                      gp_colors: dict[str, str] | None = None, 
                      interval_prob: float = 0.95,
                      ax=None, **kwargs):
    """Save plot summarizing underlying GP emulator distribution
    
    Agnostic to the underlying surrogate target. `f_target` is the true target
    function that is emulated.

    Returns:
        tuple:
            tuple: (figure, axis) objects
            Distribution: surrogate predictive distribution at grid points
    """
    x = post_em_1d.grid.flat_grid
    surrogate_pred = post_em_1d.surrogate_pred
    design = post_em_1d.post_em.surrogate.design

    fig_em, ax_em = plot_gp_1d(x=x.ravel(),
                               mean=surrogate_pred.mean,
                               sd=surrogate_pred.stdev,
                               points=design.X,
                               true_y=post_em_1d.f_target(x),
                               colors=gp_colors,
                               interval_prob=interval_prob,
                               ax=ax)

    return fig_em, ax_em


def plot_lognorm_1d(x,
                    mean,
                    sd,
                    colors=None,
                    points=None,
                    true_y=None,
                    interval_prob=0.95,
                    interval_alpha=0.3,
                    ax=None, **kwargs):
    """
    Plot lognormal process marginals, using log-scale for y-axis.
    true_y should be on log-scale.
    """
    # log of lognormal mean
    log_mean_ln = mean + 0.5 * (sd**2)

    # log of lognormal confidence interval bounds
    z = norm.ppf(1 - (1 - interval_prob) / 2)
    log_lower = mean - z * sd
    log_upper = mean + z * sd

    fig, ax = plot_marginal_pred_1d(x=x,
                                    mean=log_mean_ln,
                                    lower=log_lower,
                                    upper=log_upper,
                                    colors=colors,
                                    points=points,
                                    true_y=true_y,
                                    interval_alpha=interval_alpha,
                                    ax=ax)

    # Force ticks to land on nice powers of 10
    ln_10 = jnp.log(10)
    tick_spacing = ln_10 * 100
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    # Format the label to show 10^integer
    def log_formatter(x, pos):
        exponent = x / ln_10
        return f"$10^{{{int(round(exponent))}}}$"

    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_formatter))

    return fig, ax


# -------------------------------------------------------------------------
# Acquisition Functions
# -------------------------------------------------------------------------

def plot_pw_acq_multi_target(key: PRNGKey,
                             post_em_1d: SurrogatePost1d,
                             n_mc: int = int(1e5),
                             ax=None, **kwargs):
    
    # emulator predictive samples
    grid = post_em_1d.grid
    samp = post_em_1d.post_em.sample_surrogate_pred(key, grid.flat_grid, n=n_mc)
    X = post_em_1d.post_em.surrogate.design.X

    n_plots = 4
    if ax is None:
        fig, ax = plt.subplots(nrows=4, ncols=1)
        ax = ax.ravel()
    else:
        ax = ax.ravel()
        assert len(ax) == n_plots
        fig = ax[0].figure

    # surrogate
    plot_pw_acq(samp, grid, points=X, ax=ax[0], **kwargs)

    # log-density
    ldens_samp = post_em_1d.post_em.log_dens_from_target(grid.flat_grid, samp)
    plot_pw_acq(ldens_samp, grid, points=X, ax=ax[1], **kwargs)

    # density
    dens_samp = jnp.exp(ldens_samp)
    plot_pw_acq(dens_samp, grid, points=X, ax=ax[2], **kwargs)

    # normalized density
    dens_norm_samp, _ = normalize_density_over_grid(ldens_samp, 
                                                    cell_area=grid.cell_area,
                                                    return_log=False)
    plot_pw_acq(dens_norm_samp, grid, points=X, ax=ax[3], **kwargs)

    return fig, ax


def plot_pw_acq(samp: Array,
                grid: Grid,
                points: Array | None = None,
                ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    def scale(x):
        return (x - jnp.min(x)) / (jnp.max(x) - jnp.min(x))
    
    maxvar = -1 * jnp.var(samp, axis=0)
    maxent = -1 * fast_entropy_vmap(samp)
    maxiqr = -1 * (jnp.quantile(samp, q=0.75, axis=0) - jnp.quantile(samp, q=0.25, axis=0))
    
    fig, ax = grid.plot(z=scale(maxvar), points=points, ax=ax, **kwargs)
    fig, ax = grid.plot(z=scale(maxent), ax=ax, **kwargs)
    fig, ax = grid.plot(z=scale(maxiqr), ax=ax, **kwargs)

    return fig, ax


def fast_kde_1d(samples, grid_size=1024, bandwidth_scale=1.0):
    """
    Fast 1D KDE using linear binning and FFT.
    Works with vmap for batch processing.
    """
    n = samples.shape[0]
    
    # Buffer to the range to avoid edge effects
    s_min, s_max = jnp.min(samples), jnp.max(samples)
    range_width = s_max - s_min
    grid_min = s_min - 0.5 * range_width
    grid_max = s_max + 0.5 * range_width
    grid = jnp.linspace(grid_min, grid_max, grid_size)
    dx = grid[1] - grid[0]
    
    # Silverman's Rule for bandwidth
    sigma = jnp.std(samples)
    h = bandwidth_scale * (1.06 * sigma * n**(-1/5))
    
    # Linear Binning (O(n))
    sample_indices = (samples - grid_min) / dx
    lower_indices = jnp.floor(sample_indices).astype(int)
    upper_indices = lower_indices + 1
    weights_upper = sample_indices - lower_indices
    weights_lower = 1.0 - weights_upper
    
    # at2 to scatter weights into the grid
    counts = jnp.zeros(grid_size)
    counts = counts.at[lower_indices].add(weights_lower, indices_are_sorted=False, unique_indices=False)
    counts = counts.at[upper_indices].add(weights_upper, indices_are_sorted=False, unique_indices=False)
    
    # ]Prepare the Gaussian Kernel in Frequency Domain
    # The kernel is centered at zero; we use periodic wrapping for FFT convolution
    # Kernel: K(x) = (1/sqrt(2*pi*h^2)) * exp(-0.5 * (x/h)^2)
    half_grid = (grid_size // 2)
    dist_from_zero = jnp.fft.fftfreq(grid_size, d=1/grid_size) * (grid_max - grid_min)
    kernel_values = (1.0 / (jnp.sqrt(2 * jnp.pi) * h)) * jnp.exp(-0.5 * (dist_from_zero / h)**2)
    
    # Convolution via FFT (O(G log G))
    counts_fft = jnp.fft.fft(counts)
    kernel_fft = jnp.fft.fft(kernel_values)
    density = jnp.fft.ifft(counts_fft * kernel_fft).real
    
    # Normalize
    density = density / (n * dx)
    
    return grid, density

# Vectorized version for many sets of samples
vectorized_kde = jax.jit(jax.vmap(fast_kde_1d, in_axes=(1, None, None), out_axes=(1, 1)))


def estimate_entropy_fast(samples):
    grid, density = fast_kde_1d(samples)
    # Map density back to original samples to compute E[log p(x)]
    log_p = jnp.log(jnp.interp(samples, grid, density) + 1e-10) 
    return -jnp.mean(log_p)

fast_entropy_vmap = jax.jit(jax.vmap(estimate_entropy_fast, in_axes=1))