# uncprop/models/vsem/surrogate.py
from __future__ import annotations
from typing import Any

import jax.numpy as jnp
import jax.random as jr
from gpjax import Dataset
from gpjax.gps import Prior as GPPrior
from numpyro.distributions import MultivariateNormal
from jax.scipy.special import logsumexp

from uncprop.custom_types import Array, ArrayLike, PRNGKey
from uncprop.core.distribution import Distribution, DistributionFromDensity
from uncprop.core.inverse_problem import Posterior
from uncprop.utils.gpjax_models import construct_gp, train_gp_hyperpars
from uncprop.utils.gpjax_multioutput import BatchedRBF
from uncprop.utils.grid import normalize_density_over_grid, Grid
from uncprop.core.surrogate import (
    GPJaxSurrogate,
    SurrogateDistribution,
    LogDensGPSurrogate,
    LogDensClippedGPSurrogate,
    construct_design, 
    PredDist,
)


def fit_vsem_surrogate(key: PRNGKey,
                       surrogate_tag: str,
                       posterior: Posterior,
                       n_design: int,
                       design_method: str,
                       gp_train_args: dict | None = None,
                       verbose: bool = True,
                       jitter: float = 0.0) -> tuple[VSEMPosteriorSurrogate, dict]:
    """ Top-level function for fitting VSEM log posterior surrogate

    Note that `posterior` represents the posterior of the exact inverse problem that 
    we are trying to approximate.
    """
    key, key_design = jr.split(key, 2)

    # sample design points
    if design_method == 'lhc':
        prior_sampler = posterior.prior.sample_lhc
    elif design_method == 'uniform':
        prior_sampler = posterior.prior.sample
    else:
        raise ValueError(f'Invalid design method {design_method}')

    design = construct_design(key=key_design,
                              n_design=n_design, 
                              prior_sampler=prior_sampler,
                              f=lambda x: posterior.log_density(x))

    # fit log-posterior surrogate
    if design.in_dim != posterior.dim:
        raise ValueError(f'Design and posterior dim mismatch: {design.in_dim} vs {posterior.dim}')
    if gp_train_args is None:
        gp_train_args = {}
    
    gp_untuned, bijection = construct_gp(design, set_bounds=True)
    gp, opt_info = train_gp_hyperpars(gp_untuned, bijection, design, **gp_train_args)
    if verbose:
        _print_gp_fit_info(gp, opt_info)

    # wrap gpjax object as a GPJAXSurrogate
    ker = gp.prior.kernel
    batched_kernel = BatchedRBF(batch_dim=1, input_dim=ker.n_dims,
                                lengthscale=ker.lengthscale,
                                variance=ker.variance)
    batched_gp_prior = GPPrior(mean_function=gp.prior.mean_function,
                               kernel=batched_kernel)
    batched_gp_post = batched_gp_prior * gp.likelihood
    log_density_surrogate = GPJaxSurrogate(gp=batched_gp_post, design=design, jitter=jitter)

    # surrogate-induced posterior approximation
    if surrogate_tag == 'gp':
        post_surrogate = LogDensGPSurrogate(log_dens=log_density_surrogate,
                                            support=posterior.support)
    elif surrogate_tag == 'clip_gp':
        def log_dens_upper_bound(x: ArrayLike) -> Array:
            return posterior.likelihood.log_density_upper_bound(x) + posterior.prior.log_density(x)

        post_surrogate = LogDensClippedGPSurrogate(log_dens=log_density_surrogate,
                                                   log_dens_upper_bound=log_dens_upper_bound,
                                                   support=posterior.support)
    else:
        raise ValueError(f'Invalid surrogate tag: {surrogate_tag}')
    
    return VSEMPosteriorSurrogate(post_surrogate), opt_info


def _print_gp_fit_info(gp, opt_info):
    start_loss = opt_info['starting_loss']
    end_loss = opt_info['final_loss']
    gp_scale = jnp.sqrt(gp.prior.kernel.variance.get_value())
    gp_ls = gp.prior.kernel.lengthscale.get_value()
    gp_noise_sd = gp.likelihood.obs_stddev.get_value()

    print(f'Initial loss {start_loss}')
    print(f'Final loss {end_loss}')
    print(f'gp scale: {gp_scale}')
    print(f'gp lengthscales: {gp_ls}')
    print(f'gp noise std dev: {gp_noise_sd}')


class VSEMPosteriorSurrogate(SurrogateDistribution):
    """
    A light wrapper around either `LogDensGPSurrogate` or `LogDensClippedGPSurrogate`, with 
    the sole purpose of adding the `expected_normalized_density_approx()` method specialized
    for the grid-based VSEM experiment.

    This is really a temporary hack. A better long-term solution would consist of defining 
    a LogDensSurrogate class.
    """

    def __init__(self,
                 posterior_surrogate: LogDensGPSurrogate | LogDensClippedGPSurrogate, 
                 support: tuple[tuple, tuple] | None = None):
        if not isinstance(posterior_surrogate, (LogDensGPSurrogate, LogDensClippedGPSurrogate )):
            raise ValueError(f'VSEMPosteriorSurrogate wraps either LogDensGPSurrogate or LogDensClippedGPSurrogate')
        self._posterior_surrogate = posterior_surrogate

    @property
    def posterior_surrogate(self):
        return self._posterior_surrogate

    @property
    def surrogate(self):
        return self.posterior_surrogate.surrogate
    
    @property
    def dim(self):
        return self.posterior_surrogate.dim
    
    @property
    def support(self):
        return self.posterior_surrogate.support
    
    def log_density_from_pred(self, pred: PredDist):
        return self.posterior_surrogate.log_density_from_pred(pred)
    
    def expected_surrogate_approx(self) -> DistributionFromDensity:
        return self.posterior_surrogate.expected_surrogate_approx()
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        return self.posterior_surrogate.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        return self.posterior_surrogate.expected_density_approx()
    
    def expected_normalized_density_approx(self,
                                           key: PRNGKey,
                                           *,
                                           grid: Grid,
                                           n_mc: int = 10_000,
                                           **method_kwargs) -> DistributionFromDensity:
        """
        Since the VSEM example is over a 2d input space, we define the expected
        posterior approximation as a Monte Carlo based estimate of the density
        over a 2d grid of points. The argument `grid` is required only to extract
        the cell area, which informs how the density is approximated. The args
        `key`, `n_mc` control the Monte Carlo sampling.

        This works for both Gaussian and clipped Gaussian predictors.       
        """
    
        cell_area = grid.cell_area
        input_dim = self.surrogate.input_dim

        def log_dens(x: ArrayLike):
            log_post_samp = self.posterior_surrogate.sample_surrogate_pred(key, input=x, n=n_mc) # (n_mc, n_x)
            return _estimate_ep_grid(log_post_samp, cell_area=cell_area)

        return DistributionFromDensity(log_dens=log_dens, dim=input_dim)


def _estimate_ep_grid(logpi_samples: Array, cell_area: float | None) -> Array:
    """
    Estimate E_f[ pi(u_j; f) / Z(f) ] at grid nodes using Monte Carlo samples of log-densities.

    Args:
        logpi_samples: ndarray of shape (S, M). Row s contains [ell_j(f^{(s)})]_{j=1..M},
                       where ell_j = log pi(u_j; f^{(s)}).
                       S = number of Monte Carlo samples, M = number of grid nodes.
        cell_area: area of one grid cell, assumed constant across cells.

    Returns:
        ep: ndarray of shape (M,), log{ E_f[ pi(u_j;f)/Z(f) ] }.
        Notice that this is the log of the Monte Carlo estimate.
    """
    logp, _ = normalize_density_over_grid(logpi_samples, cell_area=cell_area, return_log=True)
    n_densities, _ = logp.shape

    # log of normalized ep density 
    log_ep = logsumexp(logp, axis=0) - jnp.log(n_densities) # (n_densities,)

    return log_ep