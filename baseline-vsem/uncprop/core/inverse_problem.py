from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable, TypeAlias, Any
from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax.random as jr

from uncprop.custom_types import Array, PRNGKey, ArrayLike
from uncprop.core.samplers import init_rwmh_kernel, mcmc_loop
from uncprop.core.distribution import Distribution, LogDensity


# -----------------------------------------------------------------------------
# Light wrappers around Distribution classes for high-level Bayesian
# inverse problem API
# -----------------------------------------------------------------------------

class Prior(Distribution):
    """
    Simply a Distribution that is required to have names associated with each
    dimension.
    """

    @property
    @abstractmethod
    def par_names(self) -> list:
        """ Returns list of parameter names of length `self.dim` """
        pass

LogLikelihood: TypeAlias = LogDensity

class Posterior(Distribution):
    """
    A Distribution representing the posterior distribution of a Bayesian inference
    problem, defined by specifying a Prior and Likelihood.
    """

    def __init__(self, prior: Prior, likelihood: LogLikelihood):
        if not isinstance(prior, Prior):
            raise ValueError(f'prior must be Prior, got {type(prior)}')
        if not isinstance(likelihood, LogLikelihood):
            raise ValueError(f'likelihood must be LogLikelihood/LogDensity, got {type(likelihood)}')

        self.prior = prior
        self.likelihood = likelihood

    def log_density(self, x: ArrayLike):
        return self.prior.log_density(x) + self.likelihood(x)
    
    @property
    def dim(self):
        return self.prior.dim
    
    @property
    def support(self):
        return self.prior.support
    
    def _get_log_density_function(self) -> Callable:
        """
        Returns a callable JAX-compatible log-density, suitable for passing 
        to blackbox sampling algorithms. Note that the default implmentation 
        simply returns `self.log_density`. Subclasses will often want to override 
        to apply change of variables adjustment to transform to unconstrained space.
        prior.log_density and likelihood must be jitable for the returned function
        to be jitable.
        """
        prior_logp = self.prior.log_density
        log_lik = self.likelihood

        def logp(x):
            return (prior_logp(x) + log_lik(x)).squeeze()
        
        return logp

    def sample(self,
               key: PRNGKey, 
               n: int = 1, 
               prop_cov: Array | None = None,
               initial_position: Array | None = None,
               **kwargs) -> Array:
        """ Default sampling method: may be overriden by subclasses
        
        The default sampler is the blackjax implementation of random walk Metropolis-Hastings.
        """
        key_init_position, key_init_kernel, key_sample = jr.split(key, 3)

        # target density
        logdensity = self._get_log_density_function()

        # default proposal covariance (diagonal)
        if prop_cov is None:
            low, high = self.support
            infinite_support = jnp.any(jnp.isinf(low) | jnp.isinf(high))
            if infinite_support:
                prop_cov = jnp.tile(1.0, self.dim)
            else:
                prop_cov = 0.1 * (jnp.broadcast_to(high, self.dim) - jnp.broadcast_to(low, self.dim))
                prop_cov = prop_cov ** 2

        # initial condition
        if initial_position is None:
            initial_position = self.prior.sample(key_init_position).squeeze()

        # initialize state and kernel
        init_state, kernel = init_rwmh_kernel(key=key_init_kernel,
                                              logdensity=logdensity,
                                              initial_position=initial_position,
                                              prop_cov=prop_cov)

        # run sampler
        states = mcmc_loop(key_sample, kernel, init_state, num_samples=n)

        return jax.block_until_ready(states.position)
