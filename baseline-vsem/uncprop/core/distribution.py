from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
from numpyro.distributions import MultivariateNormal

from uncprop.custom_types import Array, PRNGKey, ArrayLike


class Distribution(ABC):
    """
    A minimal distribution class that is sufficient for the experiments in
    this paper. Uses basic shape/batch semantics: 
        - the event shape of the distribution is always a flat array, so is defined 
          by a single number, the dimension d (number of scalar elements in one value)
        - batches of points are stored as 2d arrays with shape (n, d)
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[ArrayLike, ArrayLike]:
        """
        Should return tuple of the form lower, upper where lower and upper
        are each tuples of length `dim` giving the dimension-by-dimension
        lower and upper bounds, respectively. Should be `-jnp.inf` or 
        `jnp.inf` for unbounded dimensions. None is interpreted as unconstrained.
        """
        pass

    def sample(self, key: PRNGKey, n: int = 1) -> Array:
        """ Out: (n,d)"""
        raise NotImplementedError('sample() not implemented for distribution object.')

    def log_density(self, x: ArrayLike) -> Array:
        """ In: (n,d) or (d,), Out: (n,)"""
        raise NotImplementedError('log_density() not implemented for distribution object.')
    
    @property
    def mean(self) -> Array:
        """ Return mean of shape (d,) """
        raise NotImplementedError('mean not implemented for distribution object.')

    @property
    def cov(self) -> Array:
        """ Return covariance matrix of shape (d,d) """
        raise NotImplementedError('cov not implemented for distribution object.')

    @property
    def variance(self) -> Array:
        """ Return marginal variances of shape (d,) """
        raise NotImplementedError('variance not implemented for distribution object.')

    @property
    def stdev(self) -> Array:
        """ Return marginal standard deviations of shape (d,) """
        return jnp.sqrt(self.variance)
    

@runtime_checkable
class LogDensity(Protocol):
    def __call__(self, x: ArrayLike) -> Array:
        """ In: (n,d), Out: (n,)"""
        pass


class DistributionFromDensity(Distribution):
    """
    Convenience class for constructing a Distribution given its (unnormalized)
    log density. Passing non-infinite `support` will simply zero out the density
    outside of the support bounds.
    """
    def __init__(self, 
                 log_dens: LogDensity, 
                 dim: int, 
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        
        if not isinstance(log_dens, LogDensity):
            raise ValueError(f'DistributionFromDensity expects LogDensity, got {type(log_dens)}')
        if not isinstance(dim, int):
            raise ValueError(f'DistributionFromDensity expects integer dimension, got {type(dim)}')
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._dim = dim
        self._support = support
        self._log_density = log_dens

    def log_density(self, x: ArrayLike) -> Array:
        x = jnp.atleast_2d(x)
        low, high = self.support
        lp = self._log_density(x)
        lp = jnp.where(jnp.all((x >= low) & (x <= high), axis=1), lp, -jnp.inf)
        return lp
    
    @property
    def dim(self):
        return self._dim

    @property
    def support(self):
        return self._support


# -----------------------------------------------------------------------------
# Wrap numpyro MultivariateNormal as Distribution
# -----------------------------------------------------------------------------

class GaussianFromNumpyro(Distribution):
    """
    Supports a single batch dimension, so this distribution represents either
    a d-dim multivariate normal or a batch multivariate normal with batch size q.
    In this case the mean is (q,d) and the covariance is (q,d,d).
    """

    def __init__(self, dist: MultivariateNormal):
        assert isinstance(dist, MultivariateNormal)
        assert len(dist.batch_shape) < 2

        self._numpyro_mvn = dist

    @property
    def dim(self) -> int:
        return self._numpyro_mvn.event_shape[0]

    @property
    def batch_shape(self) -> int:
        return self._numpyro_mvn.batch_shape

    @property
    def support(self) -> tuple[ArrayLike, ArrayLike]:
        return -jnp.inf, jnp.inf

    def sample(self, key: PRNGKey, n: int = 1) -> Array:
        """ Out: (n,d) or (n,q,d)"""
        return self._numpyro_mvn.sample(key, sample_shape=(n,))

    def log_density(self, x: ArrayLike) -> Array:
        return self._numpyro_mvn.log_prob(x)
    
    @property
    def mean(self) -> Array:
        """ Return mean of shape (d,) or (q,d)"""
        return self._numpyro_mvn.mean

    @property
    def cov(self) -> Array:
        """ Return covariance matrix of shape (d,d) or (q,d,d)"""
        return self._numpyro_mvn.covariance_matrix
    
    @property
    def chol(self) -> Array:
        return self._numpyro_mvn.scale_tril
    @property
    def variance(self) -> Array:
        """ Return marginal variances of shape (d,) or (q,d) """
        return self._numpyro_mvn.variance