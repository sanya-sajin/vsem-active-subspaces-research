from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, TypeAlias
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve

from gpjax import Dataset
from gpjax.linalg.utils import add_jitter
from gpjax.gps import ConjugatePosterior
from gpjax.likelihoods import Gaussian as GPJaxGaussianLikelihood
from numpyro.distributions import (
    Distribution as NumpyroDistribution,
    MultivariateNormal,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike
from uncprop.utils.gpjax_multioutput import BatchedStationaryKernel
from uncprop.core.distribution import (
    Distribution, 
    DistributionFromDensity, 
    GaussianFromNumpyro,
)

from uncprop.utils.distribution import (
    clipped_gaussian_mean,
    log_clipped_lognormal_mean,
    _gaussian_log_density_tril
)

# predictive distribution of a surrogate at set of inputs
PredDist: TypeAlias = Distribution


def construct_design(key: PRNGKey,
                     n_design: int, 
                     prior_sampler: Callable[[PRNGKey, int], ArrayLike], 
                     f: Callable) -> Dataset:
    """ Construct design (training) data for training a surrogate

    Sample design input, then evaluate target function f to
    construct design outputs. Return design as gpjax Dataset object.
    The target function should return with a (n,) or (n, q) array 
    when evaluated at a batch of n points.
    """
    x_design = prior_sampler(key, n_design)

    x_design = jnp.asarray(x_design)
    y_design = jnp.asarray(f(x_design))

    if y_design.ndim < 2:
        y_design = y_design.reshape(-1, 1)

    return Dataset(X=x_design, y=y_design)


class Surrogate(ABC):
    """
    A Surrogate `f` is defined as a callable, such that `f(input)` returns
    a Distribution object representing the predictive distribution at the 
    set of points `input`. Note that the dimension of the returned distribution
    will depend on the number of input points.

    This class provides convenience functions to extract certain properties
    of the predictive distribution (mean, cov, variance, standard deviation),
    which are by default simple aliases; e.g., f.mean(input) simply calls
    f(input).mean. If there are more efficient ways to compute these quantities
    without realizing the full predictive distribution, then these methods should
    be overwritten.
    """

    @abstractmethod
    def __call__(self, input: ArrayLike) -> PredDist:
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    def sample_trajectory(self) -> Callable[[ArrayLike], Array]:
        """ Returns a function representing a trajectory of the surrogate 
        
        The function should be vectorized so that it can be evaluated at (n,d)
        input batches, giving the values of the surrogate trajectory at the n points.  
        """
        raise NotImplementedError('sample_trajectory() not implemented by Surrogate object.')

    def mean(self, input: ArrayLike) -> Array:
        return self(input).mean
    
    def cov(self, input: ArrayLike) -> Array:
        return self(input).cov

    def variance(self, input: ArrayLike) -> Array:
        return self(input).variance

    def stdev(self, input: ArrayLike) -> Array:
        return self(input).stdev

        
class SurrogateDistribution(ABC):
    """
    This class represents a random probability distribution, where the
    randomness stems from a Surrogate. A typical example is a
    surrogate-based approximation to an underlying forward model or 
    the log-density, which induces a random distribution.

    More formally, this class represents the random distribution:
        pi(u; f) = p(u; f) / Z(f), Z(f) = int_u p(u; f) du
    where f is a (random) surrogate. Therefore, the p(u; f), Z(f),
    and pi(u; f) represent the random unnormalized density, random
    normalizing constant, and random normalized density, respectively.
    """

    @property
    @abstractmethod
    def surrogate(self) -> Surrogate:
        pass

    @property
    @abstractmethod
    def dim(self) -> int:
        pass

    @property
    @abstractmethod
    def support(self) -> tuple[ArrayLike, ArrayLike]:
        pass

    def log_density_from_pred(self, pred: PredDist) -> PredDist:
        """
        The log density parameterized as a function of the surrogate
        predictive distribution. `pred` represents surrogate predictions
        at a set of points. This function returns a Distribution representing
        the pushforward of the surrogate predictions through the log density
        map. 
        """
        raise NotImplementedError
    
    def log_density(self, input: ArrayLike) -> PredDist:
        """ 
        Same return type as `log_density_from_pred` but first computes
        the surrogate predictions at a set of inputs. Then pushes the 
        distribution through the log density map.
        """
        pred = self.surrogate(input)
        return self.log_density_from_pred(pred)
    
    def sample_surrogate_pred(self, key: PRNGKey, input: ArrayLike, n: int = 1) -> Array:
        """
        Sample the surrogate predictive distribution at a set of inputs.
        """
        return self.surrogate(input).sample(key, n=n)

    def sample_trajectory(self, key: PRNGKey, **kwargs) -> Distribution:
        """ Returns a Distribution representing a trajectory of the random distribution 
        
        Since a SurrogateDistribution is a random distribution, a trajectory of a 
        SurrogateDistribution is a deterministic Distribution. Subclasses will often 
        implement this by sampling a log density trajectory and then returning a 
        DistributionFromDensity. However, there may be other cases.
        """
        raise NotImplementedError('sample_trajectory() not implemented by SurrogateDistribution object.')

    def expected_surrogate_approx(self) -> Distribution:
        """ aka 'plug-in mean' """
        raise NotImplementedError
    
    def expected_log_density_approx(self) -> Distribution:
        raise NotImplementedError
    
    def expected_density_approx(self) -> Distribution:
        """ aka 'expected unnormalized posterior' """
        raise NotImplementedError
    
    def expected_normalized_density_approx(self, *args, **method_kwargs) -> Distribution:
        """ aka 'expected posterior' """
        raise NotImplementedError
    
    def _expected_normalized_imputation(self, 
                                        n_trajectory: int = 100, 
                                        n_samp_per_trajectory: int = 1):
        raise NotImplementedError


# -----------------------------------------------------------------------------
# Log-Density Surrogates
# -----------------------------------------------------------------------------

class LogDensGPSurrogate(SurrogateDistribution):
    """
    A SurrogateDistribution defined by a Gaussian log-density surrogate.
    Concretely, the random distribution is:
        pi(u; f) = exp{f(u)} / Z(f), Z(f) = int_u exp{f(u)} du, where f(u) ~ N(m(u), C(u))

    That is, f is a Gaussian process.
    """

    def __init__(self, 
                 log_dens: GPJaxSurrogate, 
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        if not isinstance(log_dens, GPJaxSurrogate):
            raise ValueError(f'LogDensGPSurrogate requires `log_dens` to be a GPJaxSurrogate, got {type(log_dens)}')
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._surrogate = log_dens
        self._support = support

    @property
    def surrogate(self):
        return self._surrogate
    
    @property
    def dim(self):
        return self.surrogate.input_dim
    
    @property
    def support(self):
        return self._support
    
    def log_density_from_pred(self, pred: PredDist):
        """ Surrogate predictions are log-density predictions """
        return pred
    
    def expected_surrogate_approx(self) -> DistributionFromDensity:
        """ Plug-in surrogate mean as log-density approximation. """

        gp = self.surrogate
        surrogate_mean = lambda x: gp.mean(x)

        return DistributionFromDensity(log_dens=surrogate_mean,
                                       dim=gp.input_dim)
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        """ Surrogate is the log-density, so equivalent to expected_surrogate_approx. """
        return self.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        """ Density surrogate exp{f(u)} is log-normal, so expectation is log-normal mean """
        gp = self.surrogate

        def log_expected_dens(x):
            pred = gp(x)
            return pred.mean + 0.5 * pred.variance
        
        return DistributionFromDensity(log_dens=log_expected_dens,
                                       dim=gp.input_dim)
    

class LogDensClippedGPSurrogate(SurrogateDistribution):
    """
    The same form as LogDensGPSurrogate, but the GP f(u) is "clipped"; i.e., it
    is replaced by g(u) := min{f(u), b(u)} where b(u) is a given pointwise upper
    bound. This is useful when the posterior density being approximated has a 
    known upper bound.

    The GP f(u) is still considered the "surrogate" here (`self.surrogate`); the clipping
    transformation is simply applied post-hoc to any predictive quantities.
    """

    def __init__(self, 
                 log_dens: GPJaxSurrogate,
                 log_dens_upper_bound: Callable[[ArrayLike], Array],
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        if not isinstance(log_dens, GPJaxSurrogate):
            raise ValueError(f'LogDensGPSurrogate requires `log_dens` to be a GPJaxSurrogate, got {type(log_dens)}')
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._surrogate = log_dens
        self._log_dens_upper_bound = log_dens_upper_bound
        self._support = support

    @property
    def surrogate(self):
        return self._surrogate
    
    @property
    def dim(self):
        return self.surrogate.input_dim
    
    @property
    def support(self):
        return self._support
    
    def sample_surrogate_pred(self, key: PRNGKey, input: ArrayLike, n: int = 1) -> Array:
        """
        Clip the Gaussian samples
        """
        gaussian_samp = self.surrogate(input).sample(key, n=n)
        upper_bound = self._log_dens_upper_bound(input).ravel()
        return jnp.clip(gaussian_samp, max=upper_bound)
    
    def log_density_from_pred(self, pred: PredDist):
        return pred
    
    def expected_surrogate_approx(self) -> DistributionFromDensity:
        """ Plug-in surrogate mean as log-density approximation. """
        log_dens_upper_bound = self._log_dens_upper_bound
        gp = self.surrogate

        def surrogate_mean(x):
            gaussian_mean = gp.mean(x)
            gaussian_var = gp.variance(x)
            upper_bound = log_dens_upper_bound(x)
            return clipped_gaussian_mean(gaussian_mean, gaussian_var, upper_bound)

        return DistributionFromDensity(log_dens=surrogate_mean,
                                       dim=gp.input_dim)
    
    def expected_log_density_approx(self) -> DistributionFromDensity:
        """ Surrogate is the log-density, so equivalent to expected_surrogate_approx. """
        return self.expected_surrogate_approx()
    
    def expected_density_approx(self) -> Distribution:
        """ Density surrogate exp{f(u)} is log-normal, so expectation is clipped log-normal mean """
        log_dens_upper_bound = self._log_dens_upper_bound
        gp = self.surrogate

        def log_expected_dens(x):
            gaussian_mean = gp.mean(x)
            gaussian_var = gp.variance(x)
            upper_bound = log_dens_upper_bound(x)
            return log_clipped_lognormal_mean(gaussian_mean, gaussian_var, upper_bound)
        
        return DistributionFromDensity(log_dens=log_expected_dens,
                                       dim=gp.input_dim)


# -----------------------------------------------------------------------------
# Forward Model Surrogates
# -----------------------------------------------------------------------------

class FwdModelGaussianSurrogate(SurrogateDistribution):
    """
    A random distribution induced by a (potentially multi-output) GP pushed 
    through a Gaussian likelihood.

    Assumes the model:
        pi(u; f) = pi_0(u) * N(y | f(u), C), f ~ GP(m, k)

    If y is an (p,) array, then evaluating f(U) on a batch of m inputs should
    produce a GaussianFromNumpyro with event shape (m,) and batch shape (p,).
    This implies the mean/variance will be (p,m) and covariance/cholesky
    will be (p,m,m).

    Notes:
        If the multi-output GP models correlation across outputs, at present
        this correlation is not used in this class - for now treats all 
        GPs as batch independent multioutput models.
    """

    def __init__(self,
                 gp: GPJaxSurrogate,
                 log_prior: Callable,
                 y: Array,
                 noise_cov_tril: Array,
                 support: tuple[ArrayLike, ArrayLike] | None = None):
        
        if support is None:
            support = (-jnp.inf, jnp.inf)

        self._support = support
        self._surrogate = gp
        self.log_prior = log_prior
        self.y = y
        self.noise_cov_tril = noise_cov_tril

    @property
    def surrogate(self) -> Surrogate:
        return self._surrogate

    @property
    def dim(self) -> int:
        return self.surrogate.input_dim

    @property
    def support(self) -> tuple[ArrayLike, ArrayLike]:
        return self._support
    
    @property
    def noise_cov(self):
        return self.noise_cov_tril @ self.noise_cov_tril.T

    def expected_surrogate_approx(self) -> Distribution:
        gp = self.surrogate
        log_prior = self.log_prior
        y = self.y
        noise_cov_tril = self.noise_cov_tril

        def logdensity(x):
            m_x = jnp.atleast_2d(gp(x).mean).T
            log_prior_x = log_prior(x)
            return log_prior_x + _gaussian_log_density_tril(y, m=m_x, L=noise_cov_tril)

        return DistributionFromDensity(log_dens=logdensity,
                                       dim=gp.input_dim,
                                       support=self.support)
    
    
    def expected_density_approx(self) -> Distribution:
        """
        The expected likelihood is:

            E[N(y | f(u), C)] = N(y | m(u), C + k(u))

        Notes:
            The Cholesky factor for C + k(u) can be updated more efficiently using
            rank one Cholesky updates, but for simplicity the full Cholesky is 
            recomputed for now.
        """
        gp = self.surrogate
        log_prior = self.log_prior
        y = self.y
        noise_cov = self.noise_cov

        def logdensity(x):
            pred = gp(x)
            mean_x = jnp.atleast_2d(pred.mean).T
            var_x = jnp.atleast_2d(pred.variance).T # (n, p)
            log_prior_x = log_prior(x)
            C_x = noise_cov[None] + jax.vmap(jnp.diag)(var_x) # (n, p, p)
            L_x = jnp.linalg.cholesky(C_x, upper=False)
            return log_prior_x + _gaussian_log_density_tril(y, m=mean_x, L=L_x)

        return DistributionFromDensity(log_dens=logdensity,
                                       dim=gp.input_dim,
                                       support=self.support)


# -----------------------------------------------------------------------------
# gpjax wrapper used for all GP surrogates
# -----------------------------------------------------------------------------


class GPJaxSurrogate(Surrogate):
    """
    Wrapper around a gpjax conjugate posterior object. The primary reasons
    for this wrapper are to:
    (1) automatically wrap gpjax Gaussian predictions as GaussianFromNumpyro objects,
        so they are valid Distributions.
    (2) implement an update method for fast GP conditioning at new points.
    (3) support vectorized prediction for independent multioutput GPs.

    The `condition_then_predict()` method is written to return the conditional predictions
    without altering the current GP. This design is motivated by the primary use case in 
    the rk-pcn algorithm, which does not require repeatedly conditioning the GP. A copy
    method can easily be added to return an updated conditional GPJaxSurrogate when this
    functionality is required.

    IMPORTANT: 
        gpjax kernels do not correctly batch over kernel hyperparameters. This
        class requires that the GP prior kernel inherit from BatchedStationaryKernel
        to ensure proper batching. For now, we do not pose any such restrictions on
        the mean function - the constant mean function returns (n, q) so we simply 
        transpose. Not guaranteed that other mean functions follow this convention.

    Notes:
        - `jitter` is a value added to the diagonal of any predictive covariance. Note that 
        the gpjax jitter is added regardless; this provides an opportunity to increase the 
        jitter if necessary.
    """
    def __init__(self, 
                 gp: ConjugatePosterior, 
                 design: Dataset, 
                 jitter: float = 0.0):
        assert isinstance(gp, ConjugatePosterior)
        assert isinstance(design, Dataset)
        assert isinstance(gp.likelihood, GPJaxGaussianLikelihood)
        assert isinstance(gp.prior.kernel, BatchedStationaryKernel)

        self.gp = gp
        self.design = design

        # jitter is sum of gp jitters and additional jitter
        self.jitter = jnp.broadcast_to(jitter + gp.jitter, self.output_dim)

        # ensure noise covariances are (q,) array
        sig2_obs = jnp.square(self.gp.likelihood.obs_stddev.get_value())
        self.sig2_obs = sig2_obs.reshape(self.output_dim)

        # inverse kernel matrix
        self.P = self._compute_kernel_precision()

    @property
    def input_dim(self):
        return self.design.in_dim
    
    @property
    def output_dim(self):
        return self.design.y.shape[1]
    
    def __call__(self, input: ArrayLike) -> GaussianFromNumpyro:
        return self.predict(input)

    def predict(self, input: ArrayLike) -> GaussianFromNumpyro:
        return self._predict_using_precision(input, self.P, self.design)

    def condition_then_predict(self, 
                               input: ArrayLike,
                               given: tuple[Array, Array]):
        """
        Condition on new design points, then predict using conditioned GP.
        """        
        P_new, design_new = self._update_conditioning_cache(given)
        return self._predict_using_precision(input, P_new, design_new)


    def _predict_using_precision(self, 
                                 x: ArrayLike, 
                                 P: Array, 
                                 design: Dataset) -> GaussianFromNumpyro:
        """ 
        Compute posterior GP multivariate normal prediction at test inputs `x`, conditional
        on dataset `design`. `P` is the inverse of the kernel matrix (including the noise covariance)
        evaluated at `design.X`.

        Notes:
            - Predictions include the observation noise (predictive distribution of y, not f).
            - In single-output case, the batch dimension is dropped in the predictive distribution.
        """
        x = jnp.asarray(x).reshape(-1, self.input_dim)
        X, Y = design.X, self._flip(design.y)
        n, q = X.shape[0], self.output_dim
        m = x.shape[0]

        # prior means - gpjax mean always returns (n, q)
        meanf = self.gp.prior.mean_function
        mx = self._flip(meanf(x))
        mX = self._flip(meanf(X))

        # prior covariances
        ker = self.gp.prior.kernel
        kx = ker.gram(x).to_dense()
        kxX_P, kxX = self._compute_kxX_P(x, P, design)

        # conditional mean and covariance
        m_pred = mx[..., None] + kxX_P @ (Y - mX)[..., None]
        m_pred = m_pred.squeeze(-1)
        k_pred = kx - kxX_P @ jnp.transpose(kxX, axes=(0, 2, 1))

        # add observation noise to latent variance predictions
        k_pred = k_pred + self.jittered_noise_cov(m)

        if q == 1:
            m_pred = m_pred.squeeze(0)
            k_pred = k_pred.squeeze(0)

        gaussian_pred = MultivariateNormal(m_pred, k_pred)
        return GaussianFromNumpyro(gaussian_pred)


    def _compute_kxX_P(self,
                       x: ArrayLike, 
                       P: Array, 
                       design: Dataset):
        """
        Computes k(x, X) @ P, where X are the training inputs and
        x are m test inputs. P is the inverse kernel matrix k(X)^{-1}.

        Returns:
            tuple, with:
                kxX_P: (q, m, n)
                kxX: (q, m, n)
        """
        X = design.X

        kxX = self.gp.prior.kernel.cross_covariance(x, X)
        kxX_P = kxX @ P # (q, m, n)

        return kxX_P, kxX
    

    def _update_conditioning_cache(self, given: tuple[ArrayLike, Dataset]):
        """ 
        Updates the inverse kernel matrix P = (K + sig2*I)^{-1} and the set
        of design points when a batch of new conditioning points is added.
        
        Args:
            tuple, containing:
                Xnew: (m, d) set of new conditioning inputs
                ynew: (m, q) response values at the conditioning inputs

        Returns:
            tuple, containing:
                Pnew: (q, n+m, n+m), the updated batched inverse kernel matrix
                dataset_new: the updated set of training data
        """
        # update dataset
        Xnew = given[0].reshape(-1, self.input_dim)
        ynew = given[1].reshape(-1, self.output_dim)
        new_dataset = self.design + Dataset(X=Xnew, y=ynew)

        n = self.design.n
        X_all = new_dataset.X
        n_all = X_all.shape[0]
        
        # update inverse kernel matrix
        def _step(carry, xnew):
            P_curr, idx = carry
            P_new = self._update_single_point_kernel_precision(P_curr, X_all, xnew, idx)
            return (P_new, idx+1), None

        P_pad = jnp.zeros((self.output_dim, n_all, n_all))
        P_pad = P_pad.at[:, :n, :n].set(self.P)
        carry0 = (P_pad, n)
        (P_final, _), _ = jax.lax.scan(_step, carry0, Xnew)

        return P_final, new_dataset

    
    def _update_single_point_kernel_precision(self, P, X, x, idx):
        """
        JAX-safe update of the inverse kernel matrix when a single conditioning
        point is added.

        This function performs a rank-1 block update of the precision matrix
        corresponding to appending a new design point, without changing array
        shapes. The full precision matrix `P` is assumed to have static shape
        `(q, N, N)`, where only the top-left `idx × idx` block is currently active.

        The update activates row/column `idx` and leaves all other inactive
        entries untouched. This makes the function compatible with `jit`,
        `vmap`, and `lax.scan`.

        Args:
            P:
                (q, N, N) array.
                Current batched inverse kernel matrix, with only the top-left
                `idx × idx` block active.
            X:
                (N, d) array.
                The current conditioning inputs, where only the first `idx` rows are active.
            x:
                (d,) or (1, d) array.
                New conditioning input to be added.
            idx:
                int scalar.
                Number of active conditioning points before the update.

        Returns:
            P_new:
                (q, N, N) array.
                Updated inverse kernel matrix with row/column `idx` activated.
        """
        q = self.output_dim
        N = P.shape[-1]

        x = x.reshape(1, -1)
        kernel = self.gp.prior.kernel

        # Masks selecting active points
        mask = jnp.arange(N) < idx                      # (N,)
        mask_col = mask[None, :, None]                  # (1, N, 1)
        mask_row = mask[None, None, :]                  # (1, 1, N)
        P = P * mask_row * mask_col

        # Kernel vector between existing points and new point
        kXx = kernel.cross_covariance(X, x)              # (q, N, 1)
        kXx = kXx * mask_col                             # zero out inactive rows
        P_kXx = P @ kXx                                  # (q, N, 1)
        kxX = jnp.transpose(kXx, (0, 2, 1))              # (q, 1, N)

        kx = kernel.gram(x).to_dense().reshape(q)        # (q,)
        kappa = (
            kx
            + self.sig2_obs
            + self.jitter
            - (kxX @ P_kXx).reshape(q)
        )                                                # (q,)
        kappa = kappa[:, None, None]                     # (q, 1, 1)

        # Rank-1 update for top-left block
        outer = P_kXx @ jnp.transpose(P_kXx, (0, 2, 1))  # (q, N, N)
        P_updated = P + outer / kappa

        # New final column / row
        col = -P_kXx / kappa                             # (q, N, 1)
        row = jnp.transpose(col, (0, 2, 1))              # (q, 1, N)
        inv_kappa = 1 / kappa

        P_updated = P_updated.at[:, :, idx].set(col.squeeze(-1))
        P_updated = P_updated.at[:, idx, :].set(row.squeeze(1))
        P_updated = P_updated.at[:, idx, idx].set(inv_kappa.reshape(q))

        return P_updated


    def _compute_kernel_precision(self):
        """ 
        Batch inverse of the prior kernel evaluated at self.design.X.
        Includes the jitter and likelihood noise covariance.

        Return shape: (q, n, n)
        """
        n = self.design.n
        X = self.design.X
        ker = self.gp.prior.kernel

        # Cholesky factor of kernel matrix
        K = ker.gram(X).to_dense() + self.jittered_noise_cov(n)
        L = jnp.linalg.cholesky(K, upper=False)

        # Invert kernel matrix
        I = self._batch_diagonal_matrix(1.0, n)
        P = cho_solve((L, True), I)

        return P


    def noise_cov(self, size):
        """ 
        Batch of GP likelihood noise covariance matrices, each diagonal:
            [sig2_1 * I_size, ..., sig2_q * I_size]

            Return shape: (q, size, size)
        """
        return self._batch_diagonal_matrix(self.sig2_obs, size)
    

    def jitter_matrix(self, size):
        """ 
        Batch of jitter matrices, each diagonal:
            [jitter_1 * I_size, ..., jitter_q * I_size]

            Return shape: (q, size, size)
        """
        return self._batch_diagonal_matrix(self.jitter, size)
    

    def jittered_noise_cov(self, size):
        return self.noise_cov(size) + self.jitter_matrix(size)
    

    def _batch_diagonal_matrix(self, diagonal, size):
        """ 
        Given array `diagonal` broadcastable to (q,) creates batch diagonal matrix:
            [d1 * I_size, ..., dq * I_size]

            Return shape: (q, size, size)
        """
        q = self.output_dim
        diagonal = jnp.broadcast_to(diagonal, (q,))
        
        eye = jnp.eye(size)[None, :, :]
        eye = jnp.broadcast_to(eye, (q, size, size))
        batch_diagonal_matrix = diagonal[:, None, None] * eye

        return batch_diagonal_matrix

    @staticmethod
    def _flip(a):
        """Move last axis to the first position"""
        return jnp.moveaxis(a, -1, 0)