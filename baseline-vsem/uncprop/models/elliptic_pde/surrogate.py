# uncprop/models/elliptic_pde/surrogate.py
from __future__ import annotations
from collections.abc import Callable

from copy import deepcopy
import jax
import jax.numpy as jnp
import jax.random as jr

import optax
import gpjax as gpx
from gpjax import Dataset
from gpjax.parameters import DEFAULT_BIJECTION
from gpjax.gps import _build_fourier_features_fn
from numpyro.distributions import MultivariateNormal

from uncprop.custom_types import PRNGKey, Array, ArrayLike
from uncprop.core.distribution import DistributionFromDensity
from uncprop.core.surrogate import GaussianFromNumpyro
from uncprop.core.inverse_problem import Posterior
from uncprop.core.surrogate import construct_design, GPJaxSurrogate, FwdModelGaussianSurrogate
from uncprop.utils.gpjax_models import construct_gp
from uncprop.utils.gpjax_multioutput import (
    BatchedRBF,
    BatchIndependentGP,
    fit_batch_independent_gp,
    get_batch_gp_from_template,
)


def fit_pde_surrogate(key: PRNGKey,
                      posterior: Posterior,
                      n_design: int,
                      design_method: str,
                      gp_train_args: dict | None = None,
                      verbose: bool = True,
                      jitter: float = 0.0,
                      **fit_kwargs):
    """ Top-level function for fitting PDE forward model surrogate

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
                              f=posterior.likelihood.forward_model)
    
    # fit surrogate
    surrogate, batchgp, history = fit_pde_experiment_batch_gp(design, **fit_kwargs)

    return design, surrogate, batchgp, history



def fit_pde_experiment_batch_gp(design: Dataset,
                                learning_rate: float = 0.1,
                                num_iters: int = 1000,
                                bijection: dict = DEFAULT_BIJECTION):
    
    optim = optax.adam(learning_rate)
    objective = lambda p, d: -gpx.objectives.conjugate_mll(p, d)

    def gp_factory(dataset):
        return construct_gp(dataset, set_bounds=False, prior_mean='constant')[0]
    
    # hyperparameter optimization
    batchgp = get_batch_gp_from_template(gp_factory, design)
    batchgp, history = fit_batch_independent_gp(
        batch_gp=batchgp,
        objective=objective,
        optim=optim,
        num_iters=num_iters
    )

    # surrogate model
    surrogate = convert_gp_to_batch_kernel(batchgp.batch_posterior, design)
    
    return surrogate, batchgp, history


def convert_gp_to_batch_kernel(gp: gpx.gps.ConjugatePosterior, design: Dataset):
    gp = deepcopy(gp)
    batch_dim = design.y.shape[1]
    kernel = gp.prior.kernel
    meanf = gp.prior.mean_function
    likelihood = gp.likelihood

    batched_kernel = BatchedRBF(batch_dim=batch_dim,
                                input_dim=design.in_dim,
                                lengthscale=kernel.lengthscale, 
                                variance=kernel.variance)

    batched_prior = gpx.gps.Prior(mean_function=meanf, kernel=batched_kernel)
    batched_posterior = batched_prior * likelihood
    surrogate = GPJaxSurrogate(batched_posterior, design)

    return surrogate


class PDEFwdModelGaussianSurrogate(FwdModelGaussianSurrogate):
    """
    A FwdModelGaussianSurrogate specialized for the PDE experiment surrogate.
    Requires additional arguments related to the approximation of surrogate
    trajectories using random Fourier features/pathwise sampling.
    """
    
    def __init__(self,
                 gp: GPJaxSurrogate,
                 batchgp: BatchIndependentGP,
                 posterior: Posterior,
                 num_rff: int,
                 key_rff: PRNGKey,
                 log_prior: Callable,
                 y: Array,
                 noise_cov_tril: Array,
                 support: tuple[ArrayLike, ArrayLike] | None = None):

        super().__init__(gp=gp, 
                         log_prior=log_prior, 
                         y=y, 
                         noise_cov_tril=noise_cov_tril,
                         support=support)
        
        self.batchgp = batchgp
        self.posterior = posterior
        self.num_rff = num_rff

        # finite-dimensional basis and distribution over basis coefficients for sampling
        # approximate trajectories
        self.basis_fn = _build_batch_basis_funcs(key=key_rff, 
                                                 surrogate=self.surrogate,
                                                 batchgp=self.batchgp,
                                                 num_rff=self.num_rff)
        self.basis_coef_dist = _build_batch_basis_noise_dist(surrogate=self.surrogate,
                                                             batchgp=self.batchgp,
                                                             num_rff=self.num_rff)


    def sample_trajectory(self, key: PRNGKey, **kwargs):
         
        basis_coef = self.basis_coef_dist.sample(key)
        trajectory = sample_approx_trajectory(self.basis_fn, basis_coef, self.surrogate, self.num_rff)

        observable_to_loglik = self.posterior.likelihood.observable_to_logdensity
        log_prior = lambda u: self.posterior.prior.log_density(u).squeeze()

        def logdensity(u):
            u = jnp.atleast_2d(u)
            fu = trajectory(u).squeeze(0).T # (dim_u, dim_output)
            return observable_to_loglik(fu) + log_prior(u)
        
        return DistributionFromDensity(logdensity, 
                                       dim=self.dim, 
                                       support=self.support)


# -----------------------------------------------------------------------------
# Helpers for approximate trajectory sampling with batch independent GPs
#   Uses Matheron pathwise sampling approximation
# -----------------------------------------------------------------------------

def sample_approx_trajectory(basis_fn: Callable, 
                             noise_realization: Array,
                             surrogate: GPJaxSurrogate, 
                             num_rff: int):
    """ Generate a function representing a surrogate trajectory/sample path

    The trajectory is batched such that it can represent multiple independent
    trajectories of a batch of independent GPs. The notation used in the
    function comments is:

    n = number design points
    q = number of output variables
    b = total number of basis functions; total basis dim is b = b_rff + n
    t = number of trajectories being sampled

    Phi and K are used to denote the RFF and canonical bases, respectively.
    The trajectory can be evaluated at arbitrary batches of inputs, and m 
    is used to denote the size of the input batch.

    Args:
        basis_fn:
            function of the form returned by `_build_batch_basis_funcs()`.
        noise_realization:
            shape (t, q, b), as returned by the distriution returned
            by `_build_batch_basis_noise_dist()`.
        surrogate:
            GPJaxSurrogate instance
        num_rff:
            number of random Fourier features. Note that the dimension of the RFF basis
            is twice this number.

    Returns:
        function representing multiple independent surrogate trajectories of a BatchIndependentGP.
        The function can be evaluated at any inputs.
    """

    X = surrogate.design.X 
    Y = surrogate.design.y # (n, q)
    t, q, _ = noise_realization.shape
    n = X.shape[0]

    meanf = surrogate.gp.prior.mean_function
    Y = (Y - meanf(X)).T # (q, n)
    
    # separate into two noise sources
    dim_rff_basis = 2 * num_rff
    rff_weight = noise_realization[:, :, :dim_rff_basis]       # (t, q, b_rff)
    likelihood_noise = noise_realization[:, :, dim_rff_basis:] # (t, q, n)

    # basis functions evaluated at design points
    Phi_X, _ = basis_fn(X)                         # (q, n, b_rff)
    Phi_X_w = Phi_X[None] @ rff_weight[..., None]  # (t, q, n, 1)
    Phi_X_w = Phi_X_w.squeeze(-1)                  # (t, q, n)

    # canonical weights
    Y_term = Y[None] + likelihood_noise - Phi_X_w              # (t, q, n)
    canonical_weights = surrogate.P[None] @ Y_term[..., None]  # (t, q, n, 1)
    canonical_weights = canonical_weights.squeeze(-1)          # (t, q, n)

    def trajectory(x: Array) -> Array:
        """
        input is shape (m, d) [m points in d dimensions]

        Returns:
            Array of shape (t, q, m), consisting of the t trajectories
            of the batch of q independent GPs evaluated at the m 
            input points.
        """
        Phi_x, _ = basis_fn(x) # (q, m, b_rff)
        mx = meanf(x).T        # (q, m)

        Phi_x_w = Phi_x[None] @ rff_weight[..., None] # (t, q, m, 1)
        Phi_x_w = Phi_x_w.squeeze(-1)                 # (t, q, m)

        # data update term
        kxX = surrogate.gp.prior.kernel.cross_covariance(x, X)    # (q, m, n)
        canonical_term = kxX[None] @ canonical_weights[..., None] # (t, q, m, 1)

        zero_mean_result = Phi_x_w + canonical_term.squeeze(-1)   # (t, q, m)
        result = mx[None] + zero_mean_result

        return result
    
    return trajectory


def _build_batch_basis_funcs(key: PRNGKey,
                             surrogate: GPJaxSurrogate,
                             batchgp: BatchIndependentGP,
                             num_rff: int):
    """ Evaluate pathwise sampling basis functions at test inputs

    This is partially a generalization of the gpjax function _build_fourier_features_fn 
    to accept a BatchIndependentGP. However, it also returns the "canonical features"
    to the RFF, as used in the Matheron/pathwise sampling approach. 

    This is intended to be called one prior to running an inference algorithm. It should
    not be jitted. Note that `surrogate` and `batchgp` should be in direct correspondence -
    ideally this function would not require both arguments.
    """
    dim_out = batchgp.dim_out
    keys = jr.split(key, dim_out)
    posteriors = batchgp.posterior_list

    single_output_rff_funcs = [
        _build_fourier_features_fn(prior=post.prior, num_features=num_rff, key=k)
        for post, k, in zip(posteriors, keys)
    ]

    def basis_fn(test_inputs: Array) -> tuple[Array, Array]:
        """ Evaluates basis functions at test_inputs
        Returns (dim_out, m, n_basis_rff), (dim_out, m, n) where m is number of 
        test points. Note that the number of RFF basis functions is two times num_rff.
        """
        Phi_rff = jnp.stack([fn(test_inputs) for fn in single_output_rff_funcs])
        Phi_canonical, _ = surrogate._compute_kxX_P(test_inputs, surrogate.P, surrogate.design)

        return Phi_rff, Phi_canonical

    return basis_fn


def _build_batch_basis_noise_dist(surrogate: GPJaxSurrogate,
                                  batchgp: BatchIndependentGP,
                                  num_rff: int) -> GaussianFromNumpyro:
    """ Build Gaussian distribution driving noise in pathwise sampling

    The Matheron pathwise sampling approach is driven by finite-dimensional
    Gaussian noise. This includes the RFF weight and the noise term in the 
    GP likelihood. This function returns a single multivariate normal
    distribution that captures these two (independent) noise sources.

    Notes:
        - building a large Gaussian vector for this is not efficient, but is
          done for convenience here with regards to use in the rk-pcn implemention,
          which currently assumes a single Gaussian noise source.
        - note that this does NOT return a distribution over the weights for the 
          basis functions. The RFF portion of the distribution is the distribution
          over the RFF weights. The canonical weights are a deterministic function
          of the RFF weights and the GP likelihood noise term.
    """
    
    dim_rff_basis = 2 * num_rff
    dim_canonical_basis = surrogate.design.n
    dim_batch = batchgp.dim_out

    # rff weights: N(0, I)
    I_rff = jnp.eye(dim_rff_basis)
    rff_weight_cov = jnp.broadcast_to(I_rff, (dim_batch, dim_rff_basis, dim_rff_basis))

    # noise in canonical portion: eps ~ N(0, sig2 * I)
    I_can = jnp.eye(dim_canonical_basis)[None, :, :]
    I_can = jnp.broadcast_to(I_can, (dim_batch, dim_canonical_basis, dim_canonical_basis))
    sig2 = jnp.broadcast_to(surrogate.sig2_obs, (dim_batch,))
    eps_cov = sig2[:, None, None] * I_can

    # combined distribution has dimension dim_rff_basis + dim_canonical_basis
    combined_dim = dim_rff_basis + dim_canonical_basis
    combined_cov = jnp.zeros((dim_batch, combined_dim, combined_dim))
    combined_cov = combined_cov.at[:, :dim_rff_basis, :dim_rff_basis].set(rff_weight_cov)
    combined_cov = combined_cov.at[:, dim_rff_basis:, dim_rff_basis:].set(eps_cov)

    dist = MultivariateNormal(covariance_matrix=combined_cov)
    return GaussianFromNumpyro(dist)