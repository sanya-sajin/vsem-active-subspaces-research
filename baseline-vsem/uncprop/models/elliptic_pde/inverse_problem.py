# uncprop/models/elliptic_pde/inverse_problem.py
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from numpy.random import default_rng
from scipy.stats import qmc

import gpjax
from gpjax.kernels.stationary import PoweredExponential
from numpyro.distributions import MultivariateNormal, LogNormal

from uncprop.custom_types import Array, ArrayLike, PRNGKey
from uncprop.utils.other import _numpy_rng_seed_from_jax_key
from uncprop.core.inverse_problem import Prior, Posterior
from uncprop.core.surrogate import GPJaxSurrogate
from uncprop.models.elliptic_pde.pde_model import get_discrete_source, solve_pde_vmap
from uncprop.utils.distribution import _gaussian_log_density_tril, _sample_gaussian_tril


def generate_pde_inv_prob_rep(key: PRNGKey,
                              noise_cov: Array,
                              n_kl_modes: int,
                              obs_locations: Array,
                              settings: PDESettings | None = None):
    """ Top-level function for generating instance of the PDE inverse problem

    Generates a Posterior object representing the exact posterior corresponding
    to the (discretized) inverse problem.

    Args:
        n_kl_modes: number of Karhunen-Loeve terms to retain; determines the 
                    dimension of the parameter space.
        obs_locations: indices of the grid at which observations occur.

    Returns:
        tuple: posterior, KL eigendecomposition info, ground truth quantities
    """

    if settings is None:
        settings = PDESettings()

    # GP prior on log permeability field
    meanf = gpjax.mean_functions.Constant(1.0)
    kernel = PoweredExponential(lengthscale=0.5, variance=6.0**2, power=1.0, n_dims=1)
    gp_prior = gpjax.gps.Prior(mean_function=meanf, kernel=kernel)

    kl_info, eig_info = karhunen_loeve(gp=gp_prior,
                                       settings=settings,
                                       n_kl_modes=n_kl_modes)
    prior = PDEPrior(dim=n_kl_modes)

    # Parameter-to-observable map
    forward_model = ForwardModel(pde_settings=settings,
                                 obs_locations=obs_locations,
                                 kl_info=kl_info)
    
    # Simulate synthetic observations
    true_param, ground_truth, observation = simulate_observation(key, prior, forward_model, noise_cov)
    ground_truth = tuple(x.squeeze() for x in ground_truth)

    # Construct likelihood and posterior
    likelihood = PDELikelihood(observation=observation,
                               noise_cov=noise_cov,
                               forward_model=forward_model)
    posterior = Posterior(prior, likelihood)

    return posterior, gp_prior, eig_info, ground_truth
    

def simulate_observation(key: PRNGKey,
                         prior: PDEPrior, 
                         forward_model: ForwardModel,
                         noise_cov: Array):
    """ Simulate ground truth and observation from inverse problem model

    Samples ground truth parameters (KL coefficients) from the prior, then
    feeds them through the forward model to obtain the ground truth observable.
    Simulates additive Gaussian noise to create a synthetic observation.

    Returns:
        tuple, with the sampled parameter, observable, and observation.
    """
    
    key_param, key_obs = jr.split(key, 2)
    noise_cov_tril = jnp.linalg.cholesky(noise_cov, upper=False)

    true_kl_coefs = prior.sample(key_param).ravel()
    ground_truth = forward_model.forward_with_intermediates(true_kl_coefs)
    true_observable = ground_truth[3]
    observation = _sample_gaussian_tril(key_obs,
                                        m=true_observable,
                                        L=noise_cov_tril).ravel()
    
    return true_kl_coefs, ground_truth, observation


def karhunen_loeve(gp: gpjax.gps.Prior, 
                   settings: PDESettings,
                   n_kl_modes: int) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    """
    Approximates the GP prior as a sum of basis "functions" with N(0,1) weights.

    This is not really attempting a full approximation of the KL (e.g., as could
    be done with the Nystrom method). We simply project on a dense grid and 
    compute the eigendecomposition.
    """

    # Discretized prior
    prior_grid = gp(settings.xgrid.reshape(-1, 1))

    # Eigendecomposition - scale to account for grid spacing
    K = settings.dx * prior_grid.covariance_matrix
    K = 0.5 * (K + K.T)
    eigvals, eigvecs = jnp.linalg.eigh(K)

    # Sort descending
    idx = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Approximation using truncated basis Phi (n_grid, D)
    Phi = eigvecs[:,:n_kl_modes] * jnp.sqrt(eigvals[:n_kl_modes])
    m = prior_grid.mean

    return (m, Phi), (eigvals, eigvecs)


@dataclass
class PDESettings:
    n_grid: int = 100
    left_flux: float = -1.0
    rightbc: float = 1.0
    source_wells: Array = field(default_factory=lambda: jnp.array([0.2, 0.4, 0.6, 0.8]))
    source_strength: float = 0.8
    source_width: float = 0.05

    def __post_init__(self):
        self.xgrid = jnp.linspace(0.0, 1.0, self.n_grid)
        self.low = 0
        self.high = 1
        self.dx = (self.high - self.low) / (self.n_grid - 1)
        self.source = get_discrete_source(X=self.xgrid, 
                                          well_locations=self.source_wells, 
                                          strength=self.source_strength, 
                                          width=self.source_width)


class PDEPrior(Prior):
    """
    This represents the prior over the Karhunen-Loeve modes, which 
    is simply N(0, I).
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def support(self):
        return (-jnp.inf, jnp.inf)
    
    @property
    def truncated_support(self):
        """Useful for some surrogate-based approximations that induce 
        posterior approximations that level off in the tails.
        """
        bound = jnp.abs(jax.scipy.stats.norm.ppf(0.001))
        lower = jnp.tile(-bound, self.dim)
        upper = jnp.tile(bound, self.dim)

        return lower, upper        

    @property
    def par_names(self):
        return [f'u{i}' for i in range(self.dim)]
    
    def log_density(self, x: ArrayLike):
        x = jnp.atleast_2d(x)
        return -0.5 * jnp.sum(x**2, axis=1)
    
    def sample(self, key: PRNGKey, n: int = 1):
        return jr.normal(key, shape=(n, self.dim))
    
    def sample_lhc(self, key: PRNGKey, n: int = 1):
        """Latin hypercube sample"""
        rng_key = _numpy_rng_seed_from_jax_key(key)
        rng = default_rng(seed=rng_key)
        lhc = qmc.LatinHypercube(d=self.dim, rng=rng)

        # Sample unit hypercube [0, 1)^d
        samp = lhc.random(n=n)

        # Scale based on Gaussian prior
        samp = jax.scipy.stats.norm.ppf(samp)
        return jnp.asarray(samp)


class ForwardModel:

    def __init__(self,
                 pde_settings: PDESettings,
                 obs_locations: Array,
                 kl_info: tuple[Array, Array]):
        
        m, Phi = kl_info

        self.pde_settings = pde_settings
        self.obs_locations = obs_locations.ravel()
        self.dim_par = Phi.shape[1]
        self.dim_obs = self.obs_locations.shape[0]
        self.m = m.ravel()
        self.Phi = Phi
        self.obs_grid = pde_settings.xgrid[obs_locations]

    def __call__(self, param: ArrayLike):
        """ Parameter-to-observable map as a function of the KL coefficients

        param is (dim_par,) vector of KL coefficients, or (n, dim_par) matrix of 
        multiple such vectors.

        Returns:
            (n,) array of the log-likelihood evaluations at each parameter value.
        """
        log_field = self.param_to_log_field(param)
        pde_solution = self.log_field_to_pde_solution(log_field)
        observable = self.pde_solution_to_observable(pde_solution)
        return observable
    
    def forward_with_intermediates(self, param: ArrayLike):
        """
        Same as __call__() but returns intermediate quantities.
        """
        log_field = self.param_to_log_field(param)
        pde_solution = self.log_field_to_pde_solution(log_field)
        observable = self.pde_solution_to_observable(pde_solution)
        return param, log_field, pde_solution, observable

    def param_to_log_field(self, param: ArrayLike):
        """
        param is (dim_par,) vector of KL coefficients, or 
        (n, dim_par) matrix of multiple such vectors.

        Returns:
            (n, n_grid) matrix of discretized log permeability fields.
        """
        param = jnp.atleast_2d(param)
        return self.m + param @ self.Phi.T
    
    def log_field_to_pde_solution(self, log_field: ArrayLike):
        """
        log_field is a (n_grid,) vector representing the discretized log 
        permeability field, or a (n, n_grid) matrix of multiple such vectors.

        Returns:
            (n, n_grid) matrix of discretized PDE solution over the grid.
        """
        log_field = jnp.atleast_2d(log_field)
        s = self.pde_settings
        return solve_pde_vmap(s.xgrid, s.left_flux, jnp.exp(log_field), s.source, s.rightbc)
    
    def pde_solution_to_observable(self, pde_solution: ArrayLike):
        """
        pde_solution is a (n_grid,) vector representing the discretized PDE
        solution over the grid, or a (n, n_grid) matrix of multiple such vectors.

        Returns:
            (n, dim_obs) matrix of observable quantities resulting from application
            of the observation operator.
        """
        pde_solution = jnp.atleast_2d(pde_solution)
        idx = self.obs_locations
        return jnp.atleast_2d(pde_solution[:,idx])


class PDELikelihood:
    """ Likelihood map wrapping around the forward model

    Implemented to align with the LogDensity protocol. Assumes additive
    Gaussian noise. The likelihood is parameterized as a function of the
    KL coefficients.
    """

    def __init__(self,
                 observation: Array,
                 noise_cov: Array,
                 forward_model: ForwardModel):
        
        self.forward_model = forward_model
        self.observation = observation.ravel()
        self.noise_cov_tril = jnp.linalg.cholesky(noise_cov, upper=False)

    def __call__(self, x: ArrayLike):
        """ Log-likelihood as a function of the KL coefficients

        x is (dim_par,) vector of KL coefficients, or 
        (n, dim_par) matrix of multiple such vectors. The naming 'x'
        is chosen to align with the LogDensity protocol, not to be 
        confused with the spatial grid for the PDE.

        Returns:
            (n,) array of the log-likelihood evaluations at each parameter value.
        """
        observable = self.forward(x)
        log_likelihood = self.observable_to_logdensity(observable)
        return log_likelihood

    def forward(self, param: ArrayLike):
        return self.forward_model(param)

    def observable_to_logdensity(self, observable: ArrayLike):
        """
        observable is a (dim_obs,) vector representing the predicted 
        observation (observable), or a (n, dim_obs) matrix of multiple such vectors.

        Returns:
            (n,) array of the log-likelihood evaluations at each observable.
        """
        predicted_observable = jnp.atleast_2d(observable)
        return _gaussian_log_density_tril(x=self.observation,
                                          m=predicted_observable,
                                          L=self.noise_cov_tril)
    

def plot_inverse_problem_setup(key: PRNGKey, 
                               posterior: Posterior,
                               ground_truth: tuple,
                               observation: Array,
                               alpha: float = 0.3, 
                               n_samp: int = 0):
    key_trajectories, key_monte_carlo = jr.split(key)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    axs = axs.ravel()
    forward_model = posterior.likelihood.forward_model
    xgrid = forward_model.pde_settings.xgrid
    true_param, true_log_field, true_pde_solution, true_observable = ground_truth

    # intervals
    Phi = forward_model.Phi
    mean_log = forward_model.m
    cov_log = Phi @ Phi.T
    scale_log = jnp.sqrt(jnp.diag(cov_log))
    field_prior = LogNormal(mean_log, scale_log)
    log_field_prior = MultivariateNormal(mean_log, cov_log + 1e-6 * jnp.identity(cov_log.shape[0]))

    sd = jnp.sqrt(field_prior.variance)
    lower = field_prior.mean - 2 * sd
    upper = field_prior.mean + 2 * sd

    # Plot prior over permeability field / true field
    ax = axs[0]
    ax.fill_between(xgrid, lower, upper, alpha=alpha, color='lightblue', label='+/- 2 sd')
    ax.plot(xgrid, field_prior.mean, color='gray', label='mean')
    ax.plot(xgrid, jnp.exp(true_log_field), color='black', label='true')

    if n_samp > 0:
        samp = jnp.exp(log_field_prior.sample(key_trajectories, (n_samp,)))
        ax.plot(xgrid, samp.T, alpha=0.7, color='grey')
    ax.legend()

    # Plot ground truth and prior predictive
    n_mc = 1000
    log_field_samp = log_field_prior.sample(key_monte_carlo, (n_mc,))
    solution_samp = forward_model.log_field_to_pde_solution(log_field_samp)
    prior_pred_mean = jnp.mean(solution_samp, axis=0)
    prior_pred_sd = jnp.std(solution_samp, axis=0)
    prior_pred_lower = prior_pred_mean - 2 * prior_pred_sd
    prior_pred_upper = prior_pred_mean + 2 * prior_pred_sd

    ax = axs[1]
    ax.fill_between(xgrid, prior_pred_lower, prior_pred_upper, alpha=alpha, color='lightblue', label='+/- 2 sd')
    ax.plot(xgrid, prior_pred_mean, color='gray', label='mean')
    ax.plot(xgrid, true_pde_solution, color='black', label='true solution')
    ax.plot(forward_model.obs_grid, observation, 'ro', label='observation')
    if n_samp > 0:
        prior_predictive_samp = forward_model.log_field_to_pde_solution(jnp.log(samp))
        ax.plot(xgrid, prior_predictive_samp.T, alpha=0.7, color='grey')
    ax.legend()

    fig.tight_layout()
    return fig, ax