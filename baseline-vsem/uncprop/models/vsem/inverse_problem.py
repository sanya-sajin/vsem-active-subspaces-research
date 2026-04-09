# uncprob/models/vsem/inverse_problem.py
"""
Defines the Bayesian inverse problem for the VSEM experiment.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Callable
from typing import Protocol

import jax
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from numpyro.distributions import Uniform, MultivariateNormal
from jax.scipy.linalg import solve_triangular
from scipy.stats import qmc

from uncprop.custom_types import Array, PRNGKey
from uncprop.utils.other import _numpy_rng_seed_from_jax_key
from uncprop.utils.distribution import (
    _sample_gaussian_tril,
    _gaussian_log_density_tril,
    _gaussian_log_det_term_tril,
)
from uncprop.models.vsem import vsemjax as vsem
from uncprop.core.inverse_problem import (
    Prior,
    LogLikelihood,
    Posterior,
)

VSEM_DEFAULT_PARAMS = {
    'kext': 0.85,
    'lar': 1.86523322e+00,
    'lue': 6.84170991e-04,
    'gamma': 5.04967614e-01,
    'tauv': 2.95868049e+03,
    'taus': 2.58846896e+04,
    'taur': 1.77011520e+03,
    'av': 0.85,
    'veg_init': 3.04573410e+00,
    'soil_init': 2.11415896e+01,
    'root_init': 5.58376223e+00
}
VSEM_DEFAULT_PARAMS = jax.tree_util.tree_map(jnp.asarray, VSEM_DEFAULT_PARAMS)


VSEM_DEFAULT_PRIORS = {
    'kext': Uniform(0.4, 1.0),
    'lar' : Uniform(0.2, 3.0),
    'lue' : Uniform(5e-04, 4e-03),
    'gamma': Uniform(2e-01, 6e-01),
    'tauv': Uniform(5e+02, 3e+03),
    'taus': Uniform(4e+03, 5e+04),
    'taur': Uniform(5e+02, 3e+03), 
    'av': Uniform(0.4, 1.0),
    'veg_init': Uniform(0.0, 10.0), 
    'soil_init': Uniform(0.0, 30.0),
    'root_init': Uniform(0.0, 10.0)
}


def generate_vsem_inv_prob_rep(key: PRNGKey,
                               par_names: list[str],
                               n_windows: int,
                               n_days_per_window: int,
                               observed_variable: str,
                               noise_cov_tril: Array):
    """ Top-level function for generating instance of a VSEM inverse problem

    Generates a Posterior object representing the exact posterior corresponding
    to a VSEM inverse problem.
    """

    key, key_prior, key_driver, key_obs = jr.split(key, 4)
    n_days = n_windows * n_days_per_window

    # prior on calibration parameters
    prior = VSEMPrior(par_names)

    # ground truth data generating process
    defaults = VSEM_DEFAULT_PARAMS.copy()
    true_calibration_params = prior.sample_all_vsem_params(key_prior)
    for param, value in true_calibration_params.items():
        defaults[param] = value

    time_steps, driver = vsem.simulate_vsem_driver(key_driver, n_days)
    obs_op_info = define_vsem_observation_operator(num_days=n_days, 
                                                   window_len=n_days_per_window,
                                                   vsem_output_var=observed_variable)
    true_dgp = DataGeneratingProcess(
        driver_key=key_driver,
        driver=driver,
        vsem_params=defaults,
        observation_operator_info=obs_op_info,
        noise_cov_tril=noise_cov_tril
    )

    # calibration model
    calibration_model = CalibrationModel(
        driver=true_dgp.driver,
        vsem_params=true_dgp.vsem_params,
        calibration_params=par_names,
        observation_operator_info=true_dgp.observation_operator_info,
        noise_cov_tril=true_dgp.noise_cov_tril
    )

    # simulate observed data
    observation_info = DataRealization(
        obs_key=key_obs,
        data_generating_process=true_dgp
    )

    # construct likelihood and posterior objects
    likelihood = VSEMLikelihood(calibration_model, observation_info)
    posterior = Posterior(prior, likelihood)

    return posterior


@dataclass
class ObservationOperatorInfo:
    """Stores the observation operator and intermediate quantities
    
    The intermediate quantities define the structure of the observation operator,
    and are primarily helpful for plotting. The observation operator is assumed
    to be of the form "compute window averages of one of the VSEM outputs"
    (e.g., monthly averages of LAI).
    """
    observation_operator: Callable
    window_mask: Array
    window_lens: Array
    window_midpoints: Array
    output_name: str
    output_idx: int


@dataclass
class DataGeneratingProcess:
    """Ground truth data generating process.
    
    Assumes an additive Gaussian noise model. `noise_cov_tril` is the lower 
    Cholesky factor of the Gaussian noise.
    """
    driver_key: PRNGKey
    driver: Array
    vsem_params: dict[str, float]
    observation_operator_info: ObservationOperatorInfo
    noise_cov_tril: Array


@dataclass
class CalibrationModel:
    """The model used for the data generating process when calibrating parameters
    
    This includes specification of calibration parameters (VSEM parameters that will
    be learned from data) and "fixed parameters". `vsem_params` defines the defaults
    for all VSEM parameters - the parameters specified in `calibration_params` will
    override the defaults, and the remaining parameters will be fixed at the specified
    default values. The `forward_model`, which is a map from the calibration parameters to
    VSEM outputs, is build by `build_batch_forward_model()`. 
    The observation operator is a map from VSEM outputs to an observable quantity.
    """
    driver: Array
    vsem_params: dict[str, float]
    calibration_params: list[str]
    observation_operator_info: ObservationOperatorInfo
    noise_cov_tril: Array

    def __post_init__(self):
        forward_model = vsem.build_batch_forward_model(driver=self.driver, 
                                                       par_names=self.calibration_params, 
                                                       default_param_dict=self.vsem_params)
        self.forward_model = forward_model
        
        def param_to_observable_map(x):
            vsem_output = self.forward_model(x)
            observable = self.observation_operator_info.observation_operator(vsem_output)
            return observable
        
        self.param_to_observable_map = param_to_observable_map


@dataclass
class DataRealization:
    """Stores observed data and intermediate quantities
    
    Intermediate quantities are the VSEM forward model output and the observable
    (output of the observation operator before adding noise).
    """
    obs_key: PRNGKey
    data_generating_process: DataGeneratingProcess

    def __post_init__(self):
        # parameter to vsem output
        param = self.data_generating_process.vsem_params
        driver = self.data_generating_process.driver
        vsem_input = vsem.make_vsem_input_from_named(param, driver, param)
        vsem_output = vsem.solve_vsem(vsem_input)[jnp.newaxis] # obs_op expects vectorized/batch output

        # output to observable
        obs_op = self.data_generating_process.observation_operator_info.observation_operator
        observable = obs_op(vsem_output).ravel()

        # observable to observation
        L = self.data_generating_process.noise_cov_tril
        noise_realization = _sample_gaussian_tril(self.obs_key, L=L).ravel()
        observation = observable + noise_realization

        self.vsem_output = vsem_output
        self.observable = observable
        self.noise_realization = noise_realization
        self.observation = observation
 

def obs_op_window_means(vsem_output: Array,
                        output_idx: int,
                        window_mask: Array,
                        window_lens: Array):
    """ Observation operator: window averages of LAI """

    lai_output = vsem_output[:,:,output_idx].T # (n_days, n_ensemble)
    lai_window_sums = window_mask @ lai_output # (n_windows, n_ensemble)
    lai_window_means = lai_window_sums.T / window_lens # (n_ensemble, n_windows) 

    return lai_window_means


def define_vsem_observation_operator(num_days: int, 
                                     window_len: int = 30,
                                     vsem_output_var: str = 'lai'):
    """Constructs a ObservationOperatorInfo using the obs_op_window_means observation operator"""

    # create windows for averaging
    window_start_idx = jnp.arange(0, num_days, window_len)
    window_stop_idx = window_start_idx + window_len
    n_windows = len(window_start_idx)

    if int(window_stop_idx[-1]) != int(num_days):
        raise ValueError(f'last entry of window_stop_idx should equal n_days = {num_days}.' 
                         f' Got {window_stop_idx[-1]}.')

    # mask of shape (n_windows, n_days); ones mark days in window
    window_mask = jnp.vstack(
        [jnp.concatenate([jnp.zeros(start), jnp.ones(end-start), jnp.zeros(num_days-end)])
         for start, end in zip(window_start_idx, window_stop_idx)]
    )
    window_lens = jnp.sum(window_mask, axis=1)
    window_midpoints = 0.5 * (window_start_idx + window_stop_idx)

    # Observation operator defined as windowly averages of an VSEM output variable.
    vsem_output_idx = vsem.get_vsem_output_names().index(vsem_output_var)

    # JAX compatible closure
    def obs_op(vsem_output):
        return obs_op_window_means(vsem_output=vsem_output,
                                   output_idx=vsem_output_idx,
                                   window_mask=window_mask,
                                   window_lens=window_lens)

    return ObservationOperatorInfo(
        observation_operator=obs_op,
        window_mask=window_mask,
        window_lens=window_lens,
        window_midpoints=window_midpoints,
        output_name=vsem_output_var,
        output_idx=vsem_output_idx
    )


class VSEMPrior(Prior):
    """
    The VSEM prior primarily represents the prior distribution over the VSEM 
    parameters that will be calibrated in the experiment. However, it also stores
    `_dists`, which defines the prior over all VSEM parameters. Additional methods
    beyond the standard Prior interface provide access to this extended prior.
    """

    # dictionary of marginals for all VSEM parameters
    _dists = VSEM_DEFAULT_PRIORS

    def __init__(self, par_names: list[str] | None = None):
        """ par_names defines the set of calibration parameters. """
        self._par_names = par_names or vsem.get_vsem_par_names()
        low = jnp.array([dist.low for dist in self.dists.values()])
        high = jnp.array([dist.high for dist in self.dists.values()])
        self._prior_rv = Uniform(low, high)

    @property
    def dists(self):
        return {par: self._dists[par] for par in self._par_names}

    @property
    def dim(self):
        return len(self._par_names)
    
    @property
    def par_names(self):
        return self._par_names
    
    @property
    def support(self):
        return self._prior_rv.low, self._prior_rv.high
    
    def sample(self, key: PRNGKey, n: int = 1):
        return self._prior_rv.sample(key, sample_shape=(n,))

    def log_density(self, x):
        x = jnp.atleast_2d(x)        
        l = self._prior_rv.log_prob(x)
        l = jnp.sum(l, axis=-1)

        # numpyro Uniform doesnt throw error or return -Inf if value is outside support
        low, high = self.support
        l = jnp.where(jnp.all((x >= low) & (x <= high), axis=1), l, -jnp.inf)

        return l

    def sample_all_vsem_params(self, key: PRNGKey):
        """
        Return dictionary representing a single sample from all of the parameters, 
        not just the calibration parameters.
        """
        samp = {}
        for par_name, prior in self._dists.items():
            samp[par_name] = prior.sample(key)

        return samp
    
    def sample_lhc(self, key: PRNGKey, n: int = 1):
        """ Latin hypercube sampling """
        rng_key = _numpy_rng_seed_from_jax_key(key)
        rng = np.random.default_rng(seed=rng_key)
        lhc = qmc.LatinHypercube(d=self.dim, rng=rng)

        # Sample unit hypercube [0, 1)^d
        samp = lhc.random(n=n)

        # Scale based on prior bounds
        l, u = self.support
        samp = qmc.scale(samp, l, u)

        return jnp.asarray(samp)


class VSEMLikelihood:
    """
    Defined to conform to the LogDensity protocol, a callable that computes the 
    log-likelihood.
    """

    def __init__(self,
                 model: CalibrationModel,
                 data: DataRealization):
        self.model = model 
        self.data = data

    def __call__(self, x):
        return self.log_density(x)
    
    def log_density(self, x):
        x = jnp.atleast_2d(x)
        observation = self.data.observation
        predicted_observable = self.model.param_to_observable_map(x)
        noise_cov_tril = self.model.noise_cov_tril

        return _gaussian_log_density_tril(x=observation, 
                                          m=predicted_observable, 
                                          L=noise_cov_tril)
    
    def log_density_upper_bound(self, x):
        """
        Pointwise upper bound on the log-density.
        In shape: (n,d), Out shape: (n,)
        """
        noise_cov_tril = self.model.noise_cov_tril
        return _gaussian_log_det_term_tril(noise_cov_tril)


# -----------------------------------------------------------------------------
# Utility functions 
# -----------------------------------------------------------------------------

def visualize_vsem_dgp(true_dgp: DataGeneratingProcess, 
                       model: CalibrationModel, 
                       data: DataRealization):
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs = axs.ravel()
    observation = data.observation
    window_midpoints = model.observation_operator_info.window_midpoints

    # true data generating process
    ax = axs[0]
    output_idx = true_dgp.observation_operator_info.output_idx
    output_name = true_dgp.observation_operator_info.output_name
    time_steps = jnp.arange(true_dgp.driver.shape[0])
    ax.plot(time_steps, data.vsem_output[..., output_idx].ravel(), label=output_name)
    ax.plot(window_midpoints, data.observable, color='orange')
    ax.plot(window_midpoints, data.observation, 'ro')
    ax.set_xlabel('days')
    ax.set_ylabel('observable')

    return fig, axs
    