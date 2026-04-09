# uncprop/core/samplers.py
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import jax
import jax.random as jr
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from numpyro.distributions import MultivariateNormal
import blackjax
from blackjax.mcmc.random_walk import RWInfo
from blackjax.base import (
    UpdateFn,
    Position,
    State,
    Info,
)

from uncprop.custom_types import Array, PRNGKey, ArrayLike
from uncprop.core.distribution import Distribution
from uncprop.core.surrogate import GPJaxSurrogate, SurrogateDistribution
from uncprop.utils.distribution import _sample_gaussian_tril, _sample_batch_gaussian_tril


def mcmc_loop(key: PRNGKey, 
              kernel: UpdateFn, 
              initial_state: State, 
              num_samples: int = 4000):
    """Main MCMC loop for all samplers"""

    @jax.jit
    def one_step(state, key):
        state, _ = kernel(key, state)
        return state, state

    keys = jax.random.split(key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)

    return states


def mcmc_loop_multiple_chains(key: PRNGKey,
                              kernel: UpdateFn,
                              initial_state: State,
                              num_samples: int = 4000,
                              num_chains: int = 4):
    """Vectorized MCMC loop over multiple chains"""

    @jax.jit
    def one_step(states, key):
        keys = jr.split(key, num_chains)
        states, _ = jax.vmap(kernel)(keys, states)
        return states, states
    
    keys = jr.split(key, num_samples)
    _, states = jax.lax.scan(one_step, initial_state, keys)
    
    return states


def sample_distribution(key: PRNGKey,
                        dist: Distribution, 
                        n_samples: int,
                        initial_position: Array,
                        n_chains: int = 1, 
                        n_burnin: int = 0,
                        thin_window: int = 1,
                        prop_cov: Array | None = None,
                        adapt: bool = True,
                        adapt_kwargs: Mapping[str, Any] | None = None):
    """ MCMC sampling for Distribution objects

    Adaptive Metropolis-Hastings with various reasonable defaults for
    sampling from distributions with densities. Specifying `n_warmup > 0`
    will run an initial warmup run with covariance adaptation for `n_warmup`
    iterations. The tuned proposal covariance will then be fixed during the
    main run. `n_burnin` specifies how many samples to drop at the beginning
    of the main chain. The defaults are set for the workflow where a longer 
    warmup run is conducted and no burnin is dropped from the main chain.
    `n_samples` is the number of samples that will be returned by
    this function. The actual number of samples in the main chain will be 
    `n_burnin + n_samples * thin_window`.

    Returns:
        tuple, containing:
            - positions: (n_samples, dim) array of samples
            - states: full set of states from the main chain
            - warmup_samp: full set of states from warmup chain / None if no warmup
            - prop_cov: full proposal covariance used in main chain
    """
    
    key_warmup_kernel, key_warmup_samp, key_kernel, key_samp = jr.split(key, 4)
    
    if initial_position.shape[0] != n_chains:
        raise ValueError(f'initial_position should have leadning batch index of length n_chains = {n_chains}')

    # target density
    logdensity = lambda x: dist.log_density(x).squeeze()

    # Initialize proposal covariance (will be adapted)
    if prop_cov is None:
        prop_cov = _init_dist_proposal_cov(dist)
    
    # run mcmc
    adapt_kwargs = adapt_kwargs or {}
    adapt_settings = AdaptationSettings(**adapt_kwargs)

    initial_state, kernel = init_adaptive_rwmh_kernel(key=key_warmup_kernel,
                                                      logdensity_fn=logdensity,
                                                      initial_position=initial_position,
                                                      adapt_settings=adapt_settings,
                                                      initial_cov=prop_cov, 
                                                      initial_log_scale=0.0,
                                                      n_chains=n_chains)
    
    n_samples_total = n_burnin + thin_window * n_samples
    states = jax.block_until_ready(
        mcmc_loop_multiple_chains(key=key_warmup_samp,
                                  kernel=kernel,
                                  initial_state=initial_state,
                                  num_samples=n_samples_total,
                                  num_chains=n_chains)
    )
        
    # drop burnin and thin
    positions = states.position[n_burnin:]
    positions = positions[::thin_window]

    return {'positions': positions,
            'states': states,
            'prop_cov': prop_cov,
            'kernel': kernel,
            'initial_state': initial_state}


def sample_distribution_old(key: PRNGKey,
                        dist: Distribution, 
                        n_samples: int,
                        initial_position: Array,
                        n_chains: int = 1, 
                        n_warmup: int = 50_000,
                        n_burnin: int = 0,
                        thin_window: int = 1,
                        prop_cov: Array | None = None,
                        adapt: bool = True,
                        adapt_kwargs: Mapping[str, Any] | None = None):
    """ MCMC sampling for Distribution objects

    Adaptive Metropolis-Hastings with various reasonable defaults for
    sampling from distributions with densities. Specifying `n_warmup > 0`
    will run an initial warmup run with covariance adaptation for `n_warmup`
    iterations. The tuned proposal covariance will then be fixed during the
    main run. `n_burnin` specifies how many samples to drop at the beginning
    of the main chain. The defaults are set for the workflow where a longer 
    warmup run is conducted and no burnin is dropped from the main chain.
    `n_samples` is the number of samples that will be returned by
    this function. The actual number of samples in the main chain will be 
    `n_burnin + n_samples * thin_window`.

    Returns:
        tuple, containing:
            - positions: (n_samples, dim) array of samples
            - states: full set of states from the main chain
            - warmup_samp: full set of states from warmup chain / None if no warmup
            - prop_cov: full proposal covariance used in main chain
    """
    
    key_warmup_kernel, key_warmup_samp, key_kernel, key_samp = jr.split(key, 4)
    
    if initial_position.shape[0] != n_chains:
        raise ValueError(f'initial_position should have leadning batch index of length n_chains = {n_chains}')

    # target density
    logdensity = lambda x: dist.log_density(x).squeeze()

    # Initialize proposal covariance (will be adapted)
    if prop_cov is None:
        prop_cov = _init_dist_proposal_cov(dist)
    
    # Warm-up adaptation
    if n_warmup > 0:
        adapt_kwargs = adapt_kwargs or {}
        adapt_settings = AdaptationSettings(**adapt_kwargs)

        initial_state, warmup_kernel = init_adaptive_rwmh_kernel(key=key_warmup_kernel,
                                                                 logdensity_fn=logdensity,
                                                                 initial_position=initial_position,
                                                                 adapt_settings=adapt_settings,
                                                                 initial_cov=prop_cov, 
                                                                 initial_log_scale=0.0,
                                                                 n_chains=n_chains)

        warmup_samp = mcmc_loop_multiple_chains(key=key_warmup_samp,
                                                kernel=warmup_kernel,
                                                initial_state=initial_state,
                                                num_samples=n_warmup,
                                                num_chains=n_chains)
        
        # extract initial position and proposal covariance from warmup
        initial_position, prop_cov = _extract_tuned_warmup_quantities(warmup_samp)
    else:
        warmup_samp = None


    # main chain
    init_fn, kernel = init_custom_rwmh_kernel(key=key_kernel,
                                              logdensity_fn=logdensity)
    prop_tril = jnp.linalg.cholesky(prop_cov, upper=False)
    initial_state = jax.vmap(init_fn, in_axes=(0, 0))(initial_position, prop_tril)

    n_samples_total = n_burnin + thin_window * n_samples

    states = jax.block_until_ready(
        mcmc_loop_multiple_chains(key=key_samp, 
                                  kernel=kernel,
                                  initial_state=initial_state,
                                  num_samples=n_samples_total,
                                  num_chains=n_chains)
    )
    
    # drop burnin and thin
    positions = states.position[n_burnin:]
    positions = positions[::thin_window]

    return {'positions': positions,
            'states': states,
            'warmup_samp': warmup_samp,
            'prop_cov': prop_cov,
            'kernel': kernel,
            'initial_state': initial_state}


def _init_dist_proposal_cov(dist: Distribution):
    """A reasonable initial covariance"""

    low, high = dist.support
    infinite_support = jnp.any(jnp.isinf(low) | jnp.isinf(high))
    if infinite_support:
        prop_cov = jnp.identity(dist.dim)
    else:
        prop_sd = 0.1 * (jnp.broadcast_to(high, dist.dim) - jnp.broadcast_to(low, dist.dim))
        prop_cov = jnp.diag(prop_sd ** 2)

    return prop_cov


def _extract_tuned_warmup_quantities(warmup_samp: AdaptiveRWState):
    """Valid for single and multi-chain. For latter assumes leading batch index."""

    # Final position from warmup
    initial_position = warmup_samp.position[-1]

    # Final proposal covariance from warmup
    final_log_scale = warmup_samp.adapt_state.log_scale[-1]
    final_cov = warmup_samp.adapt_state.cov_prop[-1]
    final_adapt_state = AdaptationState(cov_prop=final_cov, 
                                        log_scale=final_log_scale, 
                                        times_adapted=0)
    L = _proposal_tril_from_adaptation(final_adapt_state)

    perm = jnp.arange(L.ndim, dtype=jnp.int64)
    perm = perm.at[-2:].set(jnp.array([perm[-1], perm[-2]]))
    cov_prop = L @ jnp.transpose(L, perm)

    return initial_position, cov_prop


# -----------------------------------------------------------------------------
# Metropolis-Hastings (light wrapper around blackjax implementation)
# -----------------------------------------------------------------------------

def init_rwmh_kernel(key: PRNGKey,
                     logdensity: Callable,
                     initial_position: Position,
                     prop_cov: Array):
    """
    Initializes blackjax random walk Metropolis-Hastings kernel with additive
    Gaussian proposals. No adaptation is done.

    Notes:
        Had some odd issues with blackjax Gaussian proposal, likely due to pytree/array
        shaping issues. Defining a simpler Gaussian proposal here that uses the fact 
        all positions are arrays in our case (not arbitrary pytrees)
    """

    # Symmetric Gaussian proposal
    L = jnp.linalg.cholesky(prop_cov, upper=False)
    
    def proposal(key: PRNGKey, position: Position):
        return _sample_gaussian_tril(key, m=position, L=L).squeeze()

    rmh = blackjax.rmh(logdensity, proposal)
    initial_state = rmh.init(initial_position, key)
    kernel = rmh.step

    return initial_state, kernel


class CustomRWState(NamedTuple):
    position: ArrayLike
    logdensity: ArrayLike
    proposal_tril: ArrayLike

class CustomRWInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool


def init_custom_rwmh_kernel(key: PRNGKey, logdensity_fn: Callable):
    """
    Alternative to blackjax random walk Metropolis-Hastings, that also stores
    the proposal Cholesky factor in the state. This is useful for vectorizing
    over multiple chains, each of which should have its own proposal.
    """

    # build kernel function
    def kernel(key: PRNGKey, state: CustomRWState):

        key_proposal, key_accept = jr.split(key, 2)
        u = state.position
        L = state.proposal_tril

        proposal = _sample_gaussian_tril(key_proposal, 
                                         m=state.position, 
                                         L=state.proposal_tril).squeeze()

        pos_next, lp_next, accept_prob, accept = _mh_accept_reject(key_accept,
                                                                   lp_curr=state.logdensity, 
                                                                   lp_prop=logdensity_fn(proposal),
                                                                   u_curr=state.position, u_prop=proposal)

        next_state = CustomRWState(position=pos_next,
                                   logdensity=lp_next,
                                   proposal_tril=state.proposal_tril)
        info = CustomRWInfo(acceptance_rate=accept_prob, 
                            is_accepted=accept)

        return next_state, info


    # build init function
    def init_fn(position, proposal_tril):
        return CustomRWState(position=position,
                             logdensity=logdensity_fn(position),
                             proposal_tril=proposal_tril)

    return init_fn, kernel


# -----------------------------------------------------------------------------
# Adaptive Metropolis-Hastings 
# -----------------------------------------------------------------------------

def init_adaptive_rwmh_kernel(key: PRNGKey,
                              logdensity_fn: Callable,
                              initial_position: Position,
                              adapt_settings: AdaptationSettings,
                              initial_cov: Array,
                              initial_log_scale: float | None = None,
                              n_chains: int = 1):

    # build kernel function
    def kernel(key: PRNGKey, 
               state: AdaptiveRWState) -> tuple[AdaptiveRWState, AdaptiveRWInfo]:

        key_proposal, key_accept = jr.split(key, 2)
        u = state.position
        L = state.proposal_tril

        # Proposal / accept-reject step
        u_prop = _sample_gaussian_tril(key_proposal, m=u, L=L).squeeze()
        u_next, lp_next, accept_prob, accept = _mh_accept_reject(key_accept,
                                                                 lp_curr=state.logdensity, 
                                                                 lp_prop=logdensity_fn(u_prop),
                                                                 u_curr=u, u_prop=u_prop)

        # Update batch histories
        new_sample_history = jax.lax.dynamic_update_slice(state.sample_history, 
                                                          u_next.reshape(1, -1), 
                                                          (state.step_in_batch, 0))
        new_acc_history = jax.lax.dynamic_update_slice(state.accept_prob_history, 
                                                       accept_prob.reshape(-1), 
                                                       (state.step_in_batch,))
                                                                    
        # Adaptation trigger
        next_step_count = state.step_in_batch + 1
        is_update_step = (next_step_count == adapt_settings.adapt_interval)
        
        def _do_update(s):
            avg_acc = jnp.mean(s.accept_prob_history)
            new_adapt = update_adaptation(
                s.adapt_state, 
                s.sample_history, 
                avg_acc, 
                adapt_settings
            )

            new_L = _proposal_tril_from_adaptation(new_adapt)
            return new_adapt, new_L

        def _no_update(s):
            return s.adapt_state, L

        new_adapt_state, L = jax.lax.cond(
            is_update_step,
            _do_update,
            _no_update,
            state
        )

        # Reset counter if adaptation occurred
        next_step_count = jnp.where(is_update_step, 0, next_step_count)

        # Updated state and auxiliary info
        next_state = AdaptiveRWState(position=u_next,
                                     logdensity=lp_next,
                                     proposal_tril=L,
                                     adapt_state=new_adapt_state,
                                     sample_history=new_sample_history,
                                     accept_prob_history=new_acc_history,
                                     step_in_batch=next_step_count)
        info = AdaptiveRWInfo(acceptance_rate=accept_prob, 
                              is_accepted=accept)

        return next_state, info

    # Initialize state    
    dim = initial_cov.shape[-1]
    adapt_interval = adapt_settings.adapt_interval
    adapt_state = init_adaptation_state(dim=dim,
                                        initial_cov=initial_cov,
                                        initial_log_scale=initial_log_scale,
                                        n_chains=n_chains)
    logdensity_init = jnp.atleast_1d(logdensity_fn(initial_position))

    initial_state = AdaptiveRWState(position=initial_position,
                                    logdensity=logdensity_init,
                                    proposal_tril=_proposal_tril_from_adaptation(adapt_state),
                                    adapt_state=adapt_state,
                                    sample_history=jnp.zeros((n_chains, adapt_interval, dim)),
                                    accept_prob_history=jnp.zeros((n_chains, adapt_interval)),
                                    step_in_batch=jnp.zeros(n_chains, dtype=jnp.int64))

    return initial_state, kernel


class AdaptationState(NamedTuple):
    """
    Holds the state of the adaptation mechanism. The proposal covariance
    is parameterized as:
        exp(2 * log_scale) * cov_prop = s^2 * C
    
    cov_prop
        The current proposal covariance matrix (shape matrix).
    log_scale
        The log of the global scaling factor.
    times_adapted
        Counter for how many adaptation steps have occurred.
    """
    cov_prop: Array
    log_scale: float
    times_adapted: int


class AdaptiveRWState(NamedTuple):
    """State for adaptive random walk Metropolis-Hastings.

    position
        Current position of the chain.
    logdensity
        log density evaluated at the current position.
    proposal_tril
        Lower Cholesky factor of the full proposal covariance s^2 C
    adapt_state
        The current AdaptationState, storing log(s) and C
    sample_history
        Buffer for batch of samples used in adaptation
    accept_prob_history
        Buffer for batch of acceptance probabilites used in adaptation
    step_in_batch
        Counts from 0 to adapt_interval
    """
    position: Array
    logdensity: float
    proposal_tril: Array
    adapt_state: AdaptationState
    sample_history: Array
    accept_prob_history: Array
    step_in_batch: int


class AdaptiveRWInfo(NamedTuple):
    acceptance_rate: float
    is_accepted: bool


class AdaptationSettings(NamedTuple):
    """Hyperparameters for the adaptation logic.
    
    The scale is updated as:
        s_new := s * exp{scale_numerator / (N + 3) ** gamma_exponent * (accept_ratio - target_accept)}
        C_new := C + (C_hat - C) / (N + 3) ** gamma_exponent

    where N = times_adapted is the number of times adaptation has occurred up to this point.
    `accept_ratio` and `C_hat` are empirical estimates computed using the recent history of the chain.
    """
    adapt_interval: int = 50
    target_accept: float = 0.234
    adapt_cov: bool = True
    adapt_scale: bool = True
    gamma_exponent: float = 0.8
    scale_numerator: float = 10.0
    jitter: float = 1e-6


def init_adaptation_state(
    dim: int, 
    initial_cov: Array | None = None, 
    initial_log_scale: float | None = None,
    n_chains: int = 1
) -> AdaptationState:
    """Initializes the adaptation state."""
    
    if initial_cov is None:
        initial_cov = jnp.eye(dim)
    
    
    if initial_log_scale is None:
        # Gelman-Roberts-Gilks heuristic: 2.38^2 / d
        initial_log_scale = jnp.log(2.38) - 0.5 * jnp.log(dim)
    
    times_adapted = jnp.array(0, dtype=jnp.int64)

    return AdaptationState(
        cov_prop=jnp.broadcast_to(initial_cov, (n_chains, dim, dim)),
        log_scale=jnp.broadcast_to(initial_log_scale, (n_chains,)),
        times_adapted=jnp.broadcast_to(times_adapted, (n_chains,))
    )


def update_adaptation(
    adapt_state: AdaptationState, 
    batch_history: Array, 
    batch_accept_rate: float,
    settings: AdaptationSettings = AdaptationSettings()
) -> AdaptationState:
    """
    Performs one step of adaptation based on a batch of MCMC history.
    
    Args:
        adapt_state: Current AdaptationState.
        batch_history: Array of shape (batch_size, dim) containing recent samples.
        batch_accept_rate: Scalar, average acceptance probability of the batch.
        settings: Hyperparameters.
    
    Returns:
        Updated AdaptationState.
    """
    
    # Calculate decay factor: gamma = (N + 3)^(-gamma_exponent)
    gamma = 1.0 / ((adapt_state.times_adapted + 3.0) ** settings.gamma_exponent)
    
    # Update Scale (Robbins-Monro)
    # l_new = l_old + eta * gamma * (acc - target)
    diff = batch_accept_rate - settings.target_accept
    scale_adjustment = settings.scale_numerator * gamma * diff
    new_log_scale = adapt_state.log_scale + scale_adjustment

    # Update Covariance (Stochastic Approximation / exponential moving average)
    # C_new = (1-gamma)*C_old + gamma*C_batch
    batch_cov = jnp.cov(batch_history, rowvar=False)
    batch_cov = jnp.atleast_2d(batch_cov)
    new_cov = adapt_state.cov_prop + gamma * (batch_cov - adapt_state.cov_prop)
    
    # Enforce symmetry and regularize to ensure positive definiteness
    new_cov = 0.5 * (new_cov + new_cov.T)
    new_cov = new_cov + settings.jitter * jnp.eye(new_cov.shape[0])

    return AdaptationState(
        cov_prop=new_cov,
        log_scale=new_log_scale,
        times_adapted=adapt_state.times_adapted + 1
    )


def _proposal_tril_from_adaptation(adapt_state: AdaptationState) -> Array:
    """
    Reconstructs the Cholesky decomposition of the full proposal matrix.
    Sigma = exp(2 * log_scale) * cov_prop
    L = exp(log_scale) * cholesky(cov_prop)

    Works for both single and multiple chains.
    """
    scale = jnp.exp(adapt_state.log_scale)
    L_cov = jnp.linalg.cholesky(adapt_state.cov_prop, upper=False)
    return jnp.asarray(scale)[..., None, None] * L_cov


# -----------------------------------------------------------------------------
# NUTS 
# -----------------------------------------------------------------------------

def init_nuts_kernel(key: PRNGKey, 
                     logdensity: Callable,
                     initial_position: Position,
                     num_warmup_steps: int = 1000):
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    warmup_key, _ = jax.random.split(key, 2)
    (initial_state, nuts_params), _ = warmup.run(warmup_key, initial_position, num_steps=num_warmup_steps)
    kernel = blackjax.nuts(logdensity, **nuts_params).step

    return initial_state, kernel


# -----------------------------------------------------------------------------
# Preconditioned Crank-Nicolson Random Kernel Algorithm (rk-pcn)
# -----------------------------------------------------------------------------

class RKPCNState(NamedTuple):
    """State of the rkpcn sampler chain.

    position
        Current position of the chain.
    f_at_position
        Current value of f evaluated at current position. Shape (q,).
    proposal_tril
        Lower Cholesky factor of proposal covariance for u.
    rho
        Correlation parameter for the pCN proposal.
    logdensity
        Current sampled value of the log density.
    """
    position: Array
    f_position: Array
    logdensity: Array
    proposal_tril: Array
    f_update_info: Any


class RKPCNInfo(NamedTuple):
    """Side information for the cut sampler chain.

    accept_prob
        acceptance probability
    is_accepted
        whether proposal was accepted or not
    """
    accept_prob: ArrayLike
    is_accepted: ArrayLike


def init_rkpcn_kernel(key: PRNGKey,
                      log_density: Callable[[Array, Array], Array],
                      gp: GPJaxSurrogate,
                      f_update_fn: Callable[..., tuple[RKPCNState, Array]],
                      f_update_info: Any) -> tuple[Callable, Callable]:
    """
    An MCMC sampler to *approximately* sample from the expectation of the random
    measure pi(u; f) with random log density of the form log_p(f(u)), where f ~ GP(m, k).
    Precisely, one seeks to sample from E_f{pi(u; f)}.
    
    The algorithm rk-pcn sampler operates on the extended state space (u, f), which is in
    principle infinite dimensional. In reality, it is only necessary to instantiate 
    bivariate projections of the GP of the form [f(u), f(u')]. Therefore, the practical
    implementation of this algorithm maintains a state of the form (u, f(u)).

    Note that `log_density` and `gp` must be specified so that they can be called like
        key = jax.random.key(3532)
        pred = gp(u) # predictive distribution
        f_pred = pred.sample(key)
        lp_pred = log_density(f_pred, u)

    In other words, `log_density` is parameterized as a function of the surrogate output,
    not as a function of the parameter u. The argument `u` is still passed, as this is 
    useful in certain cases (e.g., constraining the support of u).
    
    The Gaussian predictive distribution must 
    satisfy the following:
        - single input u: gp(u).sample() has shape (p,)
        - multiple inputs u of shape (n,d): gp(u).sample() has shape (n,p)

    Notes:
        The notation used in the function body is as follows:
            - f/g: current/proposed GP trajectories
            - u/v: current/proposed parameter positions
            - unext/fnext: next state
        We use the notation fv to mean the value of f at input v, and so on.
    """

    # build kernel function
    def kernel(key: PRNGKey, state: RKPCNState) -> tuple[RKPCNState, RKPCNInfo]:

        key_u_proposal, key_f_update, key_u_accept = jr.split(key, 3)

        u = state.position
        fu = state.f_position

        # Propose new u position
        L = state.proposal_tril
        v = _sample_gaussian_tril(key_u_proposal, m=u, L=L).squeeze()

        # f update
        state, fnext_v = f_update_fn(key=key_f_update, 
                                     state=state,
                                     gp=gp,
                                     log_density=log_density,
                                     u_prop=v)
        fnext_u = state.f_position

        # u update
        u_next, lp_next, accept_prob, accept = _mh_accept_reject(key_u_accept,
                                                                 lp_curr=state.logdensity, 
                                                                 lp_prop=log_density(fnext_v, v),
                                                                 u_curr=u, u_prop=v)
        fnext_unext = jax.lax.cond(accept, lambda _: fnext_v, lambda _: fnext_u, operand=None)

        # Updated state and auxiliary info
        info = RKPCNInfo(accept_prob=accept_prob, is_accepted=accept)
        next_state = RKPCNState(position=u_next,
                                f_position=fnext_unext,
                                logdensity=lp_next,
                                proposal_tril=L,
                                f_update_info=state.f_update_info)

        return next_state, info

    # init function
    def init_fn(key, initial_position, prop_cov):
        prop_cov_tril = jnp.linalg.cholesky(prop_cov, upper=False)
        f_init = gp(initial_position).sample(key).reshape(gp.output_dim)
        lp_init = log_density(f_init, initial_position).squeeze()

        return RKPCNState(position=initial_position,
                          f_position=f_init,
                          logdensity=lp_init,
                          proposal_tril=prop_cov_tril, 
                          f_update_info=f_update_info)

    return init_fn, kernel


def _f_update_pcn_proposal(key: PRNGKey,
                           *,
                           state: RKPCNState,
                           gp: GPJaxSurrogate,
                           log_density: Callable[[Array, Array], Array], 
                           u_prop: Array,
                           **kwargs):
    """ pCN proposal without a Metropolis correction. 
    
    Realizes pCN proposal at the finite set of points (u, u_prop), where u
    is the current position.

    f_update_info requires:
        rho: float, the pCN correlation parameter
    """

    rho = state.f_update_info.rho
    u = state.position
    fu = state.f_position

    gU, _ = _just_in_time_pcn_proposal(key=key,
                                       gp=gp,
                                       given=(u, fu),
                                       u_jit=u_prop,
                                       rho=rho)

    # Update state
    gu = gU[:, 0]
    gv = gU[:, 1]
    state = state._replace(f_position=gu.reshape(gp.output_dim), logdensity=log_density(gu, u))

    return state, gv


def _f_update_eup_cpm(key: PRNGKey,
                      *,
                      state: RKPCNState,
                      gp: GPJaxSurrogate, 
                      log_density: Callable[[Array, Array], Array],
                      u_prop: Array,
                      **kwargs):
    """ Correlation pseudo-marginal (cPM) update for targeting expected unnormalized posterior (EUP) 
    
    pCN proposal with a Metropolis correction that uses ratio pi(u; g)/pi(u; f) of unnormalized 
    densities. Does not include normalizing constants Z(f)/Z(g) in this ratio, which makes this
    update target the EUP, not the EP.

    f_update_info requires:
        rho: float, the pCN correlation parameter
    """
    key_proposal, key_accept = jr.split(key, 2)

    rho = state.f_update_info.rho
    u = state.position
    fu = state.f_position
    lu = state.logdensity

    # Proposal
    gU, fv = _just_in_time_pcn_proposal(key=key_proposal,
                                        gp=gp,
                                        given=(u, fu),
                                        u_jit=u_prop,
                                        rho=rho)
    gu = gU[:, 0]
    gv = gU[:, 1]

    # Accept-Reject with EUP target
    fnext_u, lnext_u, accept_prob, accept = _mh_accept_reject(key_accept,
                                                              lp_curr=lu, 
                                                              lp_prop=log_density(gu, u),
                                                              u_curr=fu, u_prop=gu)
    fnext_v = jax.lax.cond(accept, lambda _: gv, lambda _: fv, operand=None)

    # Update state
    state = state._replace(f_position=fnext_u, logdensity=lnext_u)

    return state, fnext_v


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _mh_accept_reject(key: PRNGKey,
                      lp_curr: float,
                      lp_prop: float,
                      u_curr: ArrayLike,
                      u_prop: ArrayLike,
                      log_correction: ArrayLike = 0.0) -> tuple[Array, Array, Array, Array]:
    """ Assumes symmetric proposal """
    log_unif = jnp.log(jr.uniform(key))

    log_alpha = jnp.squeeze(lp_prop - lp_curr + log_correction)
    log_alpha = jnp.where(jnp.isnan(log_alpha), -jnp.inf, log_alpha)
    accept = log_unif < log_alpha

    u_next, lp_next = jax.lax.cond(
        accept,
        lambda _: (u_prop, lp_prop),
        lambda _: (u_curr, lp_curr),
        operand=None,
    )   

    # acceptance probability
    accept_prob = jnp.clip(jnp.exp(log_alpha), min=0.0, max=1.0)

    return u_next, lp_next, accept_prob, accept


def _pcn_proposal(key: PRNGKey,
                  x: Array,
                  mean: Array, 
                  cov_tril: Array, 
                  rho: float) -> Array:
    """
    x, mean: (q, d) or (d,)
    cov_tril: (q, d, d) or (d, d)
    rho: scalar

    Returns:
        (q, d), batch of q d-dimensional pCN proposals.
    """
    pcn_mean = mean + rho * (x - mean)
    pcn_tril = jnp.sqrt(1 - rho**2) * cov_tril

    return _sample_batch_gaussian_tril(key, m=pcn_mean, L=pcn_tril).squeeze(0)


def _just_in_time_pcn_proposal(key: PRNGKey,
                               *,
                               gp: GPJaxSurrogate,
                               given: tuple[Array, Array],
                               u_jit: Array,
                               rho: float):
    """ Just-in-time sample then pCN proposal.

    Given a realization `given = (u, fu)` of a GP at inputs u, and another set
    of inputs `u_jit`. First, just-in-time samples values at `u_jit` via
    f_jit ~ law(f(u_jit) | f(u) = u). Then returns a pCN proposal at the 
    combined set of inputs (u, u_jit), given current value (fu, f_jit).

    Returns:
        tuple:
            fU_prop: (q, 2), the pCN proposal at inputs (u, u_jit)
            f_jit: (q,), the just-in-time sample f_jit at input u_jit
    """

    key_jit, key_proposal = jr.split(key, 2)

    # just-in-time sample [squeeze leading sample dimension]
    f_jit = gp.condition_then_predict(u_jit, given=given).sample(key_jit).squeeze(0)

    # pCN proposal, realized at points (u, u_jit)
    u, fu = given
    U = jnp.vstack([u, u_jit])
    fU = jnp.hstack([jnp.reshape(fu, (gp.output_dim, 1)), jnp.reshape(f_jit, (gp.output_dim, 1))])
    fU_dist = gp(U)
    fU_prop = _pcn_proposal(key_proposal, fU, fU_dist.mean, fU_dist.chol, rho=rho)

    return fU_prop, f_jit.reshape(gp.output_dim)


def _logZ_approx(logw: ArrayLike, lp_U: ArrayLike, axis=None):
    return logsumexp(logw + lp_U, axis=axis)


def stack_dict_arrays(data_dict: dict[str, ArrayLike], 
                      names: list[str] | None = None) -> Array:
    """
    Given a dictionary of arrays, each of shape (n,) or (n,d) (with d allowed to vary
    by array), concatenate to array of shape (n, d_total). The arrays are concatenated
    left-to-right in the order specified by `names`.
    """
    if names is None:
        names = list(data_dict.keys())

    arrays = [jnp.asarray(data_dict[name]) for name in names]
    arrays = jax.tree.map(tree=arrays, f=_ensure_2d)

    n_rows = arrays[0].shape[0]
    assert all(a.shape[0] == n_rows for a in arrays), "All arrays must have same number of rows"
    return jnp.concatenate(arrays, axis=1) 


def get_trace_plots(trace: dict[str, Array], 
                    names: list[str] | None = None,
                    num_rows: int = 1,
                    figsize: tuple = (15, 6)):

    if names is None:
        names = list(trace.keys())

    num_plots = len(names)
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_plots-num_rows+1)
    if num_plots > 1:
        axs = axs.ravel()

    for i in range(num_plots):
        ax = axs[i]
        nm = names[i]
        trajectory = trace[nm]

        ax.plot(trajectory)
        ax.set_xlabel('samples')
        ax.set_ylabel(nm)
    
    return fig, axs


def _ensure_2d(x):
    x = jnp.asarray(x)
    assert x.ndim <= 2

    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x



if __name__ == '__main__':
    import jax.scipy.stats as stats
    from datetime import date

    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    loc, scale = 10, 20
    observed = np.random.normal(loc, scale, size=1_000)

    def logdensity_fn(loc, log_scale, observed=observed):
        """Univariate Normal"""
        scale = jnp.exp(log_scale)
        logjac = log_scale
        logpdf = stats.norm.logpdf(observed, loc, scale)
        return logjac + jnp.sum(logpdf)
    logdensity = lambda x: logdensity_fn(**x)

    initial_position = {"loc": 1.0, "log_scale": 1.0}

    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    rng_key, warmup_key, sample_key = jax.random.split(rng_key, 3)
    (state, parameters), _ = warmup.run(warmup_key, initial_position, num_steps=1000)

    kernel = blackjax.nuts(logdensity, **parameters).step
    
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, _ = kernel(rng_key, state)
            return state, state

        keys = jax.random.split(rng_key, num_samples)
        _, states = jax.lax.scan(one_step, initial_state, keys)

        return states

    kernel = blackjax.nuts(logdensity, **parameters).step
    states = inference_loop(sample_key, kernel, state, 1_000)

    mcmc_samples = states.position
    mcmc_samples["scale"] = jnp.exp(mcmc_samples["log_scale"]).block_until_ready()

    fig, (ax, ax1) = plt.subplots(ncols=2, figsize=(15, 6))
    ax.plot(mcmc_samples["loc"])
    ax.set_xlabel("Samples")
    ax.set_ylabel("loc")

    ax1.plot(mcmc_samples["scale"])
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("scale")
    fig.savefig('/Users/andrewroberts/Desktop/git-repos/bip-surrogates-paper/uncprop/test.png')


