# mcmc_uncertainty_prop.py
"""
MCMC algorithms for uncertainty propagation in surrogate-based inverse problems
"""
from __future__ import annotations
import numpy as np
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.linalg import cho_solve

from gpjax.linalg import Dense, psd
from gpjax.linalg.utils import add_jitter
from gpjax.gps import ConjugatePosterior as GPJaxConjugateGP
from gpjax.dataset import Dataset as GPJaxDataset
from gpjax.distributions import GaussianDistribution as GPJaxGaussian

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.samplers import MCMCSampler
from typing import Any, Callable, Protocol, runtime_checkable, TypeVar

from modmcmc.kernels import (
    MarkovKernel,
    MetropolisKernel,
    GaussianProposal,
    GaussMetropolisKernel, 
    DiscretePCNProposal,
    DiscretePCNKernel, 
    UncalibratedDiscretePCNKernel, 
    mvn_logpdf
)

# -----------------------------------------------------------------------------
# Protocols for working with Gaussian Processes
# -----------------------------------------------------------------------------

Array = np.typing.NDArray

@runtime_checkable
class DistWithMeanCov(Protocol):
    """
    Protocol for distribution-like objects that expose
    `.mean` and `.cov` attributes/properties which are Arrays.
    """
    mean: Array
    cov: Array

    def sample(self, n: int = 1) -> Array:
        ...

@runtime_checkable
class GaussianProcess(Protocol):
    """
    Protocol for a callable GP-like object:

        f(x: Array, given: Optional[Tuple[Array, Array]] = None) -> DistWithMeanCov

    Where `given` (if provided) is a tuple (x_new, y_new). Here (x_new, y_new) is a new dataset
    on which the GP is conditioned. The call returns a object representing the multivariate
    Gaussian obtained by projecting the (potentially conditional) GP onto the finite set
    of input points `x`.
    """
    def __call__(self, x: Array, given: tuple[Array, Array] | None = None) -> DistWithMeanCov:
        ...


# -----------------------------------------------------------------------------
# Wrapper for gpjaxGP Gaussian process to align with protocol
# -----------------------------------------------------------------------------

class gpjaxGaussian():
    """
    Wrapper around a gpjax Gaussian distribution, slightly modified to align with
    the `DistWithMeanCov` protocol. Since I am primarily using a numpy random 
    generator for pseudorandomness, we use this to seed JAX key generators.
    """

    def __init__(self, loc, scale, np_rng: np.random.Generator):
        """ 
        Note that `scale` should be a gpjax linear operator.
        """
        # `loc` must have at least one dimension (zero dimensional array causes issues)
        if loc.ndim == 0:
            loc = loc[jnp.newaxis]

        self.gpjax_gaussian = GPJaxGaussian(loc=loc, scale=scale)
        self.np_rng = np_rng
    
    @property
    def mean(self):
        return self.gpjax_gaussian.mean

    @property
    def cov(self):
        return self.gpjax_gaussian.covariance_matrix
    
    def sample(self, n: int = 1):
        jax_seed_from_np = self.np_rng.integers(0, 2**32, dtype=np.uint32)
        jax_key = jr.key(jax_seed_from_np)
        return self.gpjax_gaussian.sample(jax_key, sample_shape=(n,))


class gpjaxGP:
    """
    Wrapper around a gpjax conditional posterior GP, modified to store the 
    precision matrix for fast GP updates.
    """
    
    def __init__(self, gp: GPJaxConjugateGP, design: GPJaxDataset, rng: np.random.Generator):
        self.gp = gp
        self.design = design
        self.dim = self.design.in_dim
        self.sig2_obs = jnp.square(self.gp.likelihood.obs_stddev.value)
        self.Sigma_inv = self._compute_kernel_precision()
        self.rng = rng

    def __call__(self, x: Array, given: tuple[Array, Array] | None = None) -> GPJaxGaussian:
        if given is None:
            Sigma_inv = self.Sigma_inv
            design = self.design
        else:
            Sigma_inv, design = self._update_kernel_precision(given)

        return self._predict_using_precision(x, Sigma_inv, design)

    def _predict_using_precision(self, x: Array, Sigma_inv: Array, design: GPJaxDataset) -> gpjaxGaussian:
        """ Compute posterior GP multivariate normal prediction at test inputs `x`, conditional
            on dataset `design`. `Sigma_inv` is the inverse of the kernel matrix (including the noise covariance)
            evaluated at `design.X`.
        """
        x = x.reshape(-1, self.dim)
        x_design, y_design = design.X, design.y
        k_test_design =  self.gp.prior.kernel.cross_covariance(x, x_design)
        k_test_design_Sigma_inv = k_test_design @ Sigma_inv

        prior_mean_test = self.gp.prior.mean_function(x)
        prior_mean_design = self.gp.prior.mean_function(x_design)
        prior_cov_test = self.gp.prior.kernel.gram(x).to_dense()

        pred_mean = prior_mean_test + k_test_design_Sigma_inv @ (y_design - prior_mean_design)
        pred_cov = prior_cov_test - k_test_design_Sigma_inv @ k_test_design.T
        n_pred = x.shape[0]

        return gpjaxGaussian(pred_mean.squeeze(), 
                             psd(Dense(pred_cov.reshape(n_pred, n_pred))), 
                             self.rng)
    

    def _update_kernel_precision(self, given: tuple[Array, Array] | GPJaxDataset):
        """ 
        Updates the inverse kernel matrix Sigma_inv = (K + sig2*I)^{-1} when a 
        batch of new conditioning points Xnew are added. Returns the updated inverse 
        kernel matrix as well as the updated dataset (union of the old dataset and 
        the newly added points). `given` is either a gpjax `Dataset` containing the 
        new input/output pairs or a tuple (Xnew, ynew) containing the same information.
        """
        if isinstance(given, GPJaxDataset):
            Xnew, ynew = given.X, given.y
            new_dataset = self.design + given
        elif isinstance(given, tuple):
            Xnew = given[0].reshape(-1, self.dim)
            ynew = given[1].reshape(-1, 1)
            new_dataset = self.design + GPJaxDataset(X=Xnew, y=ynew)
        else:
            raise TypeError(f"`given` must be a tuple or gpjax Dataset object. Got {type(given)}")
        
        # Partitioned matrix inverse updates.
        num_new_points = Xnew.shape[0]
        Sigma_inv = self.Sigma_inv.copy()
        Xcurr = self.design.X.copy()

        for i in range(num_new_points):
            xnew = Xnew[i]
            Sigma_inv = self._update_single_point_kernel_precision(Sigma_inv, Xcurr, xnew)
            Xcurr = jnp.vstack([Xcurr, xnew])

        return Sigma_inv, new_dataset
    

    def _update_single_point_kernel_precision(self, Sigma_inv, X, xnew):
        """ 
        Updates the inverse kernel matrix Sigma_inv = (K + sig2*I)^{-1} when a 
        single conditioning point xnew is added. X is the current conditioning 
        set and xnew is the new point to add.
        """
        xnew = xnew.reshape(1, -1) # (1, 1)
        knm = self.gp.prior.kernel.cross_covariance(X, xnew) # (n, 1)
        Kinv_knm = Sigma_inv @ knm # (n, 1)
        k_new_new = self.gp.prior.kernel.gram(xnew).to_dense().squeeze() # scalar
        kappa = k_new_new + self.sig2_obs - (knm.T @ Kinv_knm).squeeze() # scalar
        
        outer = Kinv_knm @ Kinv_knm.T  # (n,1) @ (1,n) -> (n,n)
        top_left= Sigma_inv + outer / kappa # (n,n)
        top_right = -Kinv_knm / kappa
        bottom_left = top_right.T
        bottom_right = jnp.array([[1.0 / kappa]]) # (1,1)

        top = jnp.hstack([top_left, top_right])       # shape (n, n+1)
        bottom = jnp.hstack([bottom_left, bottom_right])  # shape (1, n+1)

        return jnp.vstack([top, bottom]) # (n+1, n+1)


    def _compute_kernel_precision(self):
        X = self.design.X
        K = add_jitter(self.gp.prior.kernel.gram(X).to_dense(), self.gp.jitter)
        Sigma = K + jnp.eye(K.shape[0]) * self.sig2_obs
        L_Sigma = jnp.linalg.cholesky(Sigma, upper=False)
        Sigma_inv = cho_solve((L_Sigma, True), jnp.eye(Sigma.shape[0]))
        return Sigma_inv


# -----------------------------------------------------------------------------
# MCMC algorithms for uncertainty propagation in surrogate-based inverse problems
# -----------------------------------------------------------------------------


# def get_mwg_eup_sampler(self, u_prop_scale=0.1, pcn_cor=0.99):
#     """
#     Exactly targets the EUP.
#     """
#     L_noise = self.noise.chol

#     # Extended state space. Initialize state via prior sample.
#     state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

#     # Target density.
#     def ldens_post(state):
#         fwd = self.G @ state.primary["u"] + state.primary["e"]
#         return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

#     target = TargetDensity(LogDensityTerm("post", ldens_post))

#     # u and e updates.
#     ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
#                                     term_subset="post", block_vars="u", rng=self.rng)
#     ker_e = DiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
#                                 cor_param=pcn_cor, term_subset="post",
#                                 block_vars="e", rng=self.rng)

#     # Sampler
#     alg = BlockMCMCSampler(target, initial_state=state,
#                             kernels=[ker_u, ker_e], rng=self.rng)
#     return alg

# def get_rk_sampler(inv_prob, u_prop_scale=0.1):
#     L_noise = self.noise.chol

#     # Initialize state via prior sample.
#     state = State(primary={"u": self.prior.sample()})

#     # Noisy target density.
#     def ldens_post_noisy(state):
#         fwd = self.G @ state.primary["u"] + self.e.sample()
#         return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

#     target = TargetDensity(LogDensityTerm("post", ldens_post_noisy), use_cache=False)

#     # Metropolis-Hastings updates.
#     ker = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov, rng=self.rng)

#     # Sampler
#     alg = BlockMCMCSampler(target, initial_state=state, kernels=ker, rng=self.rng)
#     return alg


# def get_rk_pcn_sampler(self, u_prop_scale=0.1, pcn_cor=0.9):
#     L_noise = self.noise.chol

#     # Extended state space. Initialize state via prior sample.
#     state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

#     # Target density.
#     def ldens_post(state):
#         fwd = self.G @ state.primary["u"] + state.primary["e"]
#         return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])
#     target = TargetDensity(LogDensityTerm("post", ldens_post))

#     # u and e updates.
#     ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
#                                   term_subset="post", block_vars="u", rng=self.rng)
#     ker_e = UncalibratedDiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
#                                           cor_param=pcn_cor, block_vars="e", rng=self.rng)

#     # Sampler
#     alg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e], rng=self.rng)
#     return alg


class RandomKernelPCNSampler(MCMCSampler):
    """
    An MCMC sampler to *approximately* sample from the expectation of the random
    measure pi(u; f) with random log density of the form log_p(f(u)), where f ~ GP(m, k).
    Precisely, one seeks to sample from E_f{pi(u; f)}.
     
    The algorithm rk-pcn sampler operates on the extended state space (u, f), which is in
    principle infinite dimensional. In reality, it is only necessary to instantiate 
    bivariate projections of the GP of the form [f(u), f(u')]. Therefore, the practical
    implementation of this algorithm maintains a state of the form (u, f(u)).
    """
    def __init__(self,
                 log_density: Callable[[Array], float],
                 gp: GaussianProcess,
                 u_init: Array,
                 u_prop_cov: Array | None = None,
                 pcn_cor: float = 0.99,
                 rng = None):
        
        # Extended state space.
        u_dim = u_init.flatten().shape[0]
        fu_init = gp(u_init.reshape(1,-1)).mean.reshape((1))
        init_state = State(primary={"u": u_init, "fu": fu_init})

        def target_log_density(state):
            return log_density(state.primary["fu"])
        target = TargetDensity(LogDensityTerm("density_given_f", target_log_density))
        super().__init__(target=target, initial_state=init_state, rng=rng)

        # Proposal distribution for u updates.
        if u_prop_cov is None:
            u_prop_cov = np.identity(u_dim)
        self.u_proposal = GaussianProposal(u_prop_cov, rng=self.rng)

        # Data for f updates.
        self.pcn_cor = pcn_cor
        self.gp = gp
        
    def step(self) -> State:
        """
        To clarify notation, let u, f denote current values of u and f, so that fu is the 
        current value of f(u). Let v, g be proposed values. So fv is the current value of 
        f evaluated at the proposed value of u.
        """

        current_state = self.current_state.copy_with(clear_cache=False)

        # Propose new u value so that the f update knows which bivariate projection to use
        u_block = current_state.extract_block(var_names="u")
        u = u_block["u"]
        fu = current_state.extract_block(var_names="fu")["fu"]
        v = self.u_proposal.propose(u_block)["u"]
        uv = np.vstack([u, v])

        # Just in time sample for f(v)
        fv = self.gp(v, given=(u, fu)).sample().reshape(1)
        fuv = np.concatenate([fu, fv])

        # f update (always accepted)
        fuv_dist = self.gp(uv)
        guv = DiscretePCNProposal(mean=fuv_dist.mean, cov=fuv_dist.cov, 
                                  cor_param=self.pcn_cor, rng=self.rng).propose({"fuv": fuv})["fuv"] # [g(u), g(v)]
        gu, gv = guv
        current_state = current_state.copy_with(primary_updates={"fu": gu.reshape(1)}) # {u, g(u)}

        # u update: note that proposal already occurred above
        candidate = current_state.copy_with(primary_updates={"u": v, "fu": gv.reshape(1)}) # {v, g(v)}
        new_state = _mh_accept_reject(self.target, current_state, candidate, self.rng)

        return new_state


def _mh_accept_reject(target: TargetDensity, 
                      current: State, 
                      candidate: State, 
                      rng: np.random.Generator) -> State:
    """ Assumes symmetric proposal """

    log_p_old = target(current)
    log_p_new = target(candidate)

    log_ratio = log_p_new - log_p_old

    # Accept/reject step.
    if np.log(rng.uniform()) < log_ratio:
        return candidate
    else:
        return current