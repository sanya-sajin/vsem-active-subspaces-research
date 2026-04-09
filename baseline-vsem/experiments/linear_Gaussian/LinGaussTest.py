# experiments/test/linear_Gaussian/LinGaussTest.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import solve_triangular, cholesky

from Gaussian import Gaussian, trace_Ainv_B, log_det_tri, squared_mah_dist
from helper import get_col_hist_grid, get_trace_plots, get_random_corr_mat

from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import (
    MarkovKernel, 
    GaussMetropolisKernel, 
    DiscretePCNKernel, 
    UncalibratedDiscretePCNKernel, 
    mvn_logpdf
)

class LinGaussInvProb:
    # Exact (no surrogate) linear Gaussian inverse problem.

    def __init__(self, rng, G, m0=None, C0=None, Sig=None, y=None):
        self.rng = rng
        self.n = G.shape[0]
        self.d = G.shape[1]
        self.u_names = [f"u{i+1}" for i in range(self.d)]

        # Linear forward model
        self.G = G

        # Prior on parameter
        if m0 is None:
            m0 = np.zeros(self.d)
        if C0 is None:
            C0 = np.identity(self.d)
        self.prior = Gaussian(mean=m0, cov=C0, rng=rng)

        # Observation noise
        if Sig is None:
            Sig = np.identity(self.n)
        self.noise = Gaussian(cov=Sig, rng=rng, store="both")

        # Ground truth parameter and observed data
        if y is None:
            self.u_true = self.prior.sample()
            self.noise_realization = self.noise.sample()
            self.g_true = self.G @ self.u_true
            self.y = self.g_true + self.noise_realization
        else:
            self.y = y
            self.u_true = None
            self.noise_realization = None
            self.g_true = None

        # Exact posterior
        self.post = self.prior.invert_affine_Gaussian(self.y, A=self.G,
                                                      cov_noise=self.noise.cov)


    def plot_marginals(self, nrows=1, ncols=None, figsize=(5,4)):
        # Plot marginal Gaussian prior and posterior densities.

        if ncols is None:
            ncols = int(np.ceil(self.d / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(self.d):
            ax = axs[j]
            m0,s0 = self.prior.mean[j], np.sqrt(self.prior.cov[j,j])
            m,s = self.post.mean[j], np.sqrt(self.post.cov[j,j])

            bounds = norm.ppf([0.01, 0.99], loc=m0, scale=s0)
            x = np.linspace(bounds[0], bounds[1], 100)
            ax.plot(x, norm.pdf(x, loc=m0, scale=s0), label="prior")
            ax.plot(x, norm.pdf(x, loc=m, scale=s), label="posterior")
            ax.set_title(self.u_names[j])
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(self.d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig

    def plot_G_singular_vals(self, semilog=False, whiten=False):
        forward_model = self.G
        if whiten:
            forward_model = solve_triangular(self.noise.chol, forward_model, lower=True)
            title = "singular values of whitened G"
        else:
            title = "singular values of G"

        U, s, Vh = np.linalg.svd(forward_model)

        fig, ax = plt.subplots(1,1)

        if semilog:
            ax.semilogy(s)
        else:
            ax.plot(np.arange(len(s)), s)

        ax.set_title(title)
        ax.set_xlabel("index")
        ax.set_ylabel("singular value")

        plt.close(fig)
        return fig


class LinGaussTest:
    def __init__(self, inv_prob, Q, r=None):
        # `inv_prob` is a `LinGaussInvProb` object. If not provided, the
        # emulator additive bias r is sampled from N(0, Q) (and then fixed).

        # Exact inverse problem.
        self.inv_prob = inv_prob
        self.rng = inv_prob.rng
        self.d = inv_prob.d
        self.n = inv_prob.n
        self.u_names = inv_prob.u_names
        self.G = inv_prob.G
        self.prior = inv_prob.prior
        self.noise = inv_prob.noise
        self.u_true = self.prior.sample()
        self.y = inv_prob.y
        self.post = inv_prob.post

        # Well-calibrated surrogate (sampling bias mean r from N(0,Q)).
        if r is None:
            r = Gaussian(cov=Q, rng=self.rng).sample()
        self.e = Gaussian(mean=r, cov=Q, rng=self.rng) # Emulator is Gu + e, e ~ N(r, Q)

        # Surrogate-based inversion.
        self.ep_post = self.get_ep_rv()
        self.eup_post = self.get_eup_rv()
        self.mean_post = self.get_plugin_mean_rv()

        # MCMC samplers.
        self.samplers = {"mwg-eup": self.get_mwg_eup_sampler(),
                         "rk": self.get_rk_sampler(),
                         "rk-pcn": self.get_rk_pcn_sampler()}

    def get_ep_rv(self):
        L_Sig = self.noise.chol
        C1 = solve_triangular(L_Sig, self.G, lower=True)
        C2 = solve_triangular(L_Sig.T, C1, lower=False)
        B = -self.post.cov @ C2.T
        ep = self.e.convolve_with_Gaussian(A=B, b=self.post.mean, cov_new=self.post.cov)
        return ep

    def get_eup_rv(self):
        eup = self.prior.invert_affine_Gaussian(self.y, A=self.G, b=self.e.mean,
                                                cov_noise=self.noise.cov + self.e.cov)
        return eup
    
    def get_plugin_mean_rv(self):
        plugin = self.prior.invert_affine_Gaussian(self.y, A=self.G, b=self.e.mean,
                                                   cov_noise=self.noise.cov)
        return plugin

    def sample_surrogate_posterior(self):
        """
        Draws a trajectory of a surrogate, and returns the Gaussian posterior
        conditional on that trajectory.
        """
        return self.prior.invert_affine_Gaussian(self.y, A=self.G, b=self.e.sample(),
                                                 cov_noise=self.noise.cov)

    def estimate_expected_kl(self, n_samp=1000):
        """
        Estimates E[KL(pi_star || pi_EP)] and E[KL(pi_star || pi_EUP)], where
        the expectation is with respect to the emulator predictive distribution.
        """
        post_trajectories = [self.sample_surrogate_posterior() for i in range(n_samp)]
        ep_kl = np.mean([p.kl(self.ep_post) for p in post_trajectories])
        eup_kl = np.mean([p.kl(self.eup_post) for p in post_trajectories])

        return ep_kl, eup_kl

    def calc_expected_kl(self):
        """
        Calculates E[KL(pi_star || pi_EP)] and E[KL(pi_star || pi_EUP)], where
        the expectation is with respect to the emulator predictive distribution.
        """

        # E[KL(pi_star || pi_EP)]
        ep_kl = 0.5 * (log_det_tri(self.ep_post.chol) - log_det_tri(self.post.chol))

        # E[KL(pi_star || pi_EUP)]
        # H = C G^T \Sigma^{-1}
        # H = solve_triangular(self.noise.chol, self.G @ self.post.cov, lower=True).T
        H = self.post.cov @ self.G.T @ np.linalg.inv(self.noise.cov)
        shifted_Gaussian_eup = Gaussian(mean=self.eup_post.mean + H @ self.e.mean,
                                        chol=self.eup_post.chol, rng=self.rng)
        HQHt = H @ self.e.cov @ H.T
        L_HQHt = cholesky(HQHt, lower=True)
        eup_kl = self.post.kl(shifted_Gaussian_eup) + trace_Ainv_B(self.eup_post.chol, L_HQHt)

        # TEMP
        test = 0.5 * (log_det_tri(self.eup_post.chol) - log_det_tri(self.post.chol)) + \
                squared_mah_dist(self.post.mean, self.eup_post.mean + H @ self.e.mean, L=self.eup_post.chol) + \
                trace_Ainv_B(self.eup_post.chol, L_HQHt) + trace_Ainv_B(self.eup_post.chol, self.post.chol) - self.d
        print(test)

        return ep_kl, eup_kl

    def run_sampler(self, sampler_name, n_samp, sampler_kwargs=None, plot_kwargs=None):
        """
        Runs MCMC sampler, collects samples, and drops burn-in. Returns
        `n_samp` samples after burn-in.
        """
        if sampler_kwargs is None:
            sampler_kwargs = {}
        if plot_kwargs is None:
            plot_kwargs = {}

        sampler = self.samplers[sampler_name]
        sampler.sample(num_steps=2*n_samp, **sampler_kwargs)

        # Store samples in array.
        len_trace = len(sampler.trace)
        itr_range = np.arange(len_trace-n_samp, len_trace)
        samp = np.empty((n_samp, self.d))

        for samp_idx,trace_idx in enumerate(itr_range):
            samp[samp_idx,:] = sampler.trace[trace_idx].primary["u"]

        return samp, self.get_trace_plot(samp, **plot_kwargs)

    def reset_samplers(self):
        for sampler in self.samplers.values():
            sampler.reset()

    def get_sample_list(self, n_samp, include=None):
        # `include` can include ["post", "eup", "ep", "mwg-eup", "rk", "rk-pcn"]
        samp_list = []
        labels = []
        if include is None:
            include = ["post", "eup", "ep"]

        if "post" in include:
            samp_list.append(self.post.sample(n_samp))
            labels.append("post")
        if "ep" in include:
            samp_list.append(self.ep_post.sample(n_samp))
            labels.append("ep")
        if "eup" in include:
            samp_list.append(self.eup_post.sample(n_samp))
            labels.append("eup")
        if "mwg-eup" in include:
            samp_list.append(self.run_sampler("mwg-eup", n_samp)[0])
            labels.append("mwg-eup")
        if "rk" in include:
            samp_list.append(self.run_sampler("rk", n_samp)[0])
            labels.append("rk")
        if "rk-pcn" in include:
            samp_list.append(self.run_sampler("rk-pcn", n_samp)[0])
            labels.append("rk-pcn")

        return samp_list, labels

    def get_hist_plot(self, n_samp=100000, include=None, idcs=None, plot_kwargs=None):
        samp_list, plot_labs = self.get_sample_list(n_samp, include)
        if plot_kwargs is None:
            plot_kwargs = {}

        # Subset of dimensions in parameter space
        if idcs is not None:
            samp_list = [samp[:,idcs] for samp in samp_list]

        fig = get_col_hist_grid(*samp_list, plot_labs=plot_labs, col_labs=self.u_names,
                                density=True, **plot_kwargs)
        return fig

    def get_trace_plot(self, samp, plot_kwargs=None):
        if plot_kwargs is None:
            plot_kwargs = {}
        fig = get_trace_plots(samp, col_labs=self.u_names, **plot_kwargs)
        return fig

    def get_mwg_eup_sampler(self, u_prop_scale=0.1, pcn_cor=0.99):
        """
        Exactly targets the EUP.
        """
        L_noise = self.noise.chol

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

        # Target density.
        def ldens_post(state):
            fwd = self.G @ state.primary["u"] + state.primary["e"]
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

        target = TargetDensity(LogDensityTerm("post", ldens_post))

        # u and e updates.
        ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                      term_subset="post", block_vars="u", rng=self.rng)
        ker_e = DiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                  cor_param=pcn_cor, term_subset="post",
                                  block_vars="e", rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state,
                               kernels=[ker_u, ker_e], rng=self.rng)
        return alg

    def get_rk_sampler(self, u_prop_scale=0.1):
        L_noise = self.noise.chol

        # Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample()})

        # Noisy target density.
        def ldens_post_noisy(state):
            fwd = self.G @ state.primary["u"] + self.e.sample()
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])

        target = TargetDensity(LogDensityTerm("post", ldens_post_noisy), use_cache=False)

        # Metropolis-Hastings updates.
        ker = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov, rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state, kernels=ker, rng=self.rng)
        return alg

    def get_rk_pcn_sampler(self, u_prop_scale=0.1, pcn_cor=0.99):
        L_noise = self.noise.chol

        # Extended state space. Initialize state via prior sample.
        state = State(primary={"u": self.prior.sample(), "e": self.e.sample()})

        # Target density.
        def ldens_post(state):
            fwd = self.G @ state.primary["u"] + state.primary["e"]
            return mvn_logpdf(self.y, mean=fwd, L=L_noise) + self.prior.log_p(state.primary["u"])
        target = TargetDensity(LogDensityTerm("post", ldens_post))

        # u and e updates.
        ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*self.prior.cov,
                                      term_subset="post", block_vars="u", rng=self.rng)
        ker_e = UncalibratedDiscretePCNKernel(target, mean_Gauss=self.e.mean, cov_Gauss=self.e.cov,
                                              cor_param=pcn_cor, block_vars="e", rng=self.rng)

        # Sampler
        alg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e], rng=self.rng)
        return alg

    def direct_sample_ep(self, n_samp=100000):
        """
        An alternative method to sample the EP. The analytical approach is
        faster, this is mostly just used for validation.
        """
        samp = np.empty((n_samp, self.d))
        for i in range(n_samp):
            samp[i,:] = self.prior.invert_affine_Gaussian(y, A=self.G, b=self.e.sample(),
                                                          cov_noise=self.noise.cov).sample()
        return samp

    def calc_coverage(self, probs=None):
        """
        Computes marginal (one dim at a time) coverage for EP and EUP, with
        respect to the baseline exact posterior. Returns dictionary with
        keys "ep", "eup", and "probs". "probs" contains the list of
        coverage levels (e.g., 0.9 = 90% coverage interval). The other two
        elements are each `(m,d)` arrays, where `m = len(probs)`.
        The `(i,j)` element of the array is the actual coverage of the `jth`
        marginal of EUP or EP (with respect to the exact posterior baseline)
        at level `probs[i]`.
        """

        # Lower and upper quantiles defining coverage sets.
        if probs is None:
            probs = np.arange(0.1, 1.01, step=0.1)
        lower = 0.5 * (1 - probs)
        upper = 0.5 * (1 + probs)
        intervals = np.array([lower, upper])

        m = len(probs)
        actual_coverage = {"ep": np.empty((m, self.d)),
                           "eup": np.empty((m, self.d)),
                           "probs": probs}

        # Loop over 1d marginals.
        for j in range(self.d):
            # Nominal coverage.
            nominal_ep = norm.ppf(intervals, loc=self.ep_post.mean[j],
                                  scale=np.sqrt(self.ep_post.cov[j,j]))
            nominal_eup = norm.ppf(intervals, loc=self.eup_post.mean[j],
                                   scale=np.sqrt(self.eup_post.cov[j,j]))

            # Actual coverage.
            actual_ep = norm.cdf(nominal_ep, loc=self.post.mean[j],
                                 scale=np.sqrt(self.post.cov[j,j]))
            actual_eup = norm.cdf(nominal_eup, loc=self.post.mean[j],
                                  scale=np.sqrt(self.post.cov[j,j]))
            actual_coverage["ep"][:,j] = actual_ep[1,:] - actual_ep[0,:]
            actual_coverage["eup"][:,j] = actual_eup[1,:] - actual_eup[0,:]

        return actual_coverage

    def plot_coverage(self, coverage_info=None, probs=None, nrows=1,
                      ncols=None, figsize=(5,4)):

        if coverage_info is None:
            coverage_info = self.calc_coverage(probs)
        if ncols is None:
            ncols = int(np.ceil(self.d / nrows))

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(self.d):
            ax = axs[j]
            ax.plot(coverage_info["probs"], coverage_info["ep"][:,j], label="ep")
            ax.plot(coverage_info["probs"], coverage_info["eup"][:,j], label="eup")
            ax.set_title(self.u_names[j])
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Actual Coverage")

            # Add line y = x
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            y = x
            ax.plot(x, y, color="red", linestyle="--")
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(self.d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig