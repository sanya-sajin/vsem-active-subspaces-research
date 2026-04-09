import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_triangular, cholesky
from scipy.special import logsumexp

from Gaussian import Gaussian
from modmcmc import State, BlockMCMCSampler, LogDensityTerm, TargetDensity
from modmcmc.kernels import GaussMetropolisKernel, DiscretePCNKernel, UncalibratedDiscretePCNKernel, mvn_logpdf

def get_random_corr_mat(dim, rng):
    # Random orthogonal matrix
    Q, _ = np.linalg.qr(rng.normal(size=(dim,dim)))
    # Random eigenvalues between 0 and 1
    eigvals = rng.uniform(0, 1, size=dim)
    eigvals /= eigvals.sum()  # Ensure psd
    matrix = Q @ np.diag(eigvals) @ Q.T # psd matrix

    # Convert to corr.
    D = np.sqrt(np.diag(matrix))
    corr = matrix / np.outer(D, D)
    np.fill_diagonal(corr, 1)
    return corr


def get_col_hist_grid(*arrays, bins=30, nrows=1, ncols=None, figsize=(5,4),
                      col_labs=None, plot_labs=None, density=True, hist_kwargs=None):
    """
    Given multiple (n, d) arrays (same d), return one matplotlib Figure per column.
    Each Figure contains one histogram per array (for that column).

    Parameters:
        *arrays: arbitrary number of numpy arrays, each shape (n, d)
        bins: number of histogram bins (default=30)
        hist_kwargs: dict of extra kwargs for plt.hist (optional)
        col_labs: list of d names associated with the columns.
        plot_labs: list of len(arrays) names associated with the arrays.

    Returns:
        figs: list of matplotlib Figure objects (len=d)
    """

    n_dims = arrays[0].shape[1]
    if not all(a.shape[1] == n_dims for a in arrays):
        raise ValueError("All arrays must have the same number of columns")

    if hist_kwargs is None:
        hist_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_dims / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_dims)]
    if plot_labs is None:
        plot_labs = [f"Array {i+1}" for i in range(len(arrays))]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_dims):
        ax = axs[col]
        for idx, arr in enumerate(arrays):
            ax.hist(arr[:, col], bins=bins, alpha=0.5, label=plot_labs[idx],
                    density=density, **hist_kwargs)
        ax.set_title(col_labs[col])
        ax.legend()

    # Hide unused axes and close figure.
    for k in range(n_dims, nrows*ncols):
        fig.delaxes(axs[k])
    plt.close(fig)

    return fig


def get_trace_plots(samp_arr, nrows=1, ncols=None, figsize=(5,4),
                    col_labs=None, plot_kwargs=None):
    """
    Generate one trace plot per column of `samp_arr`.

    Parameters:
        arr: numpy array of shape (n, d)
        col_labs: list of labels associated with each column of `samp_arr`.

    Returns:
        fig: matplotlib Figure object
        ax: matplotlib Axes object
    """
    n_itr, n_cols = samp_arr.shape
    x = np.arange(n_itr)

    if plot_kwargs is None:
        plot_kwargs = {}

    if ncols is None:
        ncols = int(np.ceil(n_cols / nrows))

    if col_labs is None:
        col_labs = [f"Column {i}" for i in range(n_cols)]

    fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
    axs = np.array(axs).reshape(-1)
    for col in range(n_cols):
        ax = axs[col]
        ax.plot(x, samp_arr[:,col], **plot_kwargs)
        ax.set_title(col_labs[col])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Value")

    # Hide unused axes and close figure.
    for k in range(n_cols, nrows*ncols):
        fig.delaxes(axs[k])
    plt.close(fig)

    return fig

def estimate_maginal_coverage(*samp, baseline, probs=None):
    """
    Empirical marginal (one dim at a time) coverage, with respect to `(n,d)`
    samples `baseline` from a baseline (ground truth) distribution.
    `samp` is a list of sample matrices (each with `d` columns) to compare
    to the baseline. `probs` is a list of coverage levels to compute
    (e.g., 0.9 = 90% coverage interval). Returns list of length `len(samp)`
    with each element containing a `(m,d)` array, where `m = len(probs)`.
    The `(i,j)` element of matrix `l` is the actual coverage of the `jth`
    marginal of `samp[l]` (with respect to `baseline`) at level `probs[i]`.
    """

    # List of sample matrices.
    samp_list = list(samp)
    n_baseline = baseline.shape[0]

    # Lower and upper quantiles defining coverage sets.
    if probs is None:
        probs = np.arange(0.1, 1.01, step=0.1)
    lower = 0.5 * (1 - probs)
    upper = 0.5 * (1 + probs)

    actual_coverage = []

    for x in samp_list:
        # Nominal coverage.
        nominal_lower = np.quantile(x, q=lower, axis=0)
        nominal_upper = np.quantile(x, q=upper, axis=0)

        # Actual coverage.
        # nominal_lower,nominal_upper: (m, d), baseline: (n, d)
        # Broadcast: nominal_lower (m, d) to (m, 1, d)
        #            baseline      (n, d) to (1, n, d)
        # Compare along axis 1 (baseline rows).
        covered = np.logical_and(baseline[None,:,:] >= nominal_lower[:,None,:],
                                 baseline[None,:,:] <= nominal_upper[:,None,:]) # (m, n, d)
        actual_coverage.append(covered.sum(axis=1) / n_baseline)  # shape (m, d)

    return actual_coverage

# ------------------------------------------------------------------------------
# Approximate MCMC schemes.
# ------------------------------------------------------------------------------

def run_sampler(sampler, n_steps=50000, burn_in_start=25000, trace_plot_kwargs=None):
    sampler.sample(num_steps=n_steps)

    # Store samples in array.
    d = sampler.trace[0].primary["u"].shape[0]
    itr_range = range(burn_in_start, len(sampler.trace))
    samp = np.empty((len(itr_range), d))

    for i in range(len(itr_range)):
        samp[i,:] = sampler.trace[i].primary["u"]

    # Produce trace plots.
    if trace_plot_kwargs is None:
        trace_plot_kwargs = {}
    trace_plots = plot_trace(samp, **trace_plot_kwargs)

    return samp, sampler, trace_plots


def get_approx_mwg_sampler(y, u, G, noise, e, rng, u_prop_scale=0.1,
                           pcn_cor=0.9, n_samp_norm_ratio=100):
    L_noise = noise.chol
    d = u.dim

    # Extended state space. Initialize state via prior sample.
    state = State(primary={"u": u.sample(), "e": e.sample()})

    # Target density.
    # ldens_surrogate = lambda state: e.log_p(state.primary["e"])
    # def ldens_post(state):
    #     fwd = G @ state.primary["u"] + state.primary["e"]
    #     return mvn_logpdf(y, mean=fwd, L=L_noise) + u.log_p(state.primary["u"])

    llik = Gaussian(mean=y, chol=L_noise)

    # Vectorized over u_vals (num_u,d). Uses single e. Output is (num_u,).
    def ldens_post(u_vals, e):
        fwd = u_vals @ G.T + e # (num_u, n)
        return llik.log_p(fwd) + u.log_p(u_vals)

    def ldens_post_norm_approx(state):
        e_val = state.primary["e"]
        log_post = ldens_post(state.primary["u"].reshape((1,d)), e_val)
        log_Z_approx = logsumexp(ldens_post(u.sample(n_samp_norm_ratio), e_val))
        return log_post - log_Z_approx

    target = TargetDensity(LogDensityTerm("post", ldens_post_norm_approx), use_cache=False)

    # u and e updates.
    ker_u = GaussMetropolisKernel(target, proposal_cov=u_prop_scale*u.cov,
                                  term_subset="post", block_vars="u", rng=rng)
    ker_e = DiscretePCNKernel(target, mean_Gauss=e.mean, cov_Gauss=e.cov,
                              cor_param=pcn_cor, term_subset="post",
                              block_vars="e", rng=rng)

    # Sampler
    mwg = BlockMCMCSampler(target, initial_state=state, kernels=[ker_u, ker_e])
    return mwg
