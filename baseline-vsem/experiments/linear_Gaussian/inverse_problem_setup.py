# experiments/linear_Gaussian/inverse_problem_setup.py
"""
Helpers to construct the linear Gaussian deconvolution inverse problem
"""

import numpy as np
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt

from LinGaussTest import LinGaussInvProb
from Gaussian import Gaussian


def make_inverse_problem(rng, d, noise_sd, ker_length, ker_lengthscale, 
                         jitter=1e-8, G_scale=1.0, s=4):
    grid = np.arange(d)

    # Prior distribution
    m0 = np.zeros(d)
    C0 = gaussian_cov_mat(d, lengthscale=5, scale=1) + jitter * np.identity(d)

    # Forward model (convolution with sampling)
    G, G_conv, H, idx_obs, n = get_forward_model(d, ker_length, ker_lengthscale, s, G_scale)

    # Noise covariance
    variances = np.sqrt(noise_sd)**2 * np.ones(n)
    Sig = np.diag(variances)

    # Construct inverse problem object
    inv_prob = LinGaussInvProb(rng, G, m0, C0, Sig)
    g_conv_true = G_conv @ inv_prob.u_true
    
    return inv_prob, g_conv_true, grid, idx_obs


def get_forward_model(d, ker_length, ker_lengthscale, s, G_scale):
    s = 4 # every sth spatial index is observed
    ker_grid = gaussian_kernel_grid(ker_length, ker_lengthscale)
    G_conv = construct_toeplitz_forward_model(d, ker_grid)

    idx_obs = np.arange(0, d, s)
    n = len(idx_obs)
    H = np.zeros((n, d), dtype=int)
    H[np.arange(n), idx_obs] = 1
    G = G_scale * H @ G_conv

    return G, G_conv, H, idx_obs, n


def gaussian_kernel(x, lengthscale):
    return np.exp(-0.5 * (x / lengthscale) ** 2)

def gaussian_kernel_grid(size, lengthscale):
    """
    Evaluates discrete Gaussian kernel exp(-(i/sigma)^2 / 2) at 
    `size` integers. If `size` is odd this will result in symmetry 
    about zero, otherwise it will be off by one. e.g., for `size = 3`
    the values are evaluated at -1, 0, 1. For `size = 4` evaluated 
    at -2, -1, 0, 1. The kernel values are normalized to sum to one.
    """
    x = np.arange(size) - size // 2
    kernel = np.exp(-0.5 * (x / lengthscale) ** 2)
    kernel /= kernel.sum()
    return kernel

def gaussian_cov_mat(d, lengthscale, scale):
    Sig = np.zeros((d,d))
    for i in range(d):
        for j in range(i+1):
            k = gaussian_kernel(i-j, lengthscale)
            Sig[i,j] = k
            Sig[j,i] = k
            
    return scale**2 * Sig

def linear_same_convolution(u, k_vals):
    """
    Linear convolution of `u` with `k_vals`, returning vector of same 
    length as `u`. Discrete convolution of the "same" variety.
    """

    out = np.convolve(u, k_vals, mode="full")

    # Clip so output is of length equal to `len(u)`
    start = len(k_vals) // 2
    end = start + len(u)
    return out[start:end]

def construct_toeplitz_forward_model(d, ker_vals):
    """
    Returns linear matrix G representing the forward model. This is a 
    Toeplitz matrix encoding the discrete linear convolution.
    d is the dimension of the discretized signal u, and ker_vals
    is the vector of discrete kernel evaluations.

    Returns a (d,d) Toeplitz matrix encoding a discrete linear 
    "same" convolution.
    """

    kernel_length = len(ker_vals)

    first_col = np.zeros(d)
    first_col[:kernel_length//2+1] = ker_vals[kernel_length//2::-1]
    
    first_row = np.zeros(d)
    first_row[:kernel_length//2+1] = ker_vals[kernel_length//2:]
    
    G = toeplitz(first_col, first_row)
    return G


def construct_noise_cov(G, variances):
    # Compose covariance matrix in singular basis
    U, s, Vh = np.linalg.svd(G)
    C_eps = (Vh.T * variances) @ Vh

    return C_eps
    
# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_exact_post(inv_prob, grid, idx_obs, colors, g_conv_true=None, 
                    title=None, alpha_prior=0.1, alpha_post=0.3):

    fig, ax = plt.subplots()

    # prior intervals
    prior_sd = np.sqrt(np.diag(inv_prob.prior.cov))
    ci_lower_prior = inv_prob.prior.mean - 2 * prior_sd
    ci_upper_prior = inv_prob.prior.mean + 2 * prior_sd

    # posterior intervals
    post_sd = np.sqrt(np.diag(inv_prob.post.cov))
    ci_lower = inv_prob.post.mean - 2 * post_sd
    ci_upper = inv_prob.post.mean + 2 * post_sd

    title = 'Exact Posterior' if title is None else title

    ax.fill_between(grid, ci_lower_prior, ci_upper_prior, color=colors['aux'], 
                    alpha=alpha_prior, label="prior")
    ax.fill_between(grid, ci_lower, ci_upper, color=colors['exact'], 
                    alpha=alpha_post)

    if inv_prob.u_true is not None:
        ax.plot(grid, inv_prob.u_true, color="black", label="true u")
    if g_conv_true is not None:
        ax.plot(grid, g_conv_true, color="orange", label="g true")

    ax.plot(idx_obs, inv_prob.y, "o", color="red")
    ax.plot(grid, inv_prob.post.mean, color=colors['exact'], label="posterior mean")

    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_surrogate(inv_prob, test, grid, idx_obs, colors, include_u_true=False):

    fig, ax = plt.subplots()

    # true values
    u_true = inv_prob.u_true
    g_true = inv_prob.G @ u_true

    if include_u_true:
        ax.plot(grid, inv_prob.u_true, color='black', label="u true")
    ax.plot(idx_obs, g_true, 'o-', color=colors['exact'], label="true")
 
    # surrogate values
    surrogate_mean = test.G @ u_true + test.e.mean
    surrogate = Gaussian(mean=surrogate_mean, cov=test.e.cov)
    surrogate_sd = np.sqrt(np.diag(surrogate.cov))
    ci_lower = surrogate.mean - 2 * surrogate_sd
    ci_upper = surrogate.mean + 2 * surrogate_sd
    ax.fill_between(idx_obs, ci_lower, ci_upper, color=colors['mean'], alpha=0.1)
    ax.plot(idx_obs, surrogate.mean, 'o-',color=colors['mean'], label="surrogate")

    ax.set_xlabel('u')
    ax.set_title('surrogate predictive distribution')
    ax.legend()
    plt.close()

    return fig, ax


def plot_approx_post(test, grid, idx_obs, post_name):
    if post_name == "exact":
        post_rv = test.post
    elif post_name == "eup":
        post_rv = test.eup_post
    elif post_name == "ep":
        post_rv = test.ep_post
    elif post_name == "mean":
        post_rv = test.mean_post
    else:
        raise ValueError(f"Invalid post_name {post_name}")

    post_sd = np.sqrt(np.diag(post_rv.cov))
    ci_lower = post_rv.mean - 2 * post_sd
    ci_upper = post_rv.mean + 2 * post_sd

    plt.fill_between(grid, ci_lower, ci_upper, color='blue', alpha=0.1, label="+/- 2 post sd")
    plt.plot(idx_obs, test.y, "o", color="red", label="y")
    plt.plot(grid, post_rv.mean, color="blue", label="post mean")
    plt.title('exact posterior')
    plt.legend()
    plt.show()


def plot_approx_post_comparison(inv_prob, test, grid, idx_obs, post_name):
    """
    Plot approximate posterior overlaid on exact posterior. Note that the
    exact posterior is taken from `inv_prob`, not `test.inv_prob`. This
    allows plotting with respect to different baseline posteriors.
    """
    if post_name == "eup":
        approx_rv = test.eup_post
    elif post_name == "ep":
        approx_rv = test.ep_post
    elif post_name == "mean":
        approx_rv = test.mean_post
    elif post_name == "misspecified":
        approx_rv = test.inv_prob.post
    else:
        raise ValueError(f"Invalid post_name {post_name}")

    # Baseline exact
    exact_rv = inv_prob.post
    exact_sd = np.sqrt(np.diag(exact_rv.cov))
    ci_lower_exact = exact_rv.mean - 2 * exact_sd
    ci_upper_exact = exact_rv.mean + 2 * exact_sd
    plt.fill_between(grid, ci_lower_exact, ci_upper_exact, color='blue', alpha=0.1, label="+/- 2 exact post sd")
    plt.plot(grid, exact_rv.mean, color="blue", label="exact post mean")

    # Approximation
    approx_sd = np.sqrt(np.diag(approx_rv.cov))
    ci_lower_approx = approx_rv.mean - 2 * approx_sd
    ci_upper_approx = approx_rv.mean + 2 * approx_sd
    plt.fill_between(grid, ci_lower_approx, ci_upper_approx, color='green', alpha=0.1, label="+/- 2 approx post sd")
    plt.plot(grid, approx_rv.mean, color="green", label="approx post mean")

    plt.plot(idx_obs, test.y, "o", color="red", label="y")
    plt.title(f'Posterior Comparison: Exact vs. {post_name}')
    plt.legend()
    plt.show()


def summarize_setup(inv_prob, test, grid, idx_obs, g_conv_true=None):
    """
    `inv_prob` should be true data generating process. `test` is a LinGaussTest
    (which may be based on a misspecified inverse problem model).
    """
    plot_exact_post(inv_prob, grid, idx_obs, g_conv_true, title='correct model')
    plot_approx_post_comparison(inv_prob, test, grid, idx_obs, post_name='misspecified')
    plot_surrogate(inv_prob, test, grid, idx_obs, g_conv_true)
    plot_approx_post_comparison(inv_prob, test, grid, idx_obs, post_name='mean')
    plot_approx_post_comparison(inv_prob, test, grid, idx_obs, post_name='eup')
    plot_approx_post_comparison(inv_prob, test, grid, idx_obs, post_name='ep')