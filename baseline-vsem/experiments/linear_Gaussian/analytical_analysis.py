# experiments/linear_Gaussian/analytical_analysis.py

import matplotlib.pyplot as plt 
import numpy as np

def plot_eigenvalue_comparison(G, q_vals=None, c0=1.0, sig=1.0,
                               s_idcs=None):
    if q_vals is None:
        q_vals = np.linspace(0, 6, num=100)

    U, s, Vh = np.linalg.svd(G)
    if s_idcs is None:
        s_idcs = np.arange(0, len(s), len(s) // 3) 
    
    eig_exact = lambda_exact(s, c0=c0, sig=sig)
    eig_eup = lambda_eup(s, q_vals, c0=c0, sig=sig)
    eig_ep = lambda_ep(s, q_vals, c0=c0, sig=sig)

    eig_eup_norm = eig_eup / eig_exact
    eig_ep_norm = eig_ep / eig_exact

    fig, axs = plt.subplots(1, 2, figsize=(8,4))

    q_over_sigma = q_vals / sig

    # eup
    ax = axs[0]
    eup_eig_plot = ax.plot(q_over_sigma, eig_eup_norm[:,s_idcs])
    ax.set_title('eup')
    ax.set_xlabel('q / sigma')
    ax.set_ylabel('lambda approx / lambda exact')

    # ep
    ax = axs[1]
    ep_eig_plot = ax.plot(q_over_sigma, eig_ep_norm[:,s_idcs])
    ax.set_title('ep')
    ax.set_xlabel('q / sigma')
    ax.set_ylabel(None)
    ax.legend(ep_eig_plot, s_idcs, title='svd direction')

    plt.close()
    return fig, axs

def plot_mean_comparison(G, y, r, q_vals=None, c0=1.0, sig=1.0,
                         s_idcs=None):
    if q_vals is None:
        q_vals = np.linspace(0, 6, num=100)

    U, s, Vh = np.linalg.svd(G)
    if s_idcs is None:
        s_idcs = np.arange(0, len(s), len(s) // 3)

    alphas_exact = alpha_exact(s=s, y=y, U=U, c0=c0, sig=sig)
    alphas_eup = alpha_eup(s=s, y=y, U=U, r=r, q=q_vals, c0=c0, sig=sig)
    alphas_ep = alpha_ep(s=s, y=y, U=U, r=r, q=q_vals, c0=c0, sig=sig)
    alphas_eup_norm = (alphas_eup - alphas_exact) / alphas_exact
    alphas_ep_norm = (alphas_ep - alphas_exact) / alphas_exact

    fig, axs = plt.subplots(1, 2, figsize=(8,4))
    q_over_sigma = q_vals / sig

    # eup
    ax = axs[0]
    eup_mean_plot = ax.plot(q_over_sigma, alphas_eup_norm[:,s_idcs])
    ax.set_title('eup')
    ax.set_xlabel('q / sigma')
    ax.set_ylabel('alpha approx - alpha exact')

    # ep
    ax = axs[1]
    ep_mean_plot = ax.plot(q_vals, alphas_ep_norm[:,s_idcs])
    ax.set_title('ep')
    ax.set_xlabel('q / sigma')
    ax.set_ylabel(None)
    ax.legend(ep_mean_plot, s_idcs, title='svd direction')

    plt.close()
    return fig, axs


def lambda_exact(s, c0=1., sig=1.):
    """
    s, sig2 can be flat arrays of potentially different length
    """
    s = np.asarray(s)
    sig2 = np.asarray(sig) ** 2
    c02 = c0 ** 2
    c02_s = c02 * s

    if sig2.ndim == 0:
        sig2 = sig2.reshape(1)
    if c02_s.ndim == 0:
        c02_s = c02_s.reshape(1)

    denom = 1. + c02_s[np.newaxis] / sig2[:,np.newaxis] # (nsig, ns)
    return c02 / denom


def lambda_eup(s, q, c0, sig):
    """
    s, q can be flat arrays of potentially different length 
    """
    return lambda_exact(s, c0=c0, sig=np.sqrt(sig**2 + q**2))


def lambda_ep(s, q, c0, sig):
    """
    s, q can be flat arrays of potentially different length 
    """
    eig_exact = lambda_exact(s, c0=c0, sig=sig)

    q2 = np.asarray(q) ** 2
    s2 = np.asarray(s) ** 2
    sig4 = np.asarray(sig) ** 4
    c04 = c0 ** 4

    if s2.ndim == 0:
        s2 = s2.reshape(1)
    if q2.ndim == 0:
        q2 = q2.reshape(1)

    q2_s2 = s2[np.newaxis] * q2[:,np.newaxis] # (nq, ns)
    inflation = q2_s2 * c04 / sig4

    return eig_exact + inflation

def alpha_exact(s, y, U, c0=1., sig=1.):
    """
    s, q can be flat arrays of potentially different length
    This assumes m0 = 0 so we can ignore the prior term.
    U contains the left singular vectors of G; num cols = length of s
    """
    s = np.asarray(s)
    sig2 = np.asarray(sig) ** 2
    if sig2.ndim == 0:
        sig2 = sig2.reshape(1)

    eig_exact = lambda_exact(s, c0=c0, sig=sig) # (nsig, ns)
    eig_sig2 = eig_exact / sig2[:,np.newaxis] # (nsig, ns)
    Uy = U.T @ y # (ns,)
    s_Uy = s * Uy # (ns,)
    alpha = s_Uy * eig_sig2 # (nsig, ns)

    return alpha

def alpha_eup(s, y, U, r, q, c0=1., sig=1.):
    return alpha_exact(s=s, 
                       y=y-r, 
                       U=U, 
                       c0=c0, 
                       sig=np.sqrt(sig**2 + q**2))

def alpha_ep(s, y, U, r, q, c0=1., sig=1.):
    """
    sig should be a scalar here. Note that alpha_ep does not depend on q.
    This function still returns shape (nq,ns) in which all rows are equal
    for consistency with other functions.
    """
    r = np.asarray(r)
    if r.size == 1:
        r = np.tile(r, U.shape[0])

    s = np.asarray(s)
    if s.ndim == 0:
        s = s.reshape(1)

    alphas_exact = alpha_exact(s=s, y=y, U=U, c0=c0, sig=sig) # (1,ns)
    Ur = U.T @ r # (ns,)
    s_Ur = s * Ur # (ns,)
    shift = c0**2 * s_Ur / sig**2 # (ns,)
    alphas = alphas_exact.squeeze() - shift # (ns,)
    
    return np.tile(alphas, (len(q), 1))
