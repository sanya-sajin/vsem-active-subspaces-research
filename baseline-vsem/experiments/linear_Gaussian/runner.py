# experiments/linear_Gaussian/run_coverage_test.py
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from Gaussian import Gaussian
from LinGaussTest import LinGaussInvProb, LinGaussTest

def run_coverage_test(rng, n_reps, m0, C0, Sig, G, Q_true, Q=None, 
                      include_mcmc=True, n_mcmc=100_000,
                      backup_frequency=10, out_dir='out'):
    """
    The quantities (m0, C0, Sig, G, Q) are fixed throughout
    the whole test. This implies the structure of the inverse problem
        y|u ~ N(Gu, Sig)
        u ~ N(m0, C0)
    is fixed. The quantities (u_true, y, r) are randomized over different
    replications of the experiment. In particular, the ground truth
    parameter is sampled as u_true ~ N(m0, C0) and the synthetic observed
    data is then generated from the model y ~ N(Gu_true, Sig). The
    surrogate bias is sampled as r ~ N(0, Q_true) and the surrogate for
    that replication is then defined as G_star(u) ~ N(Gu + r, Q) with
    r fixed. If Q = Q_true then the surrogate is "calibrated", in the
    sense that Q correctly quantifies the uncertainty in the surrogate
    bias r.

    The experiment is run using a well-specified model (i.e., the
    observations are generated from the true likelihood). If `include_mcmc=True`
    then MCMC algorithms are also run to evaluate the MCMC-based EP approximation.
    """

    n = G.shape[0]
    d = G.shape[1]
    tests = []

    # Default to well-calibrated surrogate
    if Q is None:
        Q = Q_true

    # Quantiles to compute for coverage metrics
    probs = np.append(np.arange(0.1, 1.0, step=0.1), 0.99)
    n_probs = len(probs)

    cover = {
        "ep_cover_univariate" : np.empty((n_reps, n_probs, d)),
        "eup_cover_univariate" : np.empty((n_reps, n_probs, d)),
        "mean_cover_joint" : np.empty((n_reps, n_probs)),
        "ep_cover_joint" : np.empty((n_reps, n_probs)),
        "eup_cover_joint" : np.empty((n_reps, n_probs))
    }

    dists = {
        "mean_kl" : np.empty(n_reps),
        "ep_kl" : np.empty(n_reps),
        "eup_kl" : np.empty(n_reps),
        "mean_w2" : np.empty(n_reps),
        "ep_w2" : np.empty(n_reps),
        "eup_w2" : np.empty(n_reps),
        "ep_expected_kl" : np.empty(n_reps),
        "eup_expected_kl" : np.empty(n_reps),
        "ep_eup_kl": np.empty(n_reps),
        "ep_eup_w2": np.empty(n_reps)
    }

    if include_mcmc:
        mcmc = {
            "ep_rkpcn99_kl": np.empty(n_reps),
            "ep_rkpcn99_w2": np.empty(n_reps),
            "ep_rkpcn95_kl": np.empty(n_reps),
            "ep_rkpcn95_w2": np.empty(n_reps),
            "ep_rkpcn90_kl": np.empty(n_reps),
            "ep_rkpcn90_w2": np.empty(n_reps),
            "ep_rk_kl": np.empty(n_reps),
            "ep_rk_w2": np.empty(n_reps)
        }
    else:
        mcmc = None

    failed_iters = []

    for i in range(n_reps):
        print(f'Replication {i+1}')

        try:
            inv_prob = LinGaussInvProb(rng, G, m0, C0, Sig)
            r = Gaussian(cov=Q_true, rng=rng).sample()

            test = LinGaussTest(inv_prob, Q, r=r)
            tests.append(test)
            res = test.calc_coverage(probs=probs)

            # coverage
            cover["ep_cover_univariate"][i,:,:] = res["ep"]
            cover["eup_cover_univariate"][i,:,:] = res["eup"]
            cover["mean_cover_joint"][i,:] = test.post.compute_credible_ellipsoid_coverage(test.mean_post)
            cover["eup_cover_joint"][i,:] = test.post.compute_credible_ellipsoid_coverage(test.eup_post)
            cover["ep_cover_joint"][i,:] = test.post.compute_credible_ellipsoid_coverage(test.ep_post)

            # distances
            dists["ep_kl"][i] = test.post.kl(test.ep_post)
            dists["eup_kl"][i] = test.post.kl(test.eup_post)
            dists["mean_kl"][i] = test.post.kl(test.mean_post)
            dists["mean_w2"][i] = test.post.kl(test.mean_post)
            dists["ep_w2"][i] = test.post.wasserstein(test.ep_post)
            dists["eup_w2"][i] = test.post.wasserstein(test.eup_post)
            dists["ep_expected_kl"][i], dists["eup_expected_kl"][i] = test.estimate_expected_kl()
            dists["ep_eup_kl"][i] = test.ep_post.kl(test.eup_post)
            dists["ep_eup_w2"][i] = test.ep_post.wasserstein(test.eup_post)

            # mcmc
            if include_mcmc:
                mcmc_results = get_mcmc_results(test, n_samp=n_mcmc)
                mcmc['ep_rkpcn99_kl'][i] = mcmc_results['ep_rkpcn99_kl']
                mcmc['ep_rkpcn99_w2'][i] = mcmc_results['ep_rkpcn99_w2']
                mcmc['ep_rkpcn95_kl'][i] = mcmc_results['ep_rkpcn95_kl']
                mcmc['ep_rkpcn95_w2'][i] = mcmc_results['ep_rkpcn95_w2']
                mcmc['ep_rkpcn90_kl'][i] = mcmc_results['ep_rkpcn90_kl']
                mcmc['ep_rkpcn90_w2'][i] = mcmc_results['ep_rkpcn90_w2']
                mcmc['ep_rk_kl'][i] = mcmc_results['ep_rk_kl']
                mcmc['ep_rk_w2'][i] = mcmc_results['ep_rk_w2']
        except Exception as e:
            print(f'Iteration {i} failed with error: {e}')
            failed_iters.append(i)
            tests.append(e)
        finally:
            if i % backup_frequency == 0:
                print(f'Writing to output directory: {out_dir}')
                save_results(cover, dists, probs, failed_iters, mcmc=mcmc, out_dir=out_dir)

    print(f'Writing final results to directory: {out_dir}')
    save_results(cover, dists, probs, failed_iters, mcmc=mcmc, out_dir=out_dir)
    out = {'cover': cover, 'dists': dists, 'mcmc': mcmc}
    return tests, out, probs


def save_results(cover, dists, probs, failed_iters, mcmc=None, out_dir='out'):
    out_dir = Path(out_dir)

    cover['probs'] = probs
    np.savez(out_dir / 'coverage_results.npz', **cover)
    np.savez(out_dir / 'distance_metric_results.npz', **dists)
    np.savez(out_dir / 'failed_iters.npz', failed_iters=failed_iters)

    if mcmc is not None:
        np.savez(out_dir / 'mcmc_results.npz', **mcmc)


def read_results(out_dir='out'):
    out_dir = Path(out_dir)
    cover_path = out_dir / 'coverage_results.npz'
    dist_path = out_dir / 'distance_metric_results.npz'
    failed_iters_path = out_dir / 'failed_iters.npz'
    mcmc_path = out_dir / 'mcmc_results.npz'

    cover = np.load(cover_path)
    dists = np.load(dist_path)
    failed_iters = np.load(failed_iters_path)
    if mcmc_path.exists():
        mcmc = np.load(mcmc_path)
    else:
        mcmc = None

    return {'cover': cover, 
            'dists': dists, 
            'failed_iters': failed_iters,
            'mcmc': mcmc}


def get_mcmc_results(test, n_samp=100000):
    test.reset_samplers()

    # rkpcn (rho = .99)
    test.samplers['rk-pcn'] = test.get_rk_pcn_sampler(u_prop_scale=0.1, pcn_cor=0.99)
    samp, _ = test.get_sample_list(n_samp=n_samp, include=['rk-pcn'])
    rkpcn99_kl, rkpcn99_w2 = _calc_ep_mcmc_metrics(test, samp[0])

    # rkpcn (rho = .95)
    test.samplers['rk-pcn'] = test.get_rk_pcn_sampler(u_prop_scale=0.1, pcn_cor=0.95)
    samp, _ = test.get_sample_list(n_samp=n_samp, include=['rk-pcn'])
    rkpcn95_kl, rkpcn95_w2 = _calc_ep_mcmc_metrics(test, samp[0])

    # rkpcn (rho = .90)
    test.samplers['rk-pcn'] = test.get_rk_pcn_sampler(u_prop_scale=0.1, pcn_cor=0.90)
    samp, _ = test.get_sample_list(n_samp=n_samp, include=['rk-pcn'])
    rkpcn90_kl, rkpcn90_w2 = _calc_ep_mcmc_metrics(test, samp[0])

    # rk
    test.samplers['rk'] = test.get_rk_sampler(u_prop_scale=0.1)
    samp, _ = test.get_sample_list(n_samp=n_samp, include=['rk'])
    rk_kl, rk_w2 = _calc_ep_mcmc_metrics(test, samp[0])

    return {
        "ep_rkpcn99_kl": rkpcn99_kl,
        "ep_rkpcn99_w2": rkpcn99_w2,
        "ep_rkpcn95_kl": rkpcn95_kl,
        "ep_rkpcn95_w2": rkpcn95_w2,
        "ep_rkpcn90_kl": rkpcn90_kl,
        "ep_rkpcn90_w2": rkpcn90_w2,
        "ep_rk_kl": rk_kl,
        "ep_rk_w2": rk_w2
    }


def _calc_ep_mcmc_metrics(test, samp):
    # Fit Gaussian approximation to samples
    m = np.mean(samp, axis=0)
    C = np.cov(samp.T)
    approx = Gaussian(m, C)

    kl = test.ep_post.kl(approx)
    w2 = test.ep_post.wasserstein(approx)

    return (kl, w2)


def plot_coverage(coverage_list, probs, labels, colors,  
                  q_min=0.05, q_max=0.95, alpha=0.3, ax=None):
    """
    The first two arguments are shape (n_reps, n_probs), giving the nominal
    coverage for each replicate at each coverage probability level.
    """

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    n_plots = len(coverage_list)
    medians = [np.median(cover, axis=0) for cover in coverage_list]
    quantiles = [np.quantile(cover, q=[q_min, q_max], axis=0) for cover in coverage_list]

    for i in range(n_plots):
        m = medians[i]
        q = quantiles[i]
        lbl = labels[i]
        ax.fill_between(probs, q[0,:], q[1,:], alpha=alpha, color=colors[lbl], label=lbl)
        
    ax.set_xlabel("Nominal Coverage")
    ax.set_ylabel("Actual Coverage")

    # Add line y = x
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    y = x
    ax.plot(x, y, color=colors['aux'], linestyle="--")
    ax.legend()
    plt.close(fig)

    return fig, ax


def plot_coverage_by_dim(ep_coverage, eup_coverage, probs, q_min=0.05,
                         q_max=0.95, nrows=1, ncols=None, figsize=(5,4)):

        d = ep_coverage.shape[2]
        if ncols is None:
            ncols = int(np.ceil(d / nrows))

        ep_m = np.median(ep_coverage, axis=0)
        eup_m = np.median(eup_coverage, axis=0)
        ep_q = np.quantile(ep_coverage, q=[q_min, q_max], axis=0)
        eup_q = np.quantile(eup_coverage, q=[q_min, q_max], axis=0)

        fig, axs = plt.subplots(nrows, ncols, figsize=(figsize[0]*ncols, figsize[1]*nrows))
        axs = np.array(axs).reshape(-1)
        for j in range(d):
            ax = axs[j]

            ax.fill_between(probs, ep_q[0,:,j], ep_q[1,:,j], color='blue', alpha=0.3, label='ep')
            ax.fill_between(probs, eup_q[0,:,j], eup_q[1,:,j], color='red', alpha=0.3, label='eup')
            ax.plot(probs, ep_m[:,j], color='blue', label='ep')
            ax.plot(probs, eup_m[:,j], color='red', label='eup')
            ax.set_title(j)
            ax.set_xlabel("Nominal Coverage")
            ax.set_ylabel("Actual Coverage")

            # Add line y = x
            xmin, xmax = ax.get_xlim()
            x = np.linspace(xmin, xmax, 100)
            y = x
            ax.plot(x, y, color="red", linestyle="--")
            ax.legend()

        # Hide unused axes and close figure.
        for k in range(d, nrows*ncols):
            fig.delaxes(axs[k])
        plt.close(fig)
        return fig


if __name__ == "__main__":
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt

    from inverse_problem_setup import make_inverse_problem, summarize_setup, get_forward_model
    from Gaussian import Gaussian
    from LinGaussTest import LinGaussInvProb, LinGaussTest

    rng = np.random.default_rng(532124)

    #
    # Setup
    #

    # inverse problem
    d = 100
    ker_length = 21
    ker_lengthscale = 20
    sig = 0.2
    s = 4 # every sth index is observed

    # misspecified model
    ker_lengthscale_mispec = 2

    # surrogate
    Q_scale_factor = 1.0

    # experiment
    n_reps = 100

    # Well-specified inverse problem
    inv_prob_info = make_inverse_problem(rng=rng, 
                                         d=d, 
                                         noise_sd=sig, 
                                         ker_length=ker_length, 
                                         ker_lengthscale=ker_lengthscale,
                                         s=s)
    inv_prob, g_conv_true, grid, idx_obs = inv_prob_info

    # Calibrated surrogate model
    Q = Q_scale_factor * inv_prob.G @ inv_prob.prior.cov @ inv_prob.G.T
    test = LinGaussTest(inv_prob, Q)

    # Run test
    tests, res, probs = run_coverage_test(rng, n_reps, m0=inv_prob.prior.mean, 
                                          C0=inv_prob.prior.cov, Sig=inv_prob.noise.cov, 
                                          G=inv_prob.G, Q_true=Q, Q=Q, include_mcmc=True)

