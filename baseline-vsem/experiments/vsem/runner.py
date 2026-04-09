# experiments/vsem/runner.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import vsem_jax as vsem
from inverse_problem import InvProb, VSEMPrior, VSEMLikelihood
from surrogate import VSEMTest


def run_vsem_experiment(rng: np.random.Generator,
                        n_design: int,
                        design_method: str,
                        n_test_grid_1d: int, 
                        n_reps: int,
                        store_pred_clipped: bool = True,
                        out_dir: str = "out",
                        backup_frequency: int = 20,
                        probs: np.ndarray | None = None,
                        write_to_file: bool = True) -> tuple[list[VSEMTest], list, list, list]:
    """ Run VSEM experiment for paper

    The structure of the inverse problem and surrogate model are fixed
    across replications. Replications average over the randomness in the 
    inverse problem setup (sampling synthetic data, ground truth value, etc.)
    and the sampling of the design points. The number of design points is 
    fixed across replications.

    Returns:
        tuple, containing:
            list of VSEMTest objects over the replications
            list of evaluation metrics over the replications
            list of iteration indices that failed
    """
    
    base_out_dir = Path(out_dir)
    test_list = []
    metrics_list = []
    metrics_clip_list = []
    failed_iters = []

    if probs is None:
        probs = np.linspace(0.1, 0.99, 30)

    for i in range(n_reps):
        print(f'Replication {i+1}')

        try:
            inv_prob = get_vsem_inv_prob(rng)
            vsem_test = VSEMTest(inv_prob=inv_prob, 
                                 n_design=n_design, 
                                 n_test_grid_1d=n_test_grid_1d,
                                 store_pred_clipped=store_pred_clipped,
                                 design_method=design_method)
            
            metrics = vsem_test.compute_metrics(pred_type='pred', probs=probs)
            metrics_list.append(metrics)
            test_list.append(vsem_test)

            if store_pred_clipped:
                metrics_clip = vsem_test.compute_metrics(pred_type='pred_clip', probs=probs)
                metrics_clip_list.append(metrics_clip)
        except Exception as e:
            print(f'Iteration {i} failed with error: {e}')
            failed_iters.append(i)
            test_list.append(e)
            metrics_list.append(e)
            if store_pred_clipped:
                metrics_clip_list.append(e)
        finally:
            if write_to_file and ((i % backup_frequency == 0) or (i == n_reps-1)):
                save_results(metrics_list, failed_iters, probs, out_dir=base_out_dir / 'gaussian')
                if store_pred_clipped:
                    save_results(metrics_clip_list, failed_iters, probs, out_dir=base_out_dir / 'clipped')

    print(f'Number of failed iterations: {len(failed_iters)}')
    return test_list, metrics_list, metrics_clip_list, failed_iters


def get_vsem_inv_prob(rng: np.random.Generator) -> InvProb:
    n_days = 365 * 2
    par_names = ["kext", "av"]

    # For exact MCMC
    proposal_cov = np.diag([0.1**2, 0.04**2])

    prior = VSEMPrior(par_names, rng)
    ground_truth = prior.simulate_ground_truth()
    likelihood = VSEMLikelihood(rng, n_days, par_names, ground_truth=ground_truth)
    inv_prob = InvProb(rng, prior, likelihood, proposal_cov=proposal_cov)

    return inv_prob


def save_results(metrics_list, failed_iters, probs, out_dir: Path):
    n_total= len(metrics_list)
    n_failed = len(failed_iters)
    n_reps = n_total - n_failed
    if n_reps < 1:
        return
    else:
        metrics_list = [metric for idx, metric in enumerate(metrics_list) if idx not in failed_iters]

    print(f'Writing {n_reps} reps to output directory: {out_dir}',
          f'Total number: {n_total}; Number failed: {n_failed}')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save failed iters
    np.savez(out_dir / 'failed_iters.npz', failed_iters=failed_iters)

    # Save coverage results
    cover = np.empty((n_reps, 3, len(probs)))
    labels = metrics_list[0]['coverage']['labels']
    for i, metric in enumerate(metrics_list):
        cover[i] = metric['coverage']['cover']
    np.savez(out_dir / 'coverage_results.npz', cover=cover, probs=probs, labels=labels)

    # Save KL and log approx results
    kl_labels = list(metrics_list[0]['kl'].keys())
    log_approx_labels = list(metrics_list[0]['log_approx'].keys())

    kl = {lbl: np.empty(n_reps) for lbl in kl_labels}
    log_approx = {lbl: np.empty(n_reps) for lbl in log_approx_labels}

    for i, metric in enumerate(metrics_list):
        kl_i = metric['kl']
        for lbl,val in kl_i.items():
            kl[lbl][i] = kl_i[lbl]

        log_approx_i = metric['log_approx']
        for lbl,val in log_approx_i.items():
            log_approx[lbl][i] = log_approx_i[lbl]

    np.savez(out_dir / 'kl_results.npz', **kl)
    np.savez(out_dir / 'log_approx_results.npz', **log_approx)


if __name__ == '__main__':
    import numpy as np
    from jax import config
    from pathlib import Path
    config.update("jax_enable_x64", True)

    # Settings
    rng = np.random.default_rng(12421)
    n_design = 15
    design_method = 'lhc'
    n_test_grid_1d = 50
    n_reps = 100
    store_pred_clipped = True
    base_out_dir = 'out'

    # Define output directory
    out_dir = Path(base_out_dir) / f'{design_method}_{n_design}'

    # Run experiment
    results = run_vsem_experiment(rng=rng, 
                                  n_design=n_design,
                                  design_method=design_method,
                                  n_reps=n_reps, 
                                  n_test_grid_1d=n_test_grid_1d,
                                  store_pred_clipped=store_pred_clipped, 
                                  out_dir=out_dir)