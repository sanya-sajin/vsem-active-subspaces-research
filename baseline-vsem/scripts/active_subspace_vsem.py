from jax import config
config.update("jax_enable_x64", True)

from pathlib import Path
import json
import csv

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep


# ---------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------

OUT_DIR = Path("out/active_subspace_loglik_prior")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------

PARAMETER_SETS = {
    "baseline_2d": ["av", "veg_init"],
    "medium_4d": ["av", "veg_init", "gamma", "lue"],
    "expanded_7d": ["av", "veg_init", "gamma", "lue", "kext", "lar", "tauv"],
}

SAMPLE_SIZES = [50, 100, 200]


# ---------------------------------------------------------------------
# Build VSEM inverse problem using Andrew's code
# ---------------------------------------------------------------------

def build_inverse_problem(key, param_names):
    """
    Build one synthetic VSEM Bayesian inverse problem.

    This uses the generate_vsem_inv_prob_rep function.
    It creates an object containing:
    - the prior over selected calibration parameters
    - the likelihood from synthetic LAI observations
    - the posterior log density, available for later MCMC comparisons

    """

    inverse_problem = generate_vsem_inv_prob_rep(
        key=key,
        par_names=param_names,
        n_windows=12,
        n_days_per_window=30,
        observed_variable="lai",
        noise_cov_tril=jnp.eye(12),
    )

    return inverse_problem


# ---------------------------------------------------------------------
# Active subspace computation
# ---------------------------------------------------------------------

def compute_active_subspace(inverse_problem, key, n_samples):
    """
    Compute the active subspace matrix using prior samples and
    log-likelihood gradients.

    We work in normalized coordinates z in [0, 1]^d so that all
    parameters are comparable even if their physical units/ranges differ.

    """

    low, high = inverse_problem.prior.support
    dim = inverse_problem.prior.dim

    # Sample normalized parameter values z in [0, 1]^d.
    z = jr.uniform(key, shape=(n_samples, dim))

    # Convert normalized z values back to real VSEM parameter values.
    def z_to_theta(z_single):
        return low + z_single * (high - low)

    # Define log likelihood as a function of normalized coordinates.
    # This is the active-subspace target function.
    def loglik_normalized(z_single):
        theta = z_to_theta(z_single)
        return inverse_problem.likelihood.log_density(jnp.atleast_2d(theta)).squeeze()

    # Compute gradients of log likelihood with respect to normalized parameters.
    grad_fn = jax.grad(loglik_normalized)
    grads = jax.vmap(grad_fn)(z)

    # Estimate active subspace matrix:
    # C = average of grad log likelihood * grad log likelihood^T
    C = grads.T @ grads / n_samples

    # Eigendecompose C.
    eigvals, eigvecs = jnp.linalg.eigh(C)

    # Sort from largest eigenvalue to smallest.
    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    # Save physical parameter samples too, mainly for inspection/debugging.
    theta_samples = jax.vmap(z_to_theta)(z)

    # Evaluate log likelihood at sampled points for diagnostics/plots.
    loglik = jax.vmap(loglik_normalized)(z)

    return z, theta_samples, loglik, grads, C, eigvals, eigvecs


# ---------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------

def save_run_outputs(run_dir, param_names, z, theta_samples, loglik, grads, C, eigvals, eigvecs):
    """
    Save numerical results for one run.
    """

    run_dir.mkdir(parents=True, exist_ok=True)

    variance_explained = eigvals / jnp.sum(eigvals)
    cumulative_variance = jnp.cumsum(variance_explained)

    jnp.savez(
        run_dir / "active_subspace_results.npz",
        normalized_samples=z,
        physical_samples=theta_samples,
        log_likelihood=loglik,
        gradients_normalized=grads,
        active_subspace_matrix=C,
        eigvals=eigvals,
        eigvecs=eigvecs,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
    )

    with open(run_dir / "param_names.json", "w") as f:
        json.dump(param_names, f, indent=2)


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_eigenvalue_comparison(all_results):
    """
    Plot normalized eigenvalue decay for all runs.

    This helps us compare whether the posterior has low-dimensional
    structure across parameter sets and sample sizes.
    """

    plt.figure(figsize=(8, 5))

    for label, res in all_results.items():
        eigvals = res["eigvals"]
        normalized_eigvals = eigvals / eigvals[0]

        plt.plot(
            range(1, len(normalized_eigvals) + 1),
            normalized_eigvals,
            marker="o",
            label=label,
        )

    plt.yscale("log")
    plt.xlabel("Active direction")
    plt.ylabel("Eigenvalue / largest eigenvalue")
    plt.title("Normalized Eigenvalue Decay")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "comparison_normalized_eigenvalue_scree.png", dpi=300)
    plt.close()


def plot_variance_explained(all_results):
    """
    Plot fraction of gradient energy explained by each direction.
    """

    plt.figure(figsize=(8, 5))

    for label, res in all_results.items():
        variance_explained = res["variance_explained"]

        plt.plot(
            range(1, len(variance_explained) + 1),
            variance_explained,
            marker="o",
            label=label,
        )

    plt.yscale("log")
    plt.xlabel("Active direction")
    plt.ylabel("Fraction of total gradient energy")
    plt.title("Variance Explained by Active Directions")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "variance_explained.png", dpi=300)
    plt.close()


def plot_first_direction_loadings(label, param_names, eigvecs, run_dir):
    """
    Plot parameter loadings for the first active direction.

    This tells us which parameters contribute most to the leading direction.
    """

    plt.figure(figsize=(8, 4))
    plt.bar(param_names, eigvecs[:, 0])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Loading")
    plt.title(f"First Active Direction: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "first_direction_loadings.png", dpi=300)
    plt.close()


def plot_sufficient_summary(label, z, loglik, eigvecs, run_dir):
    """
    Make 1D and 2D active subspace diagnostic plots.

    1D plot:
    log likelihood vs first active variable

    2D plot:
    first active variable vs second active variable, colored by log likelihood
    """

    # First active variable
    y1 = z @ eigvecs[:, 0]

    plt.figure(figsize=(6, 4))
    plt.scatter(y1, loglik, alpha=0.7)
    plt.xlabel("First active variable")
    plt.ylabel("Log likelihood")
    plt.title(f"1D Sufficient Summary: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "sufficient_summary_1d.png", dpi=300)
    plt.close()

    # Second active variable, if it exists
    if eigvecs.shape[1] >= 2:
        y2 = z @ eigvecs[:, 1]

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(y1, y2, c=loglik, alpha=0.75)
        plt.xlabel("First active variable")
        plt.ylabel("Second active variable")
        plt.title(f"2D Active Subspace: {label}")
        plt.colorbar(scatter, label="Log likelihood")
        plt.tight_layout()
        plt.savefig(run_dir / "active_subspace_2d_loglikelihood.png", dpi=300)
        plt.close()


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def write_summary_csv(all_results):
    """
    Save a CSV summary of eigenvalue/variance results.
    """

    with open(OUT_DIR / "summary.csv", "w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "label",
            "dimension",
            "lambda_1",
            "lambda_2",
            "var_1",
            "var_1_plus_2",
        ])

        for label, res in all_results.items():
            eigvals = res["eigvals"]
            variance = res["variance_explained"]

            lambda_1 = float(eigvals[0])
            lambda_2 = float(eigvals[1]) if len(eigvals) > 1 else 0.0
            var_1 = float(variance[0])
            var_1_plus_2 = float(jnp.sum(variance[:2])) if len(variance) > 1 else var_1

            writer.writerow([
                label,
                len(res["param_names"]),
                lambda_1,
                lambda_2,
                var_1,
                var_1_plus_2,
            ])


# ---------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------

def main():
    base_key = jr.key(0)
    all_results = {}

    for set_idx, (set_name, param_names) in enumerate(PARAMETER_SETS.items()):
        for sample_idx, n_samples in enumerate(SAMPLE_SIZES):
            label = f"{set_name}_n{n_samples}"
            print(f"\nRunning {label}")

            # Make a reproducible key for this run.
            key = jr.fold_in(base_key, set_idx * 1000 + sample_idx)
            key_inv, key_as = jr.split(key)

            # Build inverse problem object.
            # This contains prior, likelihood, and posterior.
            # For active-subspace construction, we use prior + likelihood only.
            inverse_problem = build_inverse_problem(key_inv, param_names)

            # Compute active subspace.
            z, theta_samples, loglik, grads, C, eigvals, eigvecs = compute_active_subspace(
                inverse_problem=inverse_problem,
                key=key_as,
                n_samples=n_samples,
            )

            variance_explained = eigvals / jnp.sum(eigvals)
            cumulative_variance = jnp.cumsum(variance_explained)

            # Store results in memory.
            all_results[label] = {
                "param_names": param_names,
                "z": z,
                "theta_samples": theta_samples,
                "loglik": loglik,
                "grads": grads,
                "C": C,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "variance_explained": variance_explained,
                "cumulative_variance": cumulative_variance,
            }

            # Save outputs for this run.
            run_dir = OUT_DIR / label

            save_run_outputs(
                run_dir=run_dir,
                param_names=param_names,
                z=z,
                theta_samples=theta_samples,
                loglik=loglik,
                grads=grads,
                C=C,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )

            plot_first_direction_loadings(label, param_names, eigvecs, run_dir)
            plot_sufficient_summary(label, z, loglik, eigvecs, run_dir)

            # Print results to terminal.
            print("Eigenvalues:")
            print(eigvals)

            print("Variance explained:")
            print(variance_explained)

            print("Cumulative variance:")
            print(cumulative_variance)

            print("First active direction:")
            for name, loading in zip(param_names, eigvecs[:, 0]):
                print(f"{name:10s}: {float(loading): .4f}")

    # Comparison plots across all runs.
    plot_eigenvalue_comparison(all_results)
    plot_variance_explained(all_results)
    write_summary_csv(all_results)

    print(f"\nSaved final outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()