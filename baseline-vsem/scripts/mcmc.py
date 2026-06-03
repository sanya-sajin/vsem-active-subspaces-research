from jax import config
config.update("jax_enable_x64", True)

from pathlib import Path
import json
import csv

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import blackjax

from uncprop.models.vsem.inverse_problem import generate_vsem_inv_prob_rep


# ---------------------------------------------------------------------
# Output location
# ---------------------------------------------------------------------

OUT_DIR = Path("out/active_subspace_final")
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
# Build VSEM posterior
# ---------------------------------------------------------------------

def build_posterior(key, param_names):
    posterior = generate_vsem_inv_prob_rep(
        key=key,
        par_names=param_names,
        n_windows=12,
        n_days_per_window=30,
        observed_variable="lai",
        noise_cov_tril=jnp.eye(12),
    )
    return posterior


# ---------------------------------------------------------------------
# Active subspace computation
# ---------------------------------------------------------------------

def compute_active_subspace(posterior, key, n_samples):
    low, high = posterior.prior.support
    dim = posterior.prior.dim

    z = jr.uniform(key, shape=(n_samples, dim))

    def z_to_theta(z_single):
        return low + z_single * (high - low)

    def logpost_normalized(z_single):
        theta = z_to_theta(z_single)
        return posterior.log_density(jnp.atleast_2d(theta)).squeeze()

    grad_fn = jax.grad(logpost_normalized)
    grads = jax.vmap(grad_fn)(z)

    C = grads.T @ grads / n_samples

    eigvals, eigvecs = jnp.linalg.eigh(C)

    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    theta_samples = jax.vmap(z_to_theta)(z)
    logp = jax.vmap(logpost_normalized)(z)

    return z, theta_samples, logp, grads, C, eigvals, eigvecs


# ---------------------------------------------------------------------
# MCMC
# ---------------------------------------------------------------------

def run_mcmc_full(posterior, key, n_samples=1000, n_warmup=500):
    low, high = posterior.prior.support
    dim = posterior.prior.dim

    def logdensity(z):
        theta = low + z * (high - low)
        return posterior.log_density(jnp.atleast_2d(theta)).squeeze()

    z_init = jnp.ones(dim) * 0.5

    key, warmup_key = jr.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity)
    (state, params), _ = warmup.run(warmup_key, z_init, num_steps=n_warmup)

    def one_step(state, rng_key):
        kernel = blackjax.nuts(logdensity, **params)
        state, info = kernel.step(rng_key, state)
        return state, state.position

    keys = jr.split(key, n_samples)
    _, samples_z = jax.lax.scan(one_step, state, keys)

    samples_theta = samples_z * (high - low) + low

    return samples_z, samples_theta


def run_mcmc_active_subspace(posterior, key, eigvecs, n_components, n_samples=1000, n_warmup=500):
    low, high = posterior.prior.support
    dim = posterior.prior.dim

    W = eigvecs[:, :n_components]
    z_center = jnp.ones(dim) * 0.5

    def y_to_z(y):
        return z_center + W @ y

    def z_to_theta(z):
        return low + z * (high - low)

    def logdensity_reduced(y):
        z = y_to_z(y)
        in_bounds = jnp.all((z > 0.0) & (z < 1.0))
        z_safe = jnp.clip(z, 1e-6, 1.0 - 1e-6)
        theta = z_to_theta(z_safe)
        logp = posterior.log_density(jnp.atleast_2d(theta)).squeeze()
        return jnp.where(in_bounds, logp, -jnp.inf)

    y_init = jnp.zeros(n_components)

    key, warmup_key = jr.split(key)
    warmup = blackjax.window_adaptation(blackjax.nuts, logdensity_reduced)
    (state, params), _ = warmup.run(warmup_key, y_init, num_steps=n_warmup)

    def one_step(state, rng_key):
        kernel = blackjax.nuts(logdensity_reduced, **params)
        state, info = kernel.step(rng_key, state)
        return state, state.position

    keys = jr.split(key, n_samples)
    _, samples_y = jax.lax.scan(one_step, state, keys)

    samples_z = jax.vmap(y_to_z)(samples_y)
    samples_z = jnp.clip(samples_z, 1e-6, 1.0 - 1e-6)
    samples_theta = jax.vmap(z_to_theta)(samples_z)

    return samples_y, samples_z, samples_theta

# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def compare_mcmc_quality(all_results, posterior, key, n_samples=1000, n_warmup=500):
    comparison_results = {}

    for label, res in all_results.items():
        print(f"\nRunning MCMC comparison for {label}")

        eigvecs = res["eigvecs"]
        param_names = res["param_names"]
        dim = len(param_names)
        cumulative_variance = res["cumulative_variance"]

        key, mcmc_key = jr.split(key)
        samples_z_full, samples_theta_full = run_mcmc_full(
            posterior=posterior,
            key=mcmc_key,
            n_samples=n_samples,
            n_warmup=n_warmup,
        )

        as_results = {}
        for n_comp in range(1, min(dim, 5) + 1):
            print(f"  Running active subspace MCMC with {n_comp} components")
            key, mcmc_key = jr.split(key)
            samples_y, samples_z_as, samples_theta_as = run_mcmc_active_subspace(
                posterior=posterior,
                key=mcmc_key,
                eigvecs=eigvecs,
                n_components=n_comp,
                n_samples=n_samples,
                n_warmup=n_warmup,
            )

            full_mean = jnp.mean(samples_theta_full, axis=0)
            full_std = jnp.std(samples_theta_full, axis=0)
            as_mean = jnp.mean(samples_theta_as, axis=0)
            as_std = jnp.std(samples_theta_as, axis=0)

            mean_error = jnp.mean(jnp.abs(
                (full_mean - as_mean) / full_std
            ))
            std_error = jnp.mean(jnp.abs(
                (full_std - as_std) / full_std
            ))

            as_results[n_comp] = {
                "samples_theta": samples_theta_as,
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "variance_explained": float(cumulative_variance[n_comp - 1]),
            }

        comparison_results[label] = {
            "full_samples": samples_theta_full,
            "as_results": as_results,
            "param_names": param_names,
        }

    return comparison_results


# ---------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------

def save_run_outputs(run_dir, param_names, z, theta_samples, logp, grads, C, eigvals, eigvecs):
    run_dir.mkdir(parents=True, exist_ok=True)

    variance_explained = eigvals / jnp.sum(eigvals)
    cumulative_variance = jnp.cumsum(variance_explained)

    jnp.savez(
        run_dir / "active_subspace_results.npz",
        normalized_samples=z,
        physical_samples=theta_samples,
        log_posterior=logp,
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
    plt.figure(figsize=(8, 4))
    plt.bar(param_names, eigvecs[:, 0])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Loading")
    plt.title(f"First Active Direction: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "first_direction_loadings.png", dpi=300)
    plt.close()


def plot_sufficient_summary(label, z, logp, eigvecs, run_dir):
    y1 = z @ eigvecs[:, 0]

    plt.figure(figsize=(6, 4))
    plt.scatter(y1, logp, alpha=0.7)
    plt.xlabel("First active variable")
    plt.ylabel("Log posterior")
    plt.title(f"1D Sufficient Summary: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "sufficient_summary_1d.png", dpi=300)
    plt.close()

    if eigvecs.shape[1] >= 2:
        y2 = z @ eigvecs[:, 1]

        plt.figure(figsize=(6, 5))
        scatter = plt.scatter(y1, y2, c=logp, alpha=0.75)
        plt.xlabel("First active variable")
        plt.ylabel("Second active variable")
        plt.title(f"2D Active Subspace: {label}")
        plt.colorbar(scatter, label="Log posterior")
        plt.tight_layout()
        plt.savefig(run_dir / "active_subspace_2d_logposterior.png", dpi=300)
        plt.close()


def plot_quality_vs_variance(comparison_results, out_dir):
    plt.figure(figsize=(8, 5))

    for label, res in comparison_results.items():
        as_results = res["as_results"]

        variance_explained = [r["variance_explained"] for r in as_results.values()]
        mean_errors = [r["mean_error"] for r in as_results.values()]

        plt.plot(
            variance_explained,
            mean_errors,
            marker="o",
            label=label,
        )

    plt.xlabel("Cumulative variance explained by active subspace")
    plt.ylabel("Mean absolute error compared to full MCMC")
    plt.title("Posterior Approximation Quality vs Variance")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "quality_vs_variance.png", dpi=300)
    plt.close()

def plot_quality_vs_variance_std(comparison_results, out_dir):
    plt.figure(figsize=(8, 5))

    for label, res in comparison_results.items():
        as_results = res["as_results"]

        variance_explained = [r["variance_explained"] for r in as_results.values()]
        std_errors = [r["std_error"] for r in as_results.values()]

        plt.plot(
            variance_explained,
            std_errors,
            marker="o",
            label=label,
        )

    plt.xlabel("Cumulative variance explained by active subspace")
    plt.ylabel("Std error compared to full MCMC")
    plt.title("Posterior Uncertainty Error vs Variance Explained")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "quality_vs_variance_std.png", dpi=300)
    plt.close()

def plot_mcmc_histograms(comparison_results, out_dir):
    for label, res in comparison_results.items():
        full_samples = res["full_samples"]
        param_names = res["param_names"]
        as_results = res["as_results"]

        for n_comp, as_res in as_results.items():
            as_samples = as_res["samples_theta"]
            n_params = len(param_names)

            fig, axs = plt.subplots(1, n_params, figsize=(4 * n_params, 4))
            if n_params == 1:
                axs = [axs]

            for i, param in enumerate(param_names):
                axs[i].hist(full_samples[:, i], bins=30, alpha=0.5, label="Full MCMC", color="blue")
                axs[i].hist(as_samples[:, i], bins=30, alpha=0.5, label=f"AS MCMC ({n_comp} comp)", color="orange")
                axs[i].set_title(param)
                axs[i].set_xlabel("Parameter value")
                axs[i].set_ylabel("Count")
                axs[i].legend(fontsize=7)

            plt.suptitle(f"MCMC Comparison: {label}, {n_comp} components")
            plt.tight_layout()
            plt.savefig(out_dir / f"mcmc_histogram_{n_comp}comp.png", dpi=300)
            plt.close()


def write_mcmc_comparison_csv(comparison_results, out_dir):
    with open(out_dir / "mcmc_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "n_components",
            "variance_explained",
            "mean_error",
            "std_error",
        ])

        for label, res in comparison_results.items():
            for n_comp, as_res in res["as_results"].items():
                writer.writerow([
                    label,
                    n_comp,
                    f"{as_res['variance_explained']:.4f}",
                    f"{as_res['mean_error']:.4f}",
                    f"{as_res['std_error']:.4f}",
                ])


# ---------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------

def write_summary_csv(all_results):
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
# Main
# ---------------------------------------------------------------------

def main():
    base_key = jr.key(0)
    all_results = {}

    for set_idx, (set_name, param_names) in enumerate(PARAMETER_SETS.items()):
        for sample_idx, n_samples in enumerate(SAMPLE_SIZES):
            label = f"{set_name}_n{n_samples}"
            print(f"\nRunning {label}")

            key = jr.fold_in(base_key, set_idx * 1000 + sample_idx)
            key_post, key_as = jr.split(key)

            posterior = build_posterior(key_post, param_names)

            z, theta_samples, logp, grads, C, eigvals, eigvecs = compute_active_subspace(
                posterior=posterior,
                key=key_as,
                n_samples=n_samples,
            )

            variance_explained = eigvals / jnp.sum(eigvals)
            cumulative_variance = jnp.cumsum(variance_explained)

            all_results[label] = {
                "param_names": param_names,
                "z": z,
                "theta_samples": theta_samples,
                "logp": logp,
                "grads": grads,
                "C": C,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "variance_explained": variance_explained,
                "cumulative_variance": cumulative_variance,
                "posterior": posterior,
            }

            run_dir = OUT_DIR / label
            save_run_outputs(
                run_dir=run_dir,
                param_names=param_names,
                z=z,
                theta_samples=theta_samples,
                logp=logp,
                grads=grads,
                C=C,
                eigvals=eigvals,
                eigvecs=eigvecs,
            )

            plot_first_direction_loadings(label, param_names, eigvecs, run_dir)
            plot_sufficient_summary(label, z, logp, eigvecs, run_dir)

            print("Eigenvalues:", eigvals)
            print("Variance explained:", variance_explained)
            print("Cumulative variance:", cumulative_variance)
            print("First active direction:")
            for name, loading in zip(param_names, eigvecs[:, 0]):
                print(f"{name:10s}: {float(loading): .4f}")

    plot_eigenvalue_comparison(all_results)
    plot_variance_explained(all_results)
    write_summary_csv(all_results)

    print("\nRunning MCMC comparisons...")
    key, mcmc_key = jr.split(base_key)

    for set_name, param_names in PARAMETER_SETS.items():
        label = f"{set_name}_n{SAMPLE_SIZES[-1]}"
        res = all_results[label]
        posterior = res["posterior"]

        comparison_results = compare_mcmc_quality(
            all_results={label: res},
            posterior=posterior,
            key=mcmc_key,
            n_samples=3000,
            n_warmup=1500,
        )

        plot_quality_vs_variance(comparison_results, OUT_DIR / label)
        plot_quality_vs_variance_std(comparison_results, OUT_DIR / label) 
        plot_mcmc_histograms(comparison_results, OUT_DIR / label)
        write_mcmc_comparison_csv(comparison_results, OUT_DIR / label)

    print(f"\nDone. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()