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

SAMPLE_SIZES = [200]


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
# MCMC helpers
# ---------------------------------------------------------------------

def compute_rhat(chains):
    """
    Compute basic Gelman-Rubin R-hat for multiple MCMC chains.

    chains shape:
        (n_chains, n_samples, n_params)

    returns:
        rhat shape (n_params,)
    """
    chains = jnp.asarray(chains)
    n_chains, n_samples, _ = chains.shape

    if n_chains < 2:
        raise ValueError("R-hat requires at least 2 chains.")

    chain_means = jnp.mean(chains, axis=1)
    grand_mean = jnp.mean(chain_means, axis=0)

    # Between-chain variance
    B = n_samples * jnp.sum((chain_means - grand_mean) ** 2, axis=0) / (n_chains - 1)

    # Within-chain variance
    chain_vars = jnp.var(chains, axis=1, ddof=1)
    W = jnp.mean(chain_vars, axis=0)

    # Pooled marginal posterior variance estimate
    var_hat = ((n_samples - 1) / n_samples) * W + (B / n_samples)

    return jnp.sqrt(var_hat / W)


# ---------------------------------------------------------------------
# Full MCMC
# ---------------------------------------------------------------------

def run_mcmc_full(posterior, key, n_samples=1000, n_warmup=500):
    """
    Run one full-parameter MCMC chain.

    This samples all original parameters in normalized z-space and then
    converts samples back to physical theta-space.
    """
    low, high = posterior.prior.support
    dim = posterior.prior.dim

    def logdensity(z):
        theta = low + z * (high - low)
        return posterior.log_density(jnp.atleast_2d(theta)).squeeze()

    # Random initial position inside prior box, away from exact boundaries.
    key, init_key = jr.split(key)
    z_init = jr.uniform(init_key, shape=(dim,), minval=0.05, maxval=0.95)

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


def run_mcmc_full_chains(posterior, key, n_chains=4, n_samples=1000, n_warmup=500):
    """
    Run multiple full-parameter MCMC chains and compute R-hat.

    returns:
        samples_z_chains:     (n_chains, n_samples, n_params)
        samples_theta_chains: (n_chains, n_samples, n_params)
        rhat:                 (n_params,)
    """
    keys = jr.split(key, n_chains)

    z_chains = []
    theta_chains = []

    for i in range(n_chains):
        print(f"    Full chain {i + 1}/{n_chains}")
        z_i, theta_i = run_mcmc_full(
            posterior=posterior,
            key=keys[i],
            n_samples=n_samples,
            n_warmup=n_warmup,
        )
        z_chains.append(z_i)
        theta_chains.append(theta_i)

    samples_z_chains = jnp.stack(z_chains)
    samples_theta_chains = jnp.stack(theta_chains)
    rhat = compute_rhat(samples_theta_chains)

    return samples_z_chains, samples_theta_chains, rhat


# ---------------------------------------------------------------------
# Active-subspace MCMC
# ---------------------------------------------------------------------

def run_mcmc_active_subspace(posterior, key, eigvecs, n_components, n_samples=1000, n_warmup=500):
    """
    Run one active-subspace MCMC chain.

    MCMC samples y-space, then maps y -> z -> theta.
    """
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


def run_mcmc_active_subspace_chains(
    posterior,
    key,
    eigvecs,
    n_components,
    n_chains=4,
    n_samples=1000,
    n_warmup=500,
):
    """
    Run multiple active-subspace MCMC chains and compute R-hat.

    returns:
        samples_y_chains:     (n_chains, n_samples, n_components)
        samples_z_chains:     (n_chains, n_samples, n_params)
        samples_theta_chains: (n_chains, n_samples, n_params)
        rhat:                 (n_params,)
    """
    keys = jr.split(key, n_chains)

    y_chains = []
    z_chains = []
    theta_chains = []

    for i in range(n_chains):
        print(f"    AS chain {i + 1}/{n_chains}")
        y_i, z_i, theta_i = run_mcmc_active_subspace(
            posterior=posterior,
            key=keys[i],
            eigvecs=eigvecs,
            n_components=n_components,
            n_samples=n_samples,
            n_warmup=n_warmup,
        )
        y_chains.append(y_i)
        z_chains.append(z_i)
        theta_chains.append(theta_i)

    samples_y_chains = jnp.stack(y_chains)
    samples_z_chains = jnp.stack(z_chains)
    samples_theta_chains = jnp.stack(theta_chains)
    rhat = compute_rhat(samples_theta_chains)

    return samples_y_chains, samples_z_chains, samples_theta_chains, rhat


# ---------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------

def compare_mcmc_quality(
    all_results,
    posterior,
    key,
    n_samples=1000,
    n_warmup=500,
    n_chains=4,
):
    comparison_results = {}

    for label, res in all_results.items():
        print(f"\nRunning MCMC comparison for {label}")

        eigvecs = res["eigvecs"]
        param_names = res["param_names"]
        dim = len(param_names)
        cumulative_variance = res["cumulative_variance"]

        # Full MCMC chains: these are both the reference samples and the full R-hat diagnostic.
        key, full_key = jr.split(key)
        samples_z_full_chains, samples_theta_full_chains, full_rhat = run_mcmc_full_chains(
            posterior=posterior,
            key=full_key,
            n_chains=n_chains,
            n_samples=n_samples,
            n_warmup=n_warmup,
        )

        print(f"  Full R-hat: {full_rhat}")
        print(f"  Full Max R-hat: {float(jnp.max(full_rhat)):.4f}")

        # Flatten full chains for mean/std summaries and histogram overlays.
        samples_theta_full = samples_theta_full_chains.reshape(
            -1, samples_theta_full_chains.shape[-1]
        )

        full_mean = jnp.mean(samples_theta_full, axis=0)
        full_std = jnp.std(samples_theta_full, axis=0)

        as_results = {}

        for n_comp in range(1, min(dim, 3) + 1):
            print(f"  Running active subspace MCMC with {n_comp} components")

            key, as_key = jr.split(key)
            samples_y_chains, samples_z_as_chains, samples_theta_as_chains, as_rhat = (
                run_mcmc_active_subspace_chains(
                    posterior=posterior,
                    key=as_key,
                    eigvecs=eigvecs,
                    n_components=n_comp,
                    n_chains=n_chains,
                    n_samples=n_samples,
                    n_warmup=n_warmup,
                )
            )

            print(f"  AS {n_comp} R-hat: {as_rhat}")
            print(f"  AS {n_comp} Max R-hat: {float(jnp.max(as_rhat)):.4f}")

            # Flatten AS chains for mean/std summaries and histogram overlays.
            samples_theta_as = samples_theta_as_chains.reshape(
                -1, samples_theta_as_chains.shape[-1]
            )

            as_mean = jnp.mean(samples_theta_as, axis=0)
            as_std = jnp.std(samples_theta_as, axis=0)

            mean_error = jnp.mean(jnp.abs((full_mean - as_mean) / full_std))
            std_error = jnp.mean(jnp.abs((full_std - as_std) / full_std))

            as_results[n_comp] = {
                "samples_theta": samples_theta_as,
                "samples_theta_chains": samples_theta_as_chains,
                "rhat": as_rhat,
                "max_rhat": float(jnp.max(as_rhat)),
                "mean_error": float(mean_error),
                "std_error": float(std_error),
                "variance_explained": float(cumulative_variance[n_comp - 1]),
            }

        comparison_results[label] = {
            "full_samples": samples_theta_full,
            "full_samples_chains": samples_theta_full_chains,
            "full_rhat": full_rhat,
            "full_max_rhat": float(jnp.max(full_rhat)),
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


def plot_error_vs_components(comparison_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, res in comparison_results.items():
        as_results = res["as_results"]
        n_components = list(as_results.keys())
        mean_errors = [r["mean_error"] for r in as_results.values()]

        plt.plot(
            n_components,
            mean_errors,
            marker="o",
            label=label,
        )

    plt.xlabel("Number of active components")
    plt.ylabel("Mean absolute standardized mean error")
    plt.title("Posterior Mean Error vs Active Subspace Dimension")
    plt.xticks(n_components)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "mean_error_vs_components.png", dpi=300)
    plt.close()


def plot_std_error_vs_components(comparison_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    for label, res in comparison_results.items():
        as_results = res["as_results"]
        n_components = list(as_results.keys())
        std_errors = [r["std_error"] for r in as_results.values()]

        plt.plot(
            n_components,
            std_errors,
            marker="o",
            label=label,
        )

    plt.xlabel("Number of active components")
    plt.ylabel("Mean absolute relative standard deviation error")
    plt.title("Posterior Standard Deviation Error vs Active Subspace Dimension")
    plt.xticks(n_components)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "std_error_vs_components.png", dpi=300)
    plt.close()


def plot_mcmc_histograms(comparison_results, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

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
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "mcmc_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "label",
            "n_components",
            "variance_explained",
            "mean_error",
            "std_error",
            "full_max_rhat",
            "as_max_rhat",
        ])

        for label, res in comparison_results.items():
            for n_comp, as_res in res["as_results"].items():
                writer.writerow([
                    label,
                    n_comp,
                    f"{as_res['variance_explained']:.4f}",
                    f"{as_res['mean_error']:.4f}",
                    f"{as_res['std_error']:.4f}",
                    f"{res['full_max_rhat']:.4f}",
                    f"{as_res['max_rhat']:.4f}",
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
            n_samples=2000,
            n_warmup=1000,
            n_chains=4,
        )

        plot_error_vs_components(comparison_results, OUT_DIR / label)
        plot_std_error_vs_components(comparison_results, OUT_DIR / label)
        plot_mcmc_histograms(comparison_results, OUT_DIR / label)
        write_mcmc_comparison_csv(comparison_results, OUT_DIR / label)

    print(f"\nDone. Outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
