# Initial imports
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
import numpy as np
import arviz as az

from uncprop.utils.distribution import _gaussian_log_density_tril
from uncprop.models.vsem import vsemjax as vsem
from uncprop.models.vsem.inverse_problem import VSEM_DEFAULT_PARAMS, VSEM_DEFAULT_PRIORS, define_vsem_observation_operator

from numpyro.distributions import Normal


# Pulls the function that uses VSEM and returns Posterior object (scoring system)


# Output location
OUT_DIR = Path("out/active_subspace_final")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Experiment settings
PARAMETER_SETS = {
    "baseline_2d": ["av", "veg_init"],
    "medium_4d": ["av", "veg_init", "gamma", "lue"],
    "expanded_7d": ["av", "veg_init", "gamma", "lue", "kext", "lar", "tauv"],
}
SAMPLE_SIZES = [200]



# Uncontrained prior
class UnconstrainedPrior:
    # Picks random sample from unconstrained space, transforms to constrained space, and scores log posterior
    def __init__(self, low, high, sigma = 1.0):
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self.sigma = sigma
    # Stores bounds and sigma (1)

    def dim(self):
        return self.low.shape[0]
    # Dimension is the number of parameters

    def u_to_theta(self, u):
        return self.low + (self.high - self.low) / (1 + jnp.exp(-u))
    # Inverse transform: theta = a + (b - a)/{1 + exp(-u)}
    # Possible for other distributions?

    def sample(self, key, n = 1):
        return self.sigma * jr.normal(key, shape = (n, self.dim()))
    # Sample from unconstrained space (normal)
    # Can alter for other distributions

    def log_density(self, u):
        return jnp.sum(Normal(0.0, self.sigma).log_prob(u), axis = -1)
    # Normal log-density of u, summed across parameters
    # Uses Numpyro's Normal
    # Can alter for other distributions


# VSEM likelihood
class VSEMLikelihood:
    # Computes log-likelihood of theta using VSEM forward-model
    def __init__(self, forward_model, observation, noise_cov_tril):
        self.forward_model = forward_model
        self.observation = observation
        self.noise_cov_tril = noise_cov_tril

    def log_density(self, theta):
        predicted = self.forward_model(theta)
        return _gaussian_log_density_tril(x = self.observation, m = predicted, L = self.noise_cov_tril)
    # Returns likelihood score for theta using VSEM forward model
    # Forward model returns predicted observation
    # Log-likelihood computed


# Build unconstrained posterior
def build_unconstrained_posterior(key, param_names, n_windows, n_days_per_window, observed_variable, noise_cov_tril, sigma=1.0):
    # Replaces Sanya's build_posterior function
    # Sets up VSEM inverse problem
    key, key_true, key_driver, key_obs = jr.split(key, 4)
    n_days = n_windows * n_days_per_window
    # Setting up VSEM with key

    low = jnp.array([VSEM_DEFAULT_PRIORS[p].low for p in param_names])
    high = jnp.array([VSEM_DEFAULT_PRIORS[p].high for p in param_names])
    prior = UnconstrainedPrior(low, high, sigma=sigma)
    # Re-uses Andrew's dictionary
    # Initializes UncontrainedPrior with bounds

    u_true = prior.sample(key_true, n = 1)[0]
    theta_true = prior.u_to_theta(u_true)
    # Builds "true" sample with key
    # Converts to theta space with inverse transform

    defaults = VSEM_DEFAULT_PARAMS.copy()
    for p, v in zip(param_names, theta_true):
        defaults[p] = v
    # Copies Andrew's parameter dictionary

    time_steps, driver = vsem.simulate_vsem_driver(key_driver, n_days)
    # Weather settings (driver)
    obs_op_info = define_vsem_observation_operator(num_days = n_days, window_len = n_days_per_window, vsem_output_var = observed_variable)
    forward_model_raw = vsem.build_batch_forward_model(driver = driver, par_names = param_names, default_param_dict=defaults)
    # Runs VSEM under driver settings
    # Setting up VSEM

    def forward_model(theta):
        vsem_output = forward_model_raw(jnp.atleast_2d(theta))
        return obs_op_info.observation_operator(vsem_output).ravel()
    # Model that takes theta, runs VSEM, and returns predicted observation

    true_observable = forward_model(theta_true)
    # Uses "true" theta to get "true" data
    noise = noise_cov_tril @ jr.normal(key_obs, shape = (noise_cov_tril.shape[0],))
    # Noise
    # Something about fitting the noise
    observation = true_observable + noise
    # Creates "true" observation by running forward model and adding noise

    likelihood = VSEMLikelihood(forward_model, observation, noise_cov_tril)
    # Initializes likelihood with forward model

    return prior, likelihood


# Compute active subspace
def compute_active_subspace(prior, likelihood, key, n_samples):

    u = prior.sample(key, n = n_samples)
    # Creates u (table sized 200 x dim) containing random samples from unconstrained space

    def logpost_unconstrained(u_single):
        theta = prior.u_to_theta(u_single)
        # Converts single u to theta using inverse transform
        log_prior = prior.log_density(u_single)
        # Scores log prior at u_single
        log_likelihood = likelihood.log_density(jnp.atleast_2d(theta)).squeeze()
        # Scores log likelihood at theta
        return log_prior + log_likelihood
        # ADD THEM TOGETHER TO GET POSTERIOR

    grad_fn = jax.grad(logpost_unconstrained)
    grads = jax.vmap(grad_fn)(u)
    # Creates grad_fn (a function that computes gradient and individual u values)

    C = grads.T @ grads / n_samples
    # Creates C (AS matrix that averages gradients)

    eigvals, eigvecs = jnp.linalg.eigh(C)
    # Creates eigenvalues and eigenvectors of C

    order = jnp.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    # Sorts by eigenvalue

    theta_samples = jax.vmap(prior.u_to_theta)(u)
    logp = jax.vmap(logpost_unconstrained)(u)
    # Converts u points to theta points, scores log posterior at each point

    return u, theta_samples, logp, grads, C, eigvals, eigvecs


# Saving outputs
def save_run_outputs(run_dir, param_names, u, theta_samples, logp, grads, C, eigvals, eigvecs):
    run_dir.mkdir(parents = True, exist_ok = True)

    variance_explained = eigvals / jnp.sum(eigvals)
    # Converts eigenvalues to fractions
    cumulative_variance = jnp.cumsum(variance_explained)

    jnp.savez(
        run_dir / "active_subspace_results.npz",
        normalized_samples = u,
        physical_samples = theta_samples,
        log_posterior = logp,
        gradients_normalized = grads,
        active_subspace_matrix = C,
        eigvals = eigvals,
        eigvecs = eigvecs,
        variance_explained = variance_explained,
        cumulative_variance = cumulative_variance,
    )
    # Saves things :)

    with open(run_dir / "param_names.json", "w") as f:
        json.dump(param_names, f, indent=2)


# Plotting
def plot_first_direction_loadings(label, param_names, eigvecs, run_dir):
    plt.figure(figsize = (8, 4))
    plt.bar(param_names, eigvecs[:, 0])
    plt.xticks(rotation=45, ha = "right")
    plt.ylabel("Loading")
    plt.title(f"First Active Direction: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "first_direction_loadings.png", dpi=300)
    plt.close()
    # Bar chart of how much each parameter contributes to the first active direction


def plot_sufficient_summary(label, u, logp, eigvecs, run_dir):
    y1 = u @ eigvecs[:, 0]
    # Dot product of u with first eigenvector to get 1D sufficient summary
    # Projection of samples onto first active direction

    plt.figure(figsize = (6, 4))
    plt.scatter(y1, logp, alpha = 0.7)
    plt.xlabel("First active variable")
    plt.ylabel("Log posterior")
    plt.title(f"1D Sufficient Summary: {label}")
    plt.tight_layout()
    plt.savefig(run_dir / "sufficient_summary_1d.png", dpi = 300)
    plt.close()

    if eigvecs.shape[1] >= 2:
        y2 = u @ eigvecs[:, 1]

        plt.figure(figsize = (6, 5))
        scatter = plt.scatter(y1, y2, c=logp, alpha = 0.75)
        plt.xlabel("First active variable")
        plt.ylabel("Second active variable")
        plt.title(f"2D Active Subspace: {label}")
        plt.colorbar(scatter, label="Log posterior")
        plt.tight_layout()
        plt.savefig(run_dir / "active_subspace_2d_logposterior.png", dpi = 300)
        plt.close()
        # Scatter plot of samples projected onto first two active directions
        # Log posterior heat map


def plot_eigenvalue_comparison(all_results):
    # Compares eigenvalue decay by plotting normalized eigenvalues for all runs
    # First active direction has value 1
    plt.figure(figsize = (8, 5))

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
    plt.savefig(OUT_DIR / "comparison_normalized_eigenvalue_scree.png", dpi = 300)
    plt.close()
    # Plotting stuff


def plot_variance_explained(all_results):
    # Same thing but with variance explained instead of eigenvalues
    # Eigenvalues sum to 1
    plt.figure(figsize = (8, 5))

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
    plt.legend(fontsize = 8)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "variance_explained.png", dpi = 300)
    plt.close()
    # Plotting stuff
    # Do I really need to understand this
    # No, I don't think so


def write_summary_csv(all_results):
    # It's called write summary obviously it writes the summary
    # Variance covered by first two active directions
    with open(OUT_DIR / "summary.csv", "w", newline = "") as f:
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

# Main
def main():
    base_key = jr.key(0)
    all_results = {}

    for set_idx, (set_name, param_names) in enumerate(PARAMETER_SETS.items()):
        for sample_idx, n_samples in enumerate(SAMPLE_SIZES):
            label = f"{set_name}_n{n_samples}"
            print(f"\nRunning {label}")
            # Looping

            key = jr.fold_in(base_key, set_idx * 1000 + sample_idx)
            key_post, key_as = jr.split(key)
            # Key

            prior, likelihood = build_unconstrained_posterior(
                key = key_post,
                param_names = param_names,
                n_windows = 12,
                n_days_per_window = 30,
                observed_variable = "lai",
                noise_cov_tril = jnp.eye(12),
            )
            # Building unconstrained posterior

            u, theta_samples, logp, grads, C, eigvals, eigvecs = compute_active_subspace(
                prior = prior,
                likelihood = likelihood,
                key = key_as,
                n_samples = n_samples,
            )
            # Running active subspace

            variance_explained = eigvals / jnp.sum(eigvals)
            cumulative_variance = jnp.cumsum(variance_explained)
            # Converting eigenvalues to fractions

            all_results[label] = {
                "param_names": param_names,
                "u": u,
                "theta_samples": theta_samples,
                "logp": logp,
                "grads": grads,
                "C": C,
                "eigvals": eigvals,
                "eigvecs": eigvecs,
                "variance_explained": variance_explained,
                "cumulative_variance": cumulative_variance,
                "prior": prior,
                "likelihood": likelihood,
            }
            # Storing

            run_dir = OUT_DIR / label
            save_run_outputs(
                run_dir = run_dir,
                param_names = param_names,
                u = u,
                theta_samples = theta_samples,
                logp = logp,
                grads = grads,
                C = C,
                eigvals = eigvals,
                eigvecs = eigvecs,
            )

            plot_first_direction_loadings(label, param_names, eigvecs, run_dir)
            plot_sufficient_summary(label, u, logp, eigvecs, run_dir)

            print("Eigenvalues:", eigvals)
            print("Variance explained:", variance_explained)
            print("Cumulative variance:", cumulative_variance)
            print("First active direction:")
            for name, loading in zip(param_names, eigvecs[:, 0]):
                print(f"{name:10s}: {float(loading): .4f}")
            # Printing stuff

    plot_eigenvalue_comparison(all_results)
    plot_variance_explained(all_results)
    write_summary_csv(all_results)
    # Plotting stuff

    print(f"\nDone. Outputs saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
# THE END