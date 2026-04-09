#
# run_inv_prob_setup.r
#
# The first runner script in the simulation study workflow. An experiment 
# directory containing the file 
# `experiments/<experiment_tag>/scripts/inv_prob_setup.r` must already be 
# created. This script must define a function named `get_inv_prob()` that
# returns an R list encoding the inverse problem. This list must include
# the elements: par_names, output_names, par_true, par_prior, dim_par,
# dim_obs, n_obs, llik_obj, seed. Note the convention that `dim_obs` is the
# dimension of the output space of the forward model. `n_obs` indicates the 
# number of independent observations, with each observation having 
# dimension `dim_obs`.
#
# This script performs the following tasks:
# (1) loads the inverse problem via the `get_inv_prob()` function.
# (2) runs (exact) MCMC to obtain baseline exact posterior samples to be used 
#     for validation later on. These posterior samples, as well as samples from
#     the prior will both be saved to file.
# (3) generates two sets of validation points for later use in evaluating 
#     emulators. One set is sampled from the prior and the other from the 
#     exact posterior.
# (4) the information from steps (1) through (3) is all saved to file, for use 
#     in the subsequent parts of the simulation study.
#
# Unlike the scripts in the later parts of the simulation study, this runner 
# file should be run exactly once for each experiment. This serves to define
# the inverse problem, which is then considered fixed throughout all later 
# steps in the experiment.
#
# Andrew Roberts
#

library(ggplot2)
library(data.table)
library(assertthat)

# ------------------------------------------------------------------------------
# Settings 
# ------------------------------------------------------------------------------

# Seed for random number generator.
seed <- 865754
set.seed(seed)

# The experiment base directory.
experiment_tag <- "banana"

# Number of points in each of the two validation sets.
n_test_prior <- 500L
n_test_post <- 500L

# Sampling method to use in constructing the prior validation set.
design_method_test <- "simple"

# Number of prior samples to save.
n_samp_prior <- 50000L

# Specifications for exact MCMC.
mcmc_settings <- list(test_label="exact", mcmc_func_name="mcmc_noisy_llik",
                      n_itr=50000L, try_parallel=TRUE, n_chain=4L)

# Starting iteration defining the end of the burn-in/warm-up for exact MCMC.
burn_in_start <- 40000L

# ------------------------------------------------------------------------------
# Setup 
# ------------------------------------------------------------------------------

# Filepath definitions.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "bip-surrogates-paper")
code_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")
pecan_dir <- file.path(base_dir, "..", "sipnet_calibration", "src")
src_dir <- file.path(code_dir, "src")
experiment_dir <- file.path(base_dir, "experiments", experiment_tag)
out_dir <- file.path(experiment_dir, "output", "inv_prob_setup")

# Source required files.
source(file.path(src_dir, "general_helper_functions.r"))
source(file.path(src_dir, "inv_prob_test_functions.r"))
source(file.path(src_dir, "statistical_helper_functions.r"))
source(file.path(src_dir, "plotting_helper_functions.r"))
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "gp_helper_functions.r"))
source(file.path(src_dir, "gpWrapper.r"))
source(file.path(src_dir, "llikEmulator.r"))
source(file.path(src_dir, "mcmc_helper_functions.r"))
source(file.path(src_dir, "gp_mcmc_functions.r"))
source(file.path(pecan_dir, "prob_dists.r"))

# Create experiment output directory.
dir.create(out_dir, recursive=TRUE)

# Source function defining inverse problem.
source(file.path(experiment_dir, "scripts", "inv_prob_setup.r"))

# Save settings.
setup_settings <- list(seed=seed, experiment_tag=experiment_tag, 
                       n_test_prior=n_test_prior, n_test_post=n_test_post,
                       design_method_test=design_method_test, 
                       n_samp_prior=n_samp_prior, mcmc_settings=mcmc_settings,
                       burn_in_start=burn_in_start)
saveRDS(setup_settings, file=file.path(out_dir, "setup_settings.rds"))

# ------------------------------------------------------------------------------
# Inverse problem setup 
# ------------------------------------------------------------------------------

# In the process of changing the representation of prior distribution information.
# Need to convert to new representation in order to fetch functions for 
# transforming parameters to unbounded space.
inv_prob <- get_inv_prob()
prior_list <- convert_par_info_to_list(inv_prob$par_prior)
par_maps <- get_par_map_funcs(prior_list)
inv_prob$prior_list <- prior_list
inv_prob$par_maps <- par_maps

saveRDS(inv_prob, file=file.path(out_dir, "inv_prob_list.rds"))

# ------------------------------------------------------------------------------
# Exact MCMC 
# ------------------------------------------------------------------------------

# MCMC sampling using exact likelihood.
mcmc_list <- run_mcmc(inv_prob$llik_obj, inv_prob$par_prior, mcmc_settings)

# Table storing MCMC samples.
samp_dt <- mcmc_list$samp
mcmc_metadata <- mcmc_list$output_list

# Save to file.
fwrite(samp_dt, file=file.path(out_dir, "samp_exact_raw.csv"))
saveRDS(mcmc_metadata, file=file.path(out_dir, "samp_exact_metadata.rds"))

# Drop burn-in and save to file.
samp_dt <- select_mcmc_itr(samp_dt, itr_start=burn_in_start)
fwrite(samp_dt, file=file.path(out_dir, "samp_exact.csv"))

# Save prior samples.
prior_samp <- sample_prior(inv_prob$par_prior, n=n_samp_prior)
colnames(prior_samp) <- rownames(inv_prob$par_prior)
prior_samp_dt <- format_samples_mat(prior_samp, param_type="par", 
                                    test_label="prior", chain_idx=1L)
fwrite(prior_samp_dt, file=file.path(out_dir, "prior_samp.csv"))

# ------------------------------------------------------------------------------
# Construct validation test points.
# ------------------------------------------------------------------------------

# Validation inputs sampled from prior.
test_inputs_prior <- sample_prior(inv_prob$par_prior_trunc, n=n_test_prior)
test_info_prior <- get_init_design_list(inv_prob, design_method_test, n_test_prior,
                                        inputs=test_inputs_prior)
saveRDS(test_info_prior, file=file.path(out_dir, "test_info_prior.rds"))

# Validation inputs sub-sampled from true posterior.
samp_post_mat <- select_mcmc_samp_mat(samp_dt, param_type="par", param_names=inv_prob$par_names)
within_support <- !par_violates_bounds(samp_post_mat, inv_prob$par_prior_trunc)

test_info_post <- get_init_design_list(inv_prob, "subsample",
                                       N_design=n_test_post, 
                                       design_candidates=samp_post_mat[within_support,,drop=FALSE])
saveRDS(test_info_post, file=file.path(out_dir, "test_info_post.rds"))


# ------------------------------------------------------------------------------
# Compute statistics/summaries of exact MCMC output.
#    - Used to to compare to approximations when evaluating approximate methods.
# ------------------------------------------------------------------------------

# Compute univariate (i.e., parameter-by-parameter) statistics.
stats_univariate <- compute_mcmc_param_stats(samp_dt, subset_samp=FALSE, 
                                             format_long=FALSE,
                                             group_cols=c("test_label", "param_type", "param_name"))
saveRDS(stats_univariate, file.path(out_dir, "mcmc_exact_stats_univariate.rds"))

# Compute multivariate statistics (posterior mean and covariance).
stats_multivariate <- compute_mcmc_param_stats_multivariate(samp_dt, 
                                                            by_chain=FALSE,
                                                            param_names=inv_prob$par_names)
saveRDS(stats_multivariate, file.path(out_dir, "mcmc_exact_stats_multivariate.rds"))

# ------------------------------------------------------------------------------
# Save MCMC diagnostics.
# ------------------------------------------------------------------------------

plt_dir <- file.path(out_dir, "plots")
dir.create(plt_dir)

# R-hat.
rhat_info <- calc_R_hat(samp_dt)
saveRDS(rhat_info, file.path(out_dir, "rhat_info_exact_mcmc.rds"))

# Trace Plots.
trace_plots <- get_trace_plots(samp_dt)
saveRDS(trace_plots, file.path(out_dir, "trace_plots_exact_mcmc.rds"))

for(i in seq_along(trace_plots)) {
  plt <- trace_plots[[i]]
  lbl <- names(trace_plots)[i]
  ggsave(file.path(plt_dir, paste0("trace_exact_mcmc_", lbl, ".png")), plt)
}

# Histograms.
hist_plots <- get_hist_plots(samp_dt)
saveRDS(hist_plots, file.path(out_dir, "hist_plots_exact_mcmc.rds"))

for(i in seq_along(hist_plots)) {
  plt <- hist_plots[[i]]
  lbl <- names(hist_plots)[i]
  ggsave(file.path(plt_dir, paste0("hist_exact_mcmc_", lbl, ".png")), plt)
}
