# 
# save_alg_settings.r
# Defines the emulator models, MCMC algorithms, and sequential design methods
# that will be used throughout the experiment.
#
# Andrew Roberts
#

get_init_design_settings <- function() {
  # Returns a named list, with the names being the initial design tags.
  # Each element is a function that returns a list representing an initial
  # design, of the form as returned by `get_init_design_list()`. Each 
  # function should take two arguments: `inv_prob` and `n`, where `n` is the 
  # number of design points.
  
  l <- list()
  
  l$simple <- function(inv_prob, n, ...) {
    get_init_design_list(inv_prob, design_method="simple", N_design=n)
  }
  
  l$lhs <- function(inv_prob, n, ...) {
    get_init_design_list(inv_prob, design_method="LHS", N_design=n)
  }
  
  l$lhs_extrap <- function(inv_prob, n, frac_tail=0.2, q=0.9) {
    # Combine LHS prior-based points with fixed "extrapolation points" geared
    # towards capturing the global trend. `1-frac_tail` fraction of the points
    # will be sampled from the prior, and the remainder will explicitly target
    # the tails. The quantile `q` defines what is meant by the tails, is a 
    # marginal parameter-by-parameter sense.

    n_tails <- round(frac_tail * n)
    n_prior <- n - n_tails
    llik <- function(U) inv_prob$llik_obj$assemble_llik(U)
    
    # Inputs sampled from prior. Enforcing bound constraints here since not all
    # algorithms supported by `get_batch_design()` currently support bound constraints.
    prior_inputs <- get_batch_design("LHS", N_batch=n_prior, prior_params=inv_prob$par_prior)
    sel_violations <- par_violates_bounds(prior_inputs, inv_prob$par_prior_trunc)
    n_violations <- sum(sel_violations)
    if(n_violations > 0L) {
      prior_inputs_new <- sample_prior(inv_prob$par_prior_trunc, n=n_violations)
      prior_inputs <- rbind(prior_inputs[!sel_violations,,drop=FALSE], prior_inputs_new)
    }
    
    assert_that(sum(par_violates_bounds(prior_inputs, inv_prob$par_prior_trunc)) == 0L)
    
    # Sample the rest in the tails. Simply uniformly sampling in the tails, 
    # using latin hypercube.
    tail_bounds <- get_prior_bounds(inv_prob$par_prior_trunc, set_hard_bounds=TRUE)
    q_tails <- get_prior_bounds(inv_prob$par_prior, tail_prob_excluded=1-q,
                                set_hard_bounds=FALSE)
    tail_bounds_tauV <- tail_bounds
    tail_bounds_Cv <- tail_bounds
    tail_bounds_tauV[1, "tauV"] <- q_tails[2, "tauV"]
    tail_bounds_Cv[1, "Cv"] <- q_tails[2, "Cv"]
    
    n_tauV <- floor(n_tails/2)
    n_Cv <- n_tails - n_tauV
    inputs_tauV <- get_LHS_sample(n_tauV, bounds=tail_bounds_tauV)
    inputs_Cv <- get_LHS_sample(n_Cv, bounds=tail_bounds_Cv)
    
    # Sometimes numerical stability in the underlying ODE solver causes NA values,
    # so we also filter out inputs resulting in NAs as well.
    inputs <- rbind(prior_inputs, inputs_tauV, inputs_Cv)
    na_sel <- is.na(llik(inputs))
    n_na <- sum(na_sel)
    max_itr <- 1e3
    itr <- 1L
    
    while(n_na > 0) {
      prior_inputs_new <- sample_prior(inv_prob$par_prior_trunc, n=n_na)
      inputs[na_sel,] <- prior_inputs_new
      na_sel <- is.na(llik(inputs))
      n_na <- sum(na_sel)
      
      itr <- itr + 1
      if(itr == max_itr) stop("lhs_extrap design failed due to NAs.")
    }
    
    # Construct full design.
    design_info <- get_init_design_list(inv_prob, "LHS_extrap", n, inputs=inputs)

    return(design_info)
  }
  
  return(l)
}


get_emulator_settings <- function() {
  # Different emulator models are stored in a list. The names of the list
  # are used as the unique ID for each model. Each element of the list is
  # itself a list with elements "em_label", "is_fwd_em", and "fit_em". 
  # "label" serves as a unique identifier for the model. "is_fwd_em" is a
  # logical value indicating whether the model emulates the forward model 
  # or the log-likelihood. "fit_em" is a function that takes the arguments 
  # `design_info` and `inv_prob`, and returns an emulator object that has been 
  # fit to that design.
  
  em_list <- list()
  
  # Log-likelihood emulator. Gaussian kernel, constant mean.
  em_list$em_llik$em_label <- "em_llik"
  em_list$em_llik$is_fwd_em <- FALSE
  em_list$em_llik$fit_em <- function(design_info, inv_prob) {
    
    # Fit GP for log-likelihood.
    gp_obj <- gpWrapperKerGP(design_info$input, matrix(design_info$llik, ncol=1),
                            scale_input=TRUE, normalize_output=TRUE)
    gp_obj$set_gp_prior("Gaussian", "quadratic", include_noise=FALSE)
    gp_obj$fit(multistart=10L)
    
    # Instantiate and save log-likelihood emulator object.
    llik_em <- llikEmulatorGP("em_llik", gp_obj, default_conditional=FALSE, 
                              default_normalize=TRUE, 
                              lik_par=inv_prob$llik_obj$get_lik_par(), 
                              llik_bounds=inv_prob$llik_obj$get_llik_bounds())
    
    return(llik_em)
  }
  
  # Unnormalized log-posterior density emulator. Two step approach: fit 
  # quadratic mean function with constrained optimization, then fit stationary
  # GP with Gaussian kernel to residuals. 
  em_list$em_lpost_twostage$em_label <- "em_lpost_twostage"
  em_list$em_lpost_twostage$is_fwd_em <- FALSE
  em_list$em_lpost_twostage$fit_em <- function(design_info, inv_prob) {
    
    # Stage one: quadratic regression to use as mean function.
    fit <- fit_constr_quad_reg(design_info, inv_prob)
    mean_func <- function(U) {
      Phi <- inv_prob$par_maps$fwd(U)
      drop(quad_basis(Phi) %*% fit$par)
    }
    
    # Fit stationary GP to residuals.
    resids <- design_info$llik - mean_func(design_info$input)
    gp_obj <- gpWrapperHet(design_info$input, matrix(resids, ncol=1L),
                           scale_input=TRUE, normalize_output=TRUE)
    gp_obj$set_gp_prior("Gaussian", "constant", include_noise=FALSE)
    gp_obj$fit()
    
    # Construct llikEmulator object. Shifting so that predictions correspond
    # to log-posterior predictions. Also need to be careful when specifying
    # the bounds, since the emulator is fit to residuals, not the raw 
    # log-likelihood values.
    shift_func <- function(U, ...) mean_func(U) + calc_lprior_dens(U, inv_prob$par_prior_trunc)
    lpost_bounds <- function(U, ...) {
      llik_bounds <- inv_prob$llik_obj$get_llik_bounds()
      resid_bounds <- function(U) llik_bounds[2] - mean_func(U)
      
      list(lower = -Inf,
           upper = resid_bounds(U) + shift_func(U))
    }
    
    lpost_em <- llikEmulatorGP("em_lpost_twostage", gp_obj, 
                               default_conditional=FALSE, 
                               default_normalize=TRUE,
                               shift_func=shift_func, 
                               llik_bounds=lpost_bounds,
                               is_lpost_em=TRUE)
    return(lpost_em)
  }
  
  
  # # Unnormalized Log-posterior density emulator. Gaussian kernel, constant mean.
  # em_list$em_lpost$em_label <- "em_lpost"
  # em_list$em_lpost$is_fwd_em <- FALSE
  # em_list$em_lpost$fit_em <- function(design_info, inv_prob) {
  #   
  #   # Fit GP for log-post.
  #   gp_obj <- gpWrapperHet(design_info$input, matrix(design_info$lpost, ncol=1),
  #                          scale_input=TRUE, normalize_output=TRUE)
  #   gp_obj$set_gp_prior("Gaussian", "constant", include_noise=FALSE)
  #   gp_obj$fit()
  #   
  #   # Instantiate and save log-likelihood emulator object.
  #   lpost_em <- llikEmulatorGP("em_lpost", gp_obj, default_conditional=FALSE, 
  #                              default_normalize=TRUE, 
  #                              lik_par=inv_prob$llik_obj$get_lik_par(), 
  #                              llik_bounds=inv_prob$llik_obj$get_llik_bounds())
  #   
  #   return(lpost_em)
  # }
  
  # Forward model emulator. Gaussian kernel, constant mean.
  em_list$em_fwd$em_label <- "em_fwd"
  em_list$em_fwd$is_fwd_em <- TRUE
  em_list$em_fwd$fit_em <- function(design_info, inv_prob) {
    
    # Fit GP for forward model.
    gp_obj <- gpWrapperHet(design_info$input, design_info$fwd,
                           scale_input=TRUE, normalize_output=TRUE)
    gp_obj$set_gp_prior("Gaussian", "constant", include_noise=FALSE)
    gp_obj$fit()
    
    # Instantiate and save log-likelihood emulator object.
    llik_em <- llikEmulatorGPFwdGaussDiag("em_fwd", gp_obj,
                                          y_obs=inv_prob$llik_obj$y,
                                          sig2=inv_prob$llik_obj$get_lik_par(), 
                                          default_conditional=FALSE,
                                          default_normalize=TRUE)
    
    return(llik_em)
  }
  
  # Set list names to the labels for convenience.
  names(em_list) <- sapply(em_list, function(x) x$em_label)
  
  return(em_list)
}


get_mcmc_settings <- function() {
  # Returns list, each element of which defines an MCMC algorithm. Each of these
  # elements is itself a list containing the MCMC settings defining that 
  # algorithm. The "test_label" element of each of these lists is used as 
  # the unique identifier for the algorithm.
  
  # Common settings that will be applied to all MCMC algorithms.
  common_settings <- list(n_itr=150000L, try_parallel=TRUE, n_chain=4L,
                          itr_start=100000L)
  
  # Common settings that will be applied to all algorithms using the BayesianTools
  # wrapper function.
  common_bt_settings <- list(mcmc_func_name="mcmc_bt_wrapper", sampler="DEzs", 
                             defer_ic=TRUE, settings_list=list(consoleUpdates=25000))
  
  # Common settings that will be applied to all algorithms using the 
  # `mcmc_noisy_llik()`.
  common_noisy_settings <- list(mcmc_func_name="mcmc_noisy_llik", 
                                ic_settings=list(approx_type="marginal",
                                                 design_method="simple",
                                                 n_test_inputs=500L,
                                                 n_ic_by_method=c(design_max=2, 
                                                                  approx_max=2)))
  
  # Common settings that will be applied to all algorithms using the 
  # `mcmc_gp_unn_post_dens_approx()`. The approx_type used in the initial
  # condition method will depend on the specific algorithm.
  common_dens_settings <- list(mcmc_func_name="mcmc_gp_unn_post_dens_approx", 
                               ic_settings=list(approx_type=NULL,
                                                n_test_inputs=500L,
                                                alpha=0.8,
                                                design_method="simple",
                                                n_ic_by_method=c(design_max=2, 
                                                                 approx_max=2)))
  
  # List of MCMC settings.
  mcmc_settings_list <- list(
    c(list(test_label="mean", approx_type="mean", adjustment="none"),
      common_settings, common_dens_settings),
    c(list(test_label="marginal", approx_type="marginal", adjustment="none"),
      common_settings, common_dens_settings),
    c(list(test_label="mcwmh-joint", mode="mcwmh", use_joint=TRUE, adjustment="none"),
      common_settings, common_noisy_settings),
    c(list(test_label="mcwmh-ind", mode="mcwmh", use_joint=FALSE, adjustment="none"),
      common_settings, common_noisy_settings),
    c(list(test_label="pm-joint", mode="pseudo-marginal", use_joint=TRUE, adjustment="none"),
      common_settings, common_noisy_settings),
    c(list(test_label="pm-ind", mode="pseudo-marginal", use_joint=FALSE, adjustment="none"),
      common_settings, common_noisy_settings)
  )
  
  # Update the initial condition methods for algs using `mcmc_gp_unn_post_dens_approx()`.
  for(i in seq_along(mcmc_settings_list)) {
    if(mcmc_settings_list[[i]]$mcmc_func_name == "mcmc_gp_unn_post_dens_approx") {
      approx_type <- mcmc_settings_list[[i]]$approx_type
      mcmc_settings_list[[i]]$ic_settings$approx_type <- approx_type
    }
  }
  
  # Repeat all algorithms with the rectified adjustment.
  mcmc_settings_list_rect <- mcmc_settings_list
  for(i in seq_along(mcmc_settings_list_rect)) {
    mcmc_settings_list_rect[[i]]$adjustment <- "rectified"
    old_label <- mcmc_settings_list[[i]]$test_label
    mcmc_settings_list_rect[[i]]$test_label <- paste0(old_label, "-rect")
    
    # Also add rectified adjustment to IC selection method.
    if(mcmc_settings_list_rect[[i]]$mcmc_func_name == "mcmc_noisy_llik") {
      mcmc_settings_list_rect[[i]]$ic_settings$adjustment <- "rectified"
    }
  }
  
  # Combine into single list.
  mcmc_settings_list <- c(mcmc_settings_list, mcmc_settings_list_rect)
  names(mcmc_settings_list) <- sapply(mcmc_settings_list, function(x) x$test_label)
  
  return(mcmc_settings_list) 
}


get_acq_settings <- function() {
  # Returns a list, with each element a sublist defining a sequential design
  # method/acquisition function. The "acq_label" element of each sub-list
  # serves as a unique identifier for the method. For acquisition functions,
  # the "name" argument stored in the settings lists is used to identify 
  # a function of the form `acq_<name>`. The exception is `name = "sample"`,
  # which will instead utilize `get_batch_design()` to sample points, rather
  # than optimizing.
  
  # In this simple 2d example, we optimize over equally spaced grid-points.
  # Note that the "tensor_product_grid" method is deterministic, so this will
  # result in the same set of candidate points for all methods.
  candidate_settings <- list(n_prior=35^2, method="tensor_product_grid")
  
  # List that will be passed to acquisition functions. We pre-define two 
  # common cases here: one where no adjustment is applied to Gaussian predictive
  # distribution, and one where rectification adjustment is performed.
  acq_args_no_adj <- list(adjustment="none")
  acq_args_rect <- list(adjustment="rectified")
  
  # For acquisitions that require integrating over the design space, we define
  # different configurations of grid points used for numerical integration.
  # The number of integration grid points is partitioned between points sampled
  # from the prior and points sampled from an MCMC run, `n_prior` and `n_mcmc`,
  # respectively.
  prior_grid <- list(n_prior=200L, n_mcmc=0L, sample_method="subsample", subsample_method="support")
  post_grid <- list(n_prior=0L, n_mcmc=200L, sample_method="subsample", subsample_method="support")
  mix_grid <- list(n_prior=50L, n_mcmc=150L, sample_method="subsample", subsample_method="support")
  
  # Baseline strategies that simply require sampling from prior or approximate
  # posterior (no optimization performed). Note that the `sample_ratios` will 
  # be normalized to sum to one - they do not determine the number of points
  # sampled, only the proportions.
  l_baseline <- list(
    list(acq_label="sample_prior", name="sample", sample_settings=list(sample_method="simple")), # Simple random sample from prior.
    list(acq_label="subsample_post", name="sample", sample_settings=post_grid),
    list(acq_label="subsample_mix", name="sample", sample_settings=mix_grid)
  )
  
  # Pure sequential design strategies (requires running forward model after
  # each individual acquisition).
  l_pure <- list(
    list(acq_label="ivar_prior", name="llik_IVAR_grid_gp", int_grid_settings=prior_grid),
    list(acq_label="ivar_post", name="llik_IVAR_grid_gp", int_grid_settings=post_grid),
    list(acq_label="ivar_mix", name="llik_IVAR_grid_gp", int_grid_settings=mix_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=prior_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=post_grid),
    list(acq_label="ievar_prior", name="llik_IEVAR_grid", int_grid_settings=mix_grid),
    list(acq_label="neg_var_gp", name="llik_neg_var_gp"),
    list(acq_label="neg_var_lik", name="llik_neg_var_lik")
  )
  
  for(i in seq_along(l_pure)) {
    l_pure[[i]]$response_heuristic <- NA
    l_pure[[i]]$acq_args <- acq_args_no_adj
  }
  
  # Copy all of the pure acquisitions with rectified adjustment.
  l_pure_rect <- l_pure
  for(i in seq_along(l_pure_rect)) {
    l_pure_rect[[i]]$acq_args <- acq_args_rect
    old_label <- l_pure_rect[[i]]$acq_label
    l_pure_rect[[i]]$acq_label <- paste0(old_label, "_rect")
  }
  
  # Batch strategies. These will all use the rectified adjustment.
  l_batch <- l_pure
  for(i in seq_along(l_batch)) {
    l_batch[[i]]$response_heuristic <- "cl_mix"
    l_batch[[i]]$acq_args <- acq_args_rect
    old_label <- l_pure_rect[[i]]$acq_label
    l_batch[[i]]$acq_label <- paste0(old_label, "_clmix")
  }
  
  # Combine into single list and return.  
  l <- c(l_baseline, l_pure, l_pure_rect, l_batch)
  names(l) <- sapply(l, function(x) x$acq_label)
  
  return(l)
}


