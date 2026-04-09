#
# inv_prob_setup.r
#
# Defines a Bayesian inverse problem with a banana-shaped posterior distribution.
# The input (parameter space) dimension is 2d and the observation space
# dimension is 1d. This script is intended to be called by `run_inv_prob_setup.r`.
# Note that the name of this script is formatted as `<experiment_tag>_setup.r`,
# as required by the `run_inv_prob_setup.r` script.
#
# Andrew Roberts
#
# Details on the Bayesian inverse problem:
# y_i | u ~ N(G(u), sig2)
# u1 ~ N(0, sig02)
# u2 ~ N(0, sig02).
# G(u) = u1 + u2^2
#
# The observations y_i are conditionally independent, given u = (u1, u2).
# The variables u1 and u2 are a priori independent.
#

get_inv_prob <- function() {

  # ------------------------------------------------------------------------------
  # Ground truth parameters.
  # ------------------------------------------------------------------------------
  
  par_true <- c(2.0, 1.0)
  sig2_true <- 1.0
  
  # ------------------------------------------------------------------------------
  # Parameter setup and prior distribution.
  # ------------------------------------------------------------------------------
  
  par_names <- c("u1", "u2")
  
  par_prior <- data.frame(dist = c("Gaussian", "Gaussian"),
                          param1 = c(0, 0),
                          param2 = c(1, 1))
  rownames(par_prior) <- par_names
  par_prior$par_name <- par_names
  par_prior$bound_lower <- -Inf
  par_prior$bound_upper <- Inf
  
  # Truncating prior to achieve compact support, but capture almost all 
  # prior mass. The truncated prior is used when sampling from surrogate-induced 
  # posterior approximations.
  prob_prior <- .99
  q_tail <- sqrt(prob_prior)
  prior_bounds <- get_prior_bounds(par_prior, tail_prob_excluded=1-q_tail, 
                                   set_hard_bounds=TRUE)
  par_prior_trunc <- truncate_prior(par_prior, prior_bounds)

  # ------------------------------------------------------------------------------
  # Forward model.
  # ------------------------------------------------------------------------------
  
  # Vectorized forward model. Can take a nx2 matrix or a length 2 vector.
  fwd <- function(U) {
    if(is.null(dim(U))) U <- matrix(U, ncol=2L)
    if(ncol(U) != 2L) {
      stop("`U` must have 2 columns.")
    }
    
    matrix(4*U[,1] + U[,2]^2, ncol=1L)
  }
  
  
  # ------------------------------------------------------------------------------
  # Ground truth outputs and simulated data.
  # ------------------------------------------------------------------------------
  
  # Noiseless ground truth forward model output.
  y_true <- drop(fwd(par_true))
  
  # Noisy observations.
  n_obs <- 35L
  y_obs <- y_true + rnorm(n=n_obs, mean=0, sd=sqrt(sig2_true))
  
  
  # ------------------------------------------------------------------------------
  # Exact likelihood.
  # ------------------------------------------------------------------------------
  
  # Gaussian likelihood.
  llik_exact <- llikEmulatorExactGaussDiag(llik_lbl = "exact", 
                                           fwd_model = fwd, 
                                           fwd_model_vectorized = fwd,
                                           y_obs = matrix(y_obs, ncol=1L), 
                                           dim_par = 2L,
                                           sig2 = sig2_true,
                                           par_names = par_names, 
                                           default_conditional = FALSE, 
                                           default_normalize = TRUE)
  
  # ------------------------------------------------------------------------------
  # Create object (list) encoding inverse problem.
  # ------------------------------------------------------------------------------
  
  inv_prob <- list(par_names=par_names, output_names="y", par_true=par_true, 
                   par_prior=par_prior, dim_par=2L, dim_obs=1L, n_obs=n_obs,
                   llik_obj=llik_exact, par_prior_trunc=par_prior_trunc)
  
  return(inv_prob)
}

