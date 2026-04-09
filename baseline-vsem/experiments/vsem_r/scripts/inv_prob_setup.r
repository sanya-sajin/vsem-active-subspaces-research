#
# inv_prob_setup.r
#
# Defines a Bayesian inverse problem based around estimating the parameters
# of the Very Simple Ecosystem Model (VSEM), a toy carbon cycle model 
# consisting of a system of ODEs.
# This script is intended to be called by `run_inv_prob_setup.r`.
# Note that the name of this script is formatted as `<experiment_tag>_setup.r`,
# as required by the `run_inv_prob_setup.r` script.
#

get_inv_prob <- function() {
  
  # ------------------------------------------------------------------------------
  # Inverse problem setup 
  # ------------------------------------------------------------------------------
  
  # Helper function sets up the inverse problem.
  inv_prob <- get_vsem_test_paper(default_conditional=FALSE, 
                                  default_normalize=TRUE)
  
  # Truncating prior to achieve compact support, but capture almost all 
  # prior mass. This just trims off the tail of Cv and tauV. The truncated prior
  # is used when sampling from surrogate-induced posterior approximations.
  prob_prior <- .99
  q_tail <- sqrt(prob_prior)
  prior_bounds <- get_prior_bounds(inv_prob$par_prior, tail_prob_excluded=1-q_tail, 
                                   set_hard_bounds=TRUE)
  par_prior_trunc <- inv_prob$par_prior
  par_prior_trunc$bound_lower <- prior_bounds[1,]
  par_prior_trunc$bound_upper <- prior_bounds[2,]
  inv_prob$par_prior_trunc <- par_prior_trunc
  
  return(inv_prob)
}

