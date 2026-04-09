# sim_study_functions.r
# 
# General helper functions used to process simulation experiment results.
#
# Andrew Roberts

# ------------------------------------------------------------------------------
# General helper functions.
# ------------------------------------------------------------------------------

get_ids_by_tag <- function(experiment_dir, round=NULL, 
                           design_tag_list=NULL, em_tag_list=NULL, 
                           mcmc_tag_list=NULL, only_last_round=FALSE) {
  # Constructs a data.table of IDs that allows tracking dependencies within and
  # across rounds. This is accomplished by starting at round one and then
  # stepping forward through the rounds. The default behavior is to return a 
  # table with all IDs/tags across all rounds included. Optional arguments
  # allow restriction to the set of IDs associated with specific tags by round.
  # For example, `mcmc_tag_list` must be a list of the form:
  #   list(round1=c("mean", "marginal"), round3=c("mean")).
  # This means that the round one MCMC IDs will be restricted to those associated
  # with MCMC tags "mean" or "marginal". This means that any future steps
  # relying on other tags will be excluded. When round 3 is reached, only 
  # MCMC IDs associated with the MCMC tag "mean" will be included from that 
  # round (but this won't eliminate the "marginal" IDs from previous rounds).
  # `design_tag_list` and `em_tag_list` are analogously defined. The one 
  # thing to note about `design_tag_list` is that the round 1 tags are associated
  # with initial design methods, while other round design tags are associated
  # with sequential acquisition methods. The argument `round` allows for the 
  # specification of a stopping point; e.g., `round = 3L` implies that 
  # only IDs from the first three rounds will be fetched. If at any point
  # a required "ID map" file is missing, then the loop will terminate early 
  # and only the IDs up to that point will be returned. Note that the order
  # of the stages within each round is: design -> Emulator -> MCMC.
  # If `only_last_round = TRUE`, then only the IDs from the final round 
  # identified will be returned. Otherwise, the IDs from all rounds will 
  # be returned.

  return_cols <- c("round", "design_tag", "em_tag", "mcmc_tag", "design_id",
                   "em_id", "mcmc_id", "design_seed", "em_seed", "mcmc_seed")
  
  # Identify the rounds to loop over.
  output_dir <- file.path(experiment_dir, "output")
  if(is.null(round)) {
    round_tags <- grep("round", list.files(output_dir), value=TRUE)
    rounds <- sort(as.integer(sub("round", "", round_tags)))
  } else {
    rounds <- 1:round
  }
  
  # The design step in round 1 is treated specially, as it has no prior
  # dependencies.
  id_map_round <- fread(file.path(output_dir, "round1", "design", "id_map.csv"))
  setnames(id_map_round, "seed", "design_seed")
  id_map_round[, round := 1L]
  design_tags <- design_tag_list[["round1"]]
  if(!is.null(design_tags)) {
    id_map_round <- id_map_round[design_tag %in% design_tags]
  }
  
  for(i in seq_along(rounds)) {
    round <- rounds[i]
    round_tag <- paste0("round", round)
    round_dir <- file.path(output_dir, round_tag)
    
    # Design.
    if(i != 1L) {
      id_map_round <- try(fread(file.path(round_dir, "design", "id_map.csv")))
      if(any(class(id_map_round) == "try-error")) {
        message("`get_ids_by_tag()` early stop, round ", i, "; step: design")
        break
      } else {
        design_tags <- design_tag_list[[round_tag]]
        if(length(design_tags) > 0L) id_map_round <- id_map_round[design_tag %in% design_tags]
      }

      # Restrict to design tags/IDs that have not been dropped as off the last round.
      setnames(id_map_round, "seed", "design_seed")
      prev_round <- round - 1L
      prev_round_ids <- unique(id_map[round==prev_round, .(design_tag, design_id)])
      id_map_round <- data.table::merge.data.table(id_map_round, prev_round_ids,
                                                   by=c("design_tag", "design_id"),
                                                   all=FALSE) 
    }

    # Emulator.
    id_map_em <- try(fread(file.path(round_dir, "em", "id_map.csv")))
    if(any(class(id_map_em) == "try-error")) {
      message("`get_ids_by_tag()` early stop, round ", i, "; step: emulator")
      break
    } else {
      em_tags <- em_tag_list[[round_tag]]
      if(length(em_tags) > 0L) id_map_em <- id_map_em[em_tag %in% em_tags]
    }
    
    setnames(id_map_em, "seed", "em_seed")
    id_map_round <- data.table::merge.data.table(id_map_round, id_map_em,
                                                 by=c("design_tag", "design_id"),
                                                 all.x=TRUE, all.y=FALSE)
    
    # MCMC.
    id_map_mcmc <- try(fread(file.path(round_dir, "mcmc", "id_map.csv")))
    if(any(class(id_map_mcmc) == "try-error")) {
      message("`get_ids_by_tag()` early stop, round ", i, "; step: MCMC")
      break
    } else {
      mcmc_tags <- mcmc_tag_list[[round_tag]]
      if(length(mcmc_tags) > 0L) id_map_mcmc <- id_map_mcmc[mcmc_tag %in% mcmc_tags]
    }
    
    setnames(id_map_mcmc, "seed", "mcmc_seed")
    id_map_round <- data.table::merge.data.table(id_map_round, id_map_mcmc,
                                                 by=c("em_tag", "em_id"),
                                                 all.x=TRUE, all.y=FALSE)
    
    # Append round IDs to ID map.
    id_map_round[, round := round]
    if(i == 1L) {
      id_map <- copy(id_map_round)[, ..return_cols]
    } else {
      if(only_last_round) {
        id_map <- copy(id_map_round)[, ..return_cols]
      } else {
        id_map <- rbindlist(list(id_map, id_map_round[, ..return_cols]), use.names=TRUE)
      }
    }
  }
  
  return(id_map)
}
  
  

# ------------------------------------------------------------------------------
# MCMC Processing
# ------------------------------------------------------------------------------
  
process_mcmc_round <- function(experiment_dir, round, mcmc_ids, 
                               rhat_threshold=1.05, min_itr_threshold=500L,
                               write_files=FALSE, ...) {
  # Processes MCMC output within a specific round of a specific experiment.
  # Will process all MCMC runs specified in `mcmc_ids`, which is a data.table
  # with columns "mcmc_tag", "em_tag", "em_id", "mcmc_id". Assumes MCMC output
  # for a specific run is stored at path:
  # <experiment_dir>/round_<round>/mcmc/<mcmc_tag>/<em_tag>/em_<em_id>/mcmc_<mcmc_id>/mcmc_samp.rds
  
  # TODO: temporarily including em_tag as column, since I accidentially made 
  # mcmc_id unique by mcmc_tag and em_tag.
  
  mcmc_summary <- data.table(mcmc_tag=character(), mcmc_id=integer(), em_tag=character(),
                             status=character(), max_rhat=numeric(),
                             n_groups=integer(), n_chains=integer(),
                             n_itr=integer())
  group_info <- data.table(mcmc_tag=character(), mcmc_id=integer(), em_tag=character(),
                           group=integer(), chain_str=character(),
                           log_weight=numeric(), itr_start=integer(),
                           itr_stop=integer())
  
  mcmc_dir <- file.path(experiment_dir, "output", paste0("round",round), "mcmc")
  
  for(i in 1:nrow(mcmc_ids)) {
    print(i)
    
    tryCatch({
      # Read MCMC output.
      mcmc_tag <- mcmc_ids[i]$mcmc_tag
      mcmc_id <- mcmc_ids[i]$mcmc_id
      em_tag <- mcmc_ids[i]$em_tag
      em_id <- mcmc_ids[i]$em_id
      samp_list <- try(readRDS(file.path(mcmc_dir, mcmc_tag, em_tag, 
                                         paste0("em_", em_id),
                                         paste0("mcmc_", mcmc_id),
                                         "mcmc_samp.rds")), silent=TRUE)
      
      if(class(samp_list) != "try-error") {
        results <- process_mcmc_run(samp_list, rhat_threshold=rhat_threshold, 
                                    min_itr_threshold=min_itr_threshold, ...)
        results$mcmc_summary[, `:=`(mcmc_tag=mcmc_tag, mcmc_id=mcmc_id, em_tag=em_tag)]
        results$group_info[, `:=`(mcmc_tag=mcmc_tag, mcmc_id=mcmc_id, em_tag=em_tag)]
        
        mcmc_summary <- rbindlist(list(mcmc_summary, results$mcmc_summary), use.names=TRUE)
        group_info <- rbindlist(list(group_info, results$group_info), use.names=TRUE, fill=TRUE)
        
      } else {
        run_summary <- data.table(mcmc_tag=mcmc_tag, mcmc_id=mcmc_id, em_tag=em_tag, n_chains=0L, 
                                  n_groups=0L, max_rhat=NA, status="read_err", n_itr=0L)
        mcmc_summary <- rbindlist(list(mcmc_summary, run_summary), use.names=TRUE)
      }
    }, error = function(cond) {
      run_summary <- data.table(mcmc_tag=mcmc_tag, mcmc_id=mcmc_id, em_tag=em_tag, n_chains=0L, 
                                n_groups=0L, max_rhat=NA, status=conditionMessage(cond), n_itr=0L)
      mcmc_summary <- rbindlist(list(mcmc_summary, run_summary), use.names=TRUE)
    }
    )
    
  }
  
  if(write_files) {
    fwrite(mcmc_summary, file.path(mcmc_dir, "mcmc_summary.csv"))
    fwrite(group_info, file.path(mcmc_dir, "group_info.csv"))
  }
  
  return(invisible(list(mcmc_summary=mcmc_summary, group_info=group_info)))
}


process_mcmc_run <- function(samp_list, rhat_threshold=1.05, 
                             min_itr_threshold=500L, ...) {
  # Post-processing for a single MCMC run (with potentially multiple chains).
  # The processing logic goes as follows:
  # 1.) First checks if an error occurred during the MCMC run. If it did, then
  #     the whole run is marked as invalid.
  # 2.) Next conducts diagnostics to assess across-chain mixing. Increases
  #     burn-in gradually in attempt to satisfy across-chain Rhat threshold,
  #     also subject to a minimum iteration threshold.
  # 3.) If no burnin is found to be able to satisfy both requirements, we 
  #     loosen the requirements to allow for only within-chain mixing. The
  #     same tests are then conducted on a chain-by-chain basis, allowing for 
  #     different burn-ins for each chain. If a chain still fails to satisfy
  #     the requirements, then it is marked as invalid. This means that some
  #     chains within an MCMC run may be "valid", while others are "invalid".
  # 4.) Determine if subsets of the valid individual chains are well-mixed;
  #     i.e., if chains can be "merged". See `merge_chains()` for details.
  #     Ultimately, results in the definition of a number of "groups", each
  #     of which may contain one or more chains. A group is a subset of 
  #     well-mixed chains, which we think of, e.g., as sampling the same mode
  #     of a multi-modal distribution. Weights are assigned to each group using
  #     `calc_chain_weights()`. If the tests in step (2) pass, then there will
  #     only be a single group. If all chains are well-mixed individually but
  #     not well mixed across chains, then there will be one group per chain.
  # 5.) An MCMC run is marked as valid if it has at least one valid group.
  #
  # Args: 
  # `samp_list` is a list returned by the MCMC code, with elements "samp", 
  # "info", and "output_list". `min_itr_threshold` is the minimum number of 
  # samples we allow for a single chain.
  #
  # Returns:
  # A list with elements "mcmc_summary" and "group_info".
  # 
  # mcmc_summary:
  # Provides a high-level summary of the entire MCMC run. This is a data.table
  # with a single row that has columns "status", "max_rhat", "n_chains", 
  # "n_groups", "n_itr". The status will either be "valid", or a string 
  # indicating some type of error occurred. "max_rhat" is the maximum split
  # R-hat over all parameters in the run (these R-hats are only computed
  # within each group). "n_chains" is the total number of original chains
  # composing the groups (will be less than the original number of chains if
  # a chain was marked as invalid). "n_groups" is the number of groups, and 
  # "n_itr" is the number of valid iterations (i.e., after burn-in) summed
  # over each group (number of iterations may vary across groups).
  #
  # group_info:
  # Provides the necessary information for extracting the valid samples for 
  # an MCMC run, including the definition of the groups and their weights.
  # A data.table with columns "group", "chain_str", "max_rhat", "itr_start",
  # "itr_stop", "log_weight". "group" is the integer group index, and "chain_str"
  # is a string encoding of which chains belong to the group; e.g., "1-3" 
  # indicates chains 1 and 3; and "2" indicates chain 2. Note that invalid 
  # chains will not be included in these strings. "itr_start" and "itr_stop" 
  # define the iteration cutoffs for each group, which define the set of valid
  # iterations per group. "log_weight" is an unnormalized group weight on the
  # log scale. The actual weights can be computed by exponentiating and then
  # normalizing so that the group weights sum to one within the MCMC run.
  # This should be done via numerically stable means (e.g., logSumExp) as the
  # weights can have a large range on the log scale.

  # Check to see if error occurred during run.
  err_occurred <- !is.null(samp_list$output_list[[1]]$condition)
  group_info_empty <- data.table(group=integer(), chain_str=character(),
                                 max_rhat=numeric(), itr_start=integer(),
                                 itr_stop=integer(), log_weight=numeric())
  if(err_occurred) {
    mcmc_summary <- data.table(n_chains=0L, n_groups=0L, max_rhat=NA, 
                               status="mcmc_err", n_itr=0L)
    return(list(mcmc_summary=mcmc_summary, group_info=group_info_empty))
  }
  
  if(length(unique(samp_list$samp$test_label)) != 1L) {
    stop("`process_mcmc_run` is defined to operate on a single test label.")
  }
  
  # Determine burn-ins and define chain groups/weights.
  samp_results <- get_chain_groups(samp_list$samp, samp_list$info, 
                                   rhat_threshold=rhat_threshold, 
                                   min_itr_threshold=min_itr_threshold, ...)
  samp_dt <- samp_results$samp
  group_info <- samp_results$chain_group_map
  
  # Case that no valid samples are returned.
  if(nrow(samp_dt) == 0L) {
    mcmc_summary <- data.table(n_chains=0L, n_groups=0L, max_rhat=NA, 
                               status="failed_processing", n_itr=0L)
    return(list(mcmc_summary=mcmc_summary, group_info=group_info_empty))
  }
  
  # Compute R-hat within each group.
  rhat_by_group <- calc_R_hat(samp_dt, within_chain=TRUE, group_col="group")$R_hat_vals

  # Construct MCMC summary.
  mcmc_summary <- data.table(status="valid", max_rhat=max(rhat_by_group$R_hat),
                             n_groups=length(unique(group_info$group)),
                             n_chains=length(unique(group_info$chain_idx)),
                             n_itr=get_n_itr_by_run(samp_dt)$N)

  # String encoding of which chains belong to which group.
  chain_str_func <- function(x) paste0(x, collapse="-")
  group_info <- agg_dt_by_func_list(group_info, "chain_idx", "group", 
                                    list(chain_str=chain_str_func))
  
  # Construct group summary. NULL group weights imply equal weights for all 
  # groups (typically means there is only one group).
  group_weights <- samp_results$group_weights
  if(is.null(group_weights)) {
    group_info[, log_weight := 0]
  } else {
    group_weights[, test_label := NULL]
    group_info <- data.table::merge.data.table(group_info, group_weights,
                                               all=TRUE, by="group")
  }
  
  # Attach min/max iteration for each group.
  group_itr_info <- samp_dt[, .(itr_start=min(itr), itr_stop=max(itr)),
                            by="group"]
  group_info <- data.table::merge.data.table(group_info, group_itr_info,
                                             all=TRUE, by="group")
  
  return(list(mcmc_summary=mcmc_summary, group_info=group_info))
}


get_chain_groups <- function(samp_dt, info_dt, rhat_threshold=1.05, 
                             min_itr_threshold=500L, ...) {
  # A wrapper around `set_mcmc_burnin()`. The logic is as follows:
  # 1.) First runs across-chain diagnostics (Rhat). If `samp_dt` passes, then
  #     the chains are all considered mixed, and thus they are all assigned to
  #     the same "group".
  # 2.) If the above diagnostics don't pass, then tries to identify "groups"
  #     of well-mixed chains. First runs within-chain diagnostics, and keeps
  #     only valid chains (chains that meet the within-chain split Rhat
  #     threshold and the minimum iteration threshold). 
  # 3.) Among the valid chains, determines whether any chains can be "merged" 
  #     (i.e., determine if a subset of the chains are well-mixed). If there 
  #     are N valid chains, then first tries to identify groups of size N-1 
  #     that are well-mixed; continues like this through groups of size 2. Note 
  #     that this is quite inefficient for large numbers of chains, but is 
  #     reasonable for the typical ~4 chains. The same Rhat/minimum iteration
  #     thresholds are used to determine if chains can be merged.
  # 4.) Uses `calc_chain_weights()` to assign a weight to each group. If the
  #     tests in step 1 pass then no weight calculation is needed.
  #
  # Note that in steps 1 and 2, the burn-in is also determined; this is done
  # via calls to `set_mcmc_burnin()`.
  #
  # NOTE: currently, burn-in thresholding is applied the same to all chains
  # within a group; i.e., all chains will have the same starting iteration.
  # This is in general wasteful, and should be updated in the future. For example,
  # suppose the across-chain tests fail and then the within-chain diagnostics
  # return one chain that starts at iteration 1 (chain was well-mixed from
  # the start) and another that starts at iteration 1000 (required burning
  # in 999 samples). In the grouping stage, when considering whether these 
  # two chains should be grouped, the code will drop the first 999 iterations
  # of chain 1 so that the burn-ins of the two chains align. Then it will
  # compute diagnostics to see if the remaining samples from the two chains
  # are well-mixed.

  if(length(unique(samp_dt$test_label)) != 1L) {
    stop("`get_chain_groups` is defined to operate on a single test label.")
  }
  
  # Step 1: First try to satisfy diagnostic thresholds across chains (i.e., 
  # across-chain rhat). If fails, will loosen restrictions and attempt to  
  # satisfy the diagnostic thresholds within chains.
  samp_dt_burned_in <- set_mcmc_burnin(samp_dt, rhat_threshold=rhat_threshold, 
                                       min_itr_threshold=min_itr_threshold, ...)
  
  # Step 2: If across chain diagnostics failed, compute within-chain diagnostics.
  # Drop invalid chains.
  within_chain <- FALSE
  if(nrow(samp_dt_burned_in) == 0L) {
    within_chain <- TRUE
    chains <- unique(samp_dt$chain_idx)
    chain_list <- lapply(chains, function(i) set_mcmc_burnin(samp_dt, chain_idcs=i, 
                                                             rhat_threshold=rhat_threshold, 
                                                             min_itr_threshold=min_itr_threshold, ...))
    samp_dt_burned_in <- rbindlist(chain_list, use.names=TRUE)
  } else {
    # Only a single group in this case.
    samp_dt_burned_in[, group := 1L]
  }
  
  # If all chains are invalid, then there are no samples/chains/groups to return.
  if(nrow(samp_dt_burned_in) == 0L) {
    return(list(samp=samp_dt_burned_in, 
                chain_group_map=data.table(chain_idx=integer(0), group=integer(0)),
                group_weights=NULL))
  }
  
  # Step 3: Determine if some valid chains can be merged. Only run if across
  # chain diagnostics failed. Note that if this is true and there are only
  # 2 valid chains, then we have already determined they aren't mixed. So 
  # only run for 3 or more chains.
  chains <- unique(samp_dt_burned_in$chain_idx)
  if(within_chain) {
    if(length(chains) > 2L) {
      samp_dt_burned_in <- merge_chains(samp_dt_burned_in, 
                                        rhat_threshold=rhat_threshold,
                                        min_itr_threshold=min_itr_threshold, 
                                        starting_group_size=length(chains)-1L, ...)
    } else {
      samp_dt_burned_in[, group := as.integer(.GRP), by=chain_idx]
    }
  }

  chain_group_map <- unique(samp_dt_burned_in[, .(chain_idx, group)])
  
  # Step 4: compute chain weights by group. Only required if across chain
  # diagnostics failed.
  group_weights <- NULL
  if(within_chain) {
    n_groups <- unique(samp_dt_burned_in$group)
    if(length(n_groups) > 1L) {
      # Attach group column to info_dt. Note that `all.x=FALSE` ensures that 
      # dropped chains are also dropped from `info_dt`.
      info_dt <- data.table::merge.data.table(info_dt, chain_group_map,
                                              by="chain_idx", all.x=FALSE)
      
      # Compute weights.
      group_weights <- calc_chain_weights(info_dt, group_col="group")
    }
  }

  
  return(list(samp=samp_dt_burned_in, chain_group_map=chain_group_map,
              group_weights=group_weights))
}


set_mcmc_burnin <- function(samp_dt, chain_idcs=NULL, rhat_threshold=1.05, 
                            min_itr_threshold=500L, itr_start=1L,
                            n_tries=10L) {
  # Processes an MCMC run, or optionally a subset of the chains in the MCMC
  # run. Increases burn-in until a diagnostic threshold is satisfied or 
  # stopping condition is met. Diagnostic condition is currently just within 
  # chain (split) Rhat. If multiple chains are included, then the burn-in 
  # cutoffs are the same for all chains; the diagnostic tests target the goal
  # of all chains being well-mixed. In settings where non-mixing MCMC chains
  # are to be expected, then can instead test for within-chain mixing and apply
  # chain-by-chain burn-in cutoffs by calling this function once for each chain,
  # using the `chain_idcs` argument. Only a single "test_label" should be 
  # included in `samp_dt`; i.e., this function is designed to process a single
  # MCMC run.
  #
  # `itr_start` can be used to specify an initial burn-in, before computing 
  # any Rhat statistics. `n_tries` is the maximum number of adjustments 
  # that will be made to the burn-in before giving up.
  #
  # NOTE: currently to simplify things this function applies that same iteration
  # threshold to all chains. Thus, if chains in `samp_dt` have different 
  # iteration ranges, only iterations that are overlapping across all chains
  # are used. This is in general wasteful but simplifies things for now. 
  # In the future, should probably update to threshold iterations based on
  # their index, not the `itr` column, and allow for different chains to 
  # have different numbers of iterations.

  dt <- select_mcmc_samp(samp_dt, itr_start=itr_start, chain_idcs=chain_idcs)
  
  # Ensure samp_dt includes exactly one MCMC run.
  if(length(unique(dt$test_label)) != 1L) {
    stop("set_mcmc_burnin() requires `samp_dt` have exactly one test_label.")
  }
  
  # Determine whether within or across chain Rhat is to be used. For now,
  # always uses across chain if there are multiple chains in `dt`.
  within_chain <- FALSE
  if(length(unique(dt$chain_idx)) == 1L) within_chain <- TRUE

  # If the MCMC samples already don't meet the min itr threshold, return 
  # empty data.table.
  if(!samp_dt_meets_min_itr_threshold(dt, min_itr_threshold)) {
    return(get_empty_samp_dt())
  }
  
  # Maximum rhat over all parameters.
  rhat <- calc_R_hat(dt, within_chain=within_chain)$R_hat_vals[,max(R_hat)]
  
  # Compute candidate burn-in values to try sequentially until rhat threshold
  # is met.
  min_itr_by_chain <- dt[, .(itr=min(itr)), by=chain_idx]$itr
  max_itr_by_chain <- dt[, .(itr=max(itr)), by=chain_idx]$itr
  max_min_itr <- max(min_itr_by_chain)
  min_max_itr <- min(max_itr_by_chain)
  itr_cutoffs <- round(seq(max_min_itr, min_max_itr, length.out=n_tries))
  i <- 2L
  
  # Note that the condition on minimum iteration threshold prevents infinite 
  # loop here, since the final cutoff will be the final iteration and thus 
  # violate the minimum iteration threshold.
  while(isTRUE(rhat > rhat_threshold)) {
    
    # Increase burn-in size.
    dt <- select_mcmc_itr(dt, itr_start=itr_cutoffs[i])
    
    # If the run doesn't meet the min itr threshold, return empty data.table.
    if(!samp_dt_meets_min_itr_threshold(dt, min_itr_threshold)) {
      return(get_empty_samp_dt())
    }
    
    # Maximum rhat over all parameters.
    rhat <- calc_R_hat(dt, within_chain=within_chain)$R_hat_vals[,max(R_hat)]
    
    i <- i + 1L
  } 
  
  return(dt)
}


samp_dt_meets_min_itr_threshold <- function(samp_dt, min_itr_threshold) {
  # For now this function assumes that all chains have the same number of 
  # iterations. This is not necessary and may be generalized in the future
  # to allow different numbers of iterations per chain.
  
  # Ensure samp_dt includes exactly one MCMC run.
  if(length(unique(samp_dt$test_label)) != 1L) {
    stop("samp_dt_meets_min_itr_threshold() requires `samp_dt` have exactly one test_label.")
  }
  
  n_itr_by_par <- samp_dt[, .N, by=.(param_type, param_name, chain_idx)]
  n_itr_by_par <- unique(n_itr_by_par$N)
  
  if(length(n_itr_by_par) > 1L) {
    stop("Different parameters/chains within MCMC run have different number of iterations.")
  }
  
  n_itr_by_par >= min_itr_threshold
}


merge_chains <- function(samp_dt, rhat_threshold=1.05, min_itr_threshold=500L, 
                         starting_group_size=NULL, ...) {
  # Given MCMC samples `samp_dt` potentially containing multiple nonmixing 
  # chains, tries to identify subsets of chains that are well-mixed (e.g., 
  # sampling the same mode of a multi-modal distribution). Typically, this 
  # function is called after it has already been established that each 
  # individual chain in `samp_dt` is well-mixed. This function is then called
  # to determined subsets of chains that are collectively well-mixed; i.e.,
  # if chains can be "merged". The function proceeds as follows:
  # 1.) Let `n` be the number of chains. The process starts by looking for
  #     groups of size `starting_group_size` < n to merge. The default 
  #     behavior is to start with groups of size `n - 1`. If a group of this
  #     size is found to pass the diagnostics (determined via a call to
  #     `set_mcmc_burnin()`), then a group index is defined indicating that 
  #     the chains belong to the same group.
  # 2.) The process repeats for each group size decreasing until size 2. Once
  #     chains are grouped, they are removed from consideration.
  # 3.) At the end of the loop, chains may be remaining if they were unable 
  #     to be merged. Each of these chains are assigned their own group ID.
  #
  # A modified version of `samp_dt` is returned that includes a column `group`.
  # Note that the merging process may alter the burn-ins 
  # (see `set_mcmc_burnin()`), so the returned data.table may contain different 
  # numbers of iterations than the argument `samp_dt`. If no merges are able to
  # be made, then the returned table will be identical to `samp_dt`, with the
  # addition of the `group`, which will equal the `chain_idx` column in this
  # case. Note also that the process of looping over all combinations of all 
  # sizes is an expensive operation if the number of chains is large. This
  # function is written for the typical case of a small number (~4) chains.
  
  samp_dt <- copy(samp_dt)
  
  # Ensure samp_dt includes exactly one MCMC run.
  if(length(unique(samp_dt$test_label)) != 1L) {
    stop("samp_dt_meets_min_itr_threshold() requires `samp_dt` have exactly one test_label.")
  }
  
  # Determine the (sub)set of chains to try to merge. If there is no potential
  # for merging, return `samp_dt` unchanged.
  chains_unmerged <- unique(samp_dt$chain_idx)
  if(is.null(starting_group_size)) {
    n_merge <- length(chains_unmerged)
  } else {
    n_merge <- starting_group_size
  }
  
  stopifnot(n_merge <= length(chains_unmerged))  
  if(n_merge < 2L) return(samp_dt)
  
  # Start with empty data.table, and build up sequentially by adding back
  # the (potentially merged) chains.
  samp_dt_merged <- get_empty_samp_dt()
  samp_dt_merged[, group := character(0)] 
  group_idx <- 1L
  
  for(n in seq(n_merge, 2L)) {
    chain_combs <- combn(chains_unmerged, n, simplify=FALSE)
    for(comb in chain_combs) {
      if(all(comb %in% chains_unmerged)) {
        samp_dt_comb <- set_mcmc_burnin(samp_dt, chain_idcs=comb,
                                        rhat_threshold=rhat_threshold, 
                                        min_itr_threshold=min_itr_threshold, ...)
        if(nrow(samp_dt_comb) > 0L) {
          samp_dt_comb[, group := group_idx]
          samp_dt_merged <- rbindlist(list(samp_dt_merged, samp_dt_comb), use.names=TRUE)
          chains_unmerged <- setdiff(chains_unmerged, comb)
          group_idx <- group_idx + 1L
        }
      }
      
      if(length(chains_unmerged) < 2L) break
    }
    if(length(chains_unmerged) < 2L) break
  }

  # Any chains that failed to merge will be added as their own group.
  if(length(chains_unmerged) > 0L) {
    samp_dt <- samp_dt[chain_idx %in% chains_unmerged]
    samp_dt[, group := as.integer(.GRP + group_idx - 1L), by=chain_idx]
    samp_dt_merged <- rbindlist(list(samp_dt_merged, samp_dt), use.names=TRUE)
  }

  return(samp_dt_merged)
}


# ------------------------------------------------------------------------------
# MCMC Analysis
#   Functions that are used after MCMC output has already been processed
#   (i.e., burn-in already identified, invalid chains dropped, group weights
#    assigned, etc.). These functions load the processed outputs and 
#   assemble summary statistics/produce plots/etc.
# ------------------------------------------------------------------------------

get_mcmc_ids <- function(experiment_dir, only_valid=TRUE, ...) {
  # A wrapper around `get_ids_by_tag()` that subsets MCMC IDs by user-specified
  # conditions on the tags. If `only_valid = TRUE` then the IDs are further 
  # subset by reading the "mcmc_summary.csv" file from each round (if it 
  # exists) and restricting to IDs that are marked as valid. Note that this 
  # is done on a round-by-round basis; there is no notion of excluding 
  # runs from future rounds that depend on an invalid run from a previous round,
  # as it is assumed that invalid MCMC runs are never used as inputs to future 
  # steps. This assumption could be dropped in the future if needed.
  # The `...` arguments are passed to `get_ids_by_tag()` which allows users
  # to subset by round, design tag, em tag, and mcmc tag. If `only_valid = FALSE`
  # then simply returns the return value from `get_ids_by_tag()`.
  # When `only_valid = TRUE` the returned data.table includes all of the 
  # extra columns from the "mcmc_summary.csv" files: "status", "max_rhat",
  # "n_groups", "n_chains", "n_itr". 

  mcmc_ids <- get_ids_by_tag(experiment_dir, ...)
  if(!only_valid) return(mcmc_ids)
  
  # Loop over rounds and select valid IDs.
  rounds <- unique(mcmc_ids$round)
  output_dir <- file.path(experiment_dir, "output")
  mcmc_ids_valid <- NULL
  
  for(i in seq_along(rounds)) {
    round_num <- rounds[i]
    round_tag <- paste0("round", i)
    mcmc_summary_path <- file.path(output_dir, round_tag, "mcmc", "mcmc_summary.csv")
    
    if(file.exists(mcmc_summary_path)) {
      mcmc_summary <- fread(mcmc_summary_path)[status=="valid"]
      mcmc_ids_round <- data.table::merge.data.table(mcmc_ids[round==round_num], 
                                                     mcmc_summary, all=FALSE, 
                                                     by=c("mcmc_tag", "mcmc_id", "em_tag"))
      if(i == 1L) mcmc_ids_valid <- copy(mcmc_ids_round)
      else mcmc_ids_valid <- rbindlist(list(mcmc_ids_valid, mcmc_ids_round), 
                                       use.names=TRUE)
    } else {
      message("MCMC summary file not found at path: ", mcmc_summary_path)
    }
  }
  
  if(is.null(mcmc_ids_valid)) {
    message("No MCMC summary files found, so returning all IDs from `get_ids_by_tag()`.")
    return(mcmc_ids)
  }
  
  return(mcmc_ids_valid)
}


read_mcmc_samp <- function(experiment_dir, round, mcmc_tag, em_tag,
                           mcmc_id, em_id, only_valid=TRUE, group_info=NULL) {
  # Reads the "mcmc_samp.rds" list for the specified MCMC ID. If 
  # `only_valid = TRUE`, then filters modifies both the "samp_dt" and "info_dt"
  # tables based on the information in `group_info`. In particular:
  # (i.) restricts to the "itr_start", "itr_stop" iteration range.
  # (ii.) keeps only chains specified in "chain_str".
  # (iii.) assigns a "group" tag and associated group weight for each group.
  #        This is done by adding "group" and "log_weight" columns to both 
  #        data.tables.
  #
  # If `group_info` is not given as an argument, then it will be loaded from 
  # file. Also, note that the arguments "experiment_dir", "round", "mcmc_tag",
  # and "mcmc_id" are sufficient to identify a specific MCMC run. However, we 
  # require "em_tag" and "em_id" here to facilitate easy specification of the
  # necessary filepath.

  mcmc_dir <- file.path(experiment_dir, "output", paste0("round", round), "mcmc")
  samp_path <- file.path(mcmc_dir, mcmc_tag, em_tag, paste0("em_", em_id),
                         paste0("mcmc_", mcmc_id), "mcmc_samp.rds")
    
  # Read file.
  samp_list <- try(readRDS(samp_path))
  if(any(class(samp_list) == "try-error")) {
    message("MCMC samples not found. Returning NULL. Path: ",
            samp_path)
    return(NULL)
  }
  
  if(!only_valid) return(samp_list)
  
  # If necessary, read group info file.
  if(is.null(group_info)) {
    group_info_path <- file.path(mcmc_dir, "group_info.csv")
    group_info <- try(fread(group_info_path))
    
    if(any(class(group_info) == "try-error")) {
      message("`group_info` file not found, returning samples without any sort ",
              " of filtering/grouping. Path: ", group_info_path)
      return(samp_list)
    }
  }
  
  tag <- mcmc_tag
  id <- mcmc_id
  e_tag <- em_tag
  group_info <- group_info[(mcmc_tag==tag) & (mcmc_id==id) & (em_tag==e_tag)]
  if(nrow(group_info) == 0L) {
    message("Subsetting `group_info` resulted in zero rows.", 
            " Should result in >= 1 row. Returning samples without any sort ",
            " of filtering/grouping.")
    return(samp_list)
  }
  
  # Restrict to specified chains.
  extract_chains <- function(x) as.integer(strsplit(x, "-", fixed=TRUE)[[1]])
  chain_group_map <- data.table(chain_idx=integer(), group=integer(), log_weight=numeric())
  for(i in 1:nrow(group_info)) {
    group_chains <- extract_chains(group_info[i, chain_str])
    chain_group_map <- rbindlist(list(chain_group_map, 
                                      data.table(chain_idx=group_chains,
                                                 group=group_info[i, group],
                                                 log_weight=group_info[i, log_weight])))
  }
  
  chains <- chain_group_map$chain_idx
  samp_dt <- samp_list$samp
  info_dt <- samp_list$info
  samp_dt <- samp_dt[chain_idx %in% chains]
  info_dt <- info_dt[chain_idx %in% chains]
  
  # Restrict to specified iteration range.
  itr_min <- group_info$itr_start
  itr_max <- group_info$itr_stop
  samp_dt <- samp_dt[(itr >= itr_min) & (itr <= itr_max)]
  info_dt <- info_dt[(itr >= itr_min) & (itr <= itr_max)]
 
  # Add group label and group weights.
  info_dt <- data.table::merge.data.table(info_dt, chain_group_map,
                                          by="chain_idx", all.x=TRUE)

  # Update samp_list and return.
  samp_list$samp <- samp_dt
  samp_list$info <- info_dt
  
  return(samp_list)
}


get_mcmc_stats_agg <- function(experiment_dir, mcmc_ids, format_long=FALSE,
                               interval_probs=NULL, only_valid=TRUE) {
  # This function loops over a specified set of MCMC IDs and loads the 
  # samples associated with each, and computes summary statistics of the 
  # output. A data.table containing the summary statistics from all of the 
  # specified IDs is returned.
  #
  # Args:
  #   experiment_dir: experiment directory, used as base directory for paths.
  #   mcmc_ids: data.table that must have columns "round", "mcmc_tag", "mcmc_id",
  #             and "em_tag". The last one is only included as a temporary
  #             fix and will eventually be removed. This data.table will 
  #             often be constructed via a call to `get_mcmc_ids()`.
  #   format_long: TODO
  #   interval_probs: vector of values in (0,1) representing probabilities 
  #                   defining credible intervals.
  #   only_valid: logical, passed to `read_mcmc_samp()`. Determines whether
  #               MCMC samples will be filtered/group info will be added.
  
  ids <- unique(mcmc_ids[, .(round, mcmc_tag, mcmc_id, em_tag, em_id)])
  rounds <- unique(ids$round)
  output_dir <- file.path(experiment_dir, "output")
  group_cols <- c("test_label", "param_type", "param_name")
  
  # Separate data.tables will store means/vars and marginal credible intervals.
  par_stats <- NULL
  cred_intervals <- NULL
  
  # Load MCMC group info, which contains burn-in and group weight information.
  for(i in seq_along(rounds)) {
    round_num <- rounds[i]
    round_tag <- paste0("round", round_num)
    mcmc_dir <- file.path(output_dir, round_tag, "mcmc")
    group_info <- fread(file.path(mcmc_dir, "group_info.csv"))
    ids_round <- ids[round == round_num]
    
    for(j in 1:nrow(ids_round)) {
      print(paste(i, j, sep = " --- "))
      
      mcmc_tag <- ids_round[j, mcmc_tag]
      mcmc_id <- ids_round[j, mcmc_id]
      em_tag <- ids_round[j, em_tag]
      em_id <- ids_round[j, em_id]
      
      # Read/format samples.
      samp_list <- read_mcmc_samp(experiment_dir, round=round_num, 
                                  mcmc_tag=mcmc_tag, 
                                  em_tag=em_tag,
                                  mcmc_id=mcmc_id, 
                                  em_id=em_id,
                                  only_valid=only_valid,
                                  group_info=group_info)
      
      # Compute univariate aggregate statistics (mean, variance).
      stat_info <- compute_mcmc_param_stats(samp_list$samp, subset_samp=FALSE, 
                                            format_long=format_long,
                                            group_cols=group_cols,
                                            interval_probs=interval_probs,
                                            info_dt=info_dt)
      
      # Attach additional info.
      par_stats_curr <- stat_info$par_stats
      cred_intervals_curr <- stat_info$cred_intervals
      par_stats_curr[, `:=`(round=round_num, mcmc_tag=mcmc_tag, 
                            mcmc_id=mcmc_id, em_tag=em_tag, em_id=em_id)]
      cred_intervals_curr[, `:=`(round=round_num, mcmc_tag=mcmc_tag, 
                                 mcmc_id=mcmc_id, em_tag=em_tag, em_id=em_id)]
      
      # Append.
      if(is.null(par_stats)) par_stats <- copy(par_stats_curr)
      else par_stats <- rbindlist(list(par_stats, par_stats_curr), use.names=TRUE)
      
      if(is.null(cred_intervals)) cred_intervals <- copy(cred_intervals_curr)
      else cred_intervals <- rbindlist(list(cred_intervals, cred_intervals_curr), use.names=TRUE)
    }
  }
  
  return(list(par_stats=par_stats, cred_intervals=cred_intervals))
}



load_samp_mat <- function(experiment_dir, round, mcmc_tag, mcmc_id, 
                          only_valid=TRUE, n_subsamp=NULL, ...) {
  # Loads MCMC samples from a single run, which is uniquely identified 
  # by (round, mcmc_tag, mcmc_id) within the given experiment directory.
  # If `only_valid = TRUE` drops invalid runs/chains/iterations based on the
  # MCMC preprocessing step. If `n_subsamp` is an integer less than the 
  # number of valid samples, will return a subsample of the (valid) MCMC
  # output of size `n_subsamp`.
  #
  # NOTE: call to `select_mcmc_samp_mat()` assumes that the samp_dt contains
  #       only a single param type. 
  #
  # TODO: take into account weights when subsampling.
  
  mcmc_dir <- file.path(experiment_dir, paste0("round", round), "mcmc")
  
  # Load samples.
  samp_dt <- readRDS(file.path(mcmc_dir, mcmc_tag, mcmc_id, "samp.rds"))$samp
  
  # Extract valid samples.
  if(only_valid) {
    chain_summary <- fread(file.path(mcmc_dir, "summary_files", "chain_summary.csv"))
    mcmc_id_curr <- mcmc_id
    mcmc_tag_curr <- mcmc_tag
    chain_summary <- chain_summary[(mcmc_id==mcmc_id_curr) & (mcmc_tag==mcmc_tag_curr),
                                   .(chain_idx, itr_min, itr_max)]
    samp_dt <- select_itr_by_chain(samp_dt, chain_summary)
  }
  
  # Convert to matrix.
  samp_mat <- select_mcmc_samp_mat(samp_dt, ...)
  
  # Sub-sample.
  if(!is.null(n_subsamp) && isTRUE(n_subsamp < nrow(samp_mat))) {
    idcs <- sample(1:nrow(samp_mat), size=n_subsamp, replace=FALSE)
    samp_mat <- samp_mat[idcs,]
  }
  
  return(samp_mat)
}


get_samp_dt_reps <- function(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE) {
  # Loads all MCMC samples found within the given experiment, round, MCMC tag,
  # emulator tag, and design tag. If `only_valid = TRUE`, drops invalid 
  # runs/chains/iterations - see `get_mcmc_ids()` for details. All MCMC runs
  # within the given set of tags are viewed as random replications of a single
  # experimental setup (e.g., replications stemming from different initial 
  # design samples). All of these are compiled into a single data.table, with 
  # a `rep_id` column added, which is set to the corresponding design_id for
  # each run.

  mcmc_id_dt <- get_mcmc_ids(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE)
  
  mcmc_ids <- unique(mcmc_id_dt$mcmc_id)
  mcmc_tag_dir <- file.path(experiment_dir, paste0("round", round), "mcmc",
                            mcmc_tag)
  
  # Read MCMC samples, and subset chains/itrs based on `mcmc_ids`.
  dt_list <- vector(mode="list", length=length(mcmc_ids))

  for(i in seq_along(dt_list)) {
    id <- mcmc_ids[i]
    design_id <- mcmc_id_dt[mcmc_id==id, design_id][1]
    samp_list <- readRDS(file.path(mcmc_tag_dir, id, "samp.rds"))
    samp_dt <- samp_list$samp
    
    # Subset to include only valid iterations.
    if(only_valid) {
      chain_itr_dt <- mcmc_id_dt[mcmc_id==id, .(chain_idx, itr_start=itr_min, 
                                                itr_stop=itr_max)]
      samp_dt <- select_itr_by_chain(samp_dt, chain_itr_dt)
    }
    
    samp_dt[, rep_id := design_id]
    dt_list[[i]] <- samp_dt
  }
  
  samp_dt_reps <- rbindlist(dt_list, use.names=TRUE)
  return(list(samp=samp_dt_reps, ids=mcmc_id_dt))
}


get_samp_dt_reps_agg <- function(experiment_dir, round, mcmc_tag, em_tag, 
                                 design_tag, only_valid=TRUE, format_long=FALSE,
                                 interval_probs=NULL) {
  # This is similar to `get_samp_dt_reps()`, but here each MCMC run is 
  # aggregated after it is loaded, producing one-dimensional summaries of each
  # variable. Currently, this includes mean, variance, and coverage. If such 
  # summaries are all that is needed, this function is a much more 
  # space-efficient option compared to `get_samp_dt_reps()`.
  #
  # TODO:
  # 1.) add support for using chain weights.
  
  # Define grouping columns for aggregation.
  group_cols <- c("test_label", "param_type", "param_name")
  
  # Determine set of MCMC runs to read.  
  mcmc_id_dt <- get_mcmc_ids(experiment_dir, round, mcmc_tag, em_tag, design_tag, 
                             only_valid=TRUE)
  
  mcmc_ids <- unique(mcmc_id_dt$mcmc_id)
  mcmc_tag_dir <- file.path(experiment_dir, paste0("round", round), "mcmc",
                            mcmc_tag)
  
  # Separate data.tables will store means/vars and marginal credible intervals.
  stats_list <- vector(mode="list", length=length(mcmc_ids))
  cred_interval_list <- vector(mode="list", length=length(mcmc_ids))
  
  # Read MCMC samples, and subset chains/itrs based on `mcmc_ids`.
  for(i in seq_along(stats_list)) {
    
    # Read samples from run.
    id <- mcmc_ids[i]
    design_id <- mcmc_id_dt[mcmc_id==id, design_id][1]
    samp_list <- readRDS(file.path(mcmc_tag_dir, id, "samp.rds"))
    samp_dt <- samp_list$samp
    
    # Subset to include only valid iterations.
    if(only_valid) {
      chain_itr_dt <- mcmc_id_dt[mcmc_id==id, .(chain_idx, itr_start=itr_min, 
                                                itr_stop=itr_max)]
      samp_dt <- select_itr_by_chain(samp_dt, chain_itr_dt)
    }
    
    # Compute univariate aggregate statistics (mean, variance).
    stat_info <- compute_mcmc_param_stats(samp_dt, subset_samp=FALSE, 
                                          format_long=format_long,
                                          group_cols=group_cols,
                                          interval_probs=interval_probs)
    
    # Means/variances.
    par_stats <- stat_info$par_stats
    par_stats[, rep_id := design_id]
    stats_list[[i]] <- par_stats
    
    # Marginal credible intervals.
    cred_intervals <- stat_info$cred_intervals
    cred_intervals[, rep_id := design_id]
    cred_interval_list[[i]] <- cred_intervals
  }
  
  par_stats_dt <- rbindlist(stats_list, use.names=TRUE)
  cred_intervals_dt <- rbindlist(cred_interval_list, use.names=TRUE)
  
  return(list(par_stats=par_stats_dt, cred_intervals=cred_intervals_dt,
              ids=mcmc_id_dt))
}


get_mcmc_rep_ids <- function(experiment_dir, round, mcmc_tags_prev, 
                             design_tag_prev, em_tag_prev) {
  # Fetches MCMC IDs that are associated with 
  # the given round, MCMC tags (from the previous round), design tag
  # (from the previous round) and emulator tag (from the previous round), all
  # within the specified `experiment_dir`. This is typically intended to 
  # identify "replications" of the same experimental setup. Note that 
  # `mcmc_tags_prev` can be a vector of multiple tags, while the other 
  # arguments should only specify a single tag.
  
  # Previous round - ensure that it exists (i.e., that this is not the first round).
  prev_round <- round - 1L
  assert_that(prev_round > 0)
  
  # Identify proper directories.
  prev_round_dir <- file.path(experiment_dir, paste0("round", prev_round))
  mcmc_dir_prev <- file.path(prev_round_dir, "mcmc")
  em_dir_prev <- file.path(prev_round_dir, "em")
  
  # Restrict to specified em tag and design tag.
  em_id_map <- fread(file.path(em_dir_prev, em_tag_prev, "id_map.csv"))
  em_id_map <- em_id_map[design_tag==design_tag_prev]
  id_map <- NULL
  
  for(mcmc_tag in mcmc_tags_prev) {
    mcmc_id_map <- fread(file.path(mcmc_dir_prev, mcmc_tag, "id_map.csv"))
    mcmc_id_map <- mcmc_id_map[em_tag==em_tag_prev]
    mcmc_id_map <- data.table::merge.data.table(mcmc_id_map, em_id_map, by="em_id",
                                                all=FALSE)
    mcmc_id_map[, mcmc_tag := mcmc_tag]
    
    if(is.null(id_map)) id_map <- copy(mcmc_id_map)
    else id_map <- rbindlist(list(id_map, mcmc_id_map), use.names=TRUE)
  }
  
  return(id_map)
}


# ------------------------------------------------------------------------------
# Loading/Assembling sequential design/acquisition results.
# ------------------------------------------------------------------------------

load_acq_data <- function(experiment_dir, round, acq_id, mcmc_tag_prev,
                          mcmc_id_prev) {
  # Loads a single acquisition results file.
  
  results_path <- file.path(experiment_dir, paste0("round", round), 
                            "design", paste0("acq_", acq_id), mcmc_tag_prev,
                            mcmc_id_prev, "acq_results.rds")
  acq_results <- readRDS(results_path)
  return(acq_results)
}


process_acq_results <- function(acq_results) {
  # Given a single acquisition results file, converts all of the tracked 
  # quantities into a data.table, and also returns a separate data.table with 
  # the responses at the acquired points and the values of the acquisition 
  # function at each iteration. This latter data.table will have info for
  # every iteration, while the former may not, depending on the interval at 
  # which the tracked quantities were computed.
  
  tracked_quantities <- extract_acq_tracked_quantities_table(acq_results)
  itr_info <- extract_acq_itr_info(acq_results)
  
  return(list(tracked_quantities=tracked_quantities, itr_info=itr_info))
}


extract_acq_tracked_quantities_table <- function(acq_results) {
  # Returns data.table with columns: itr, name, metric, val. In the current
  # setting "name" corresponds to the validation dataset used (prior vs.
  # posterior validation). This function assumes a specific structure
  # as I have currently set it up, but should be generalized in the future.
  
  dt <- data.table(itr=integer(), name=character(), metric=character(),
                   val=numeric())
  tracked_quantities <- acq_results$tracking_list$computed_quantities
  
  # Tracked quantities may not be computed every iteration.
  itrs_tracked_str <- names(tracked_quantities)
  itrs_tracked <- sapply(strsplit(itrs_tracked_str, "_", fixed=TRUE), 
                         function(x) as.integer(x[2]))
  
  # Aggregated metrics.
  log_scores_post <- sapply(itrs_tracked_str, 
                            function(itr) drop(tracked_quantities[[itr]]$agg_post))
  log_scores_prior <- sapply(itrs_tracked_str, 
                             function(itr) drop(tracked_quantities[[itr]]$agg_prior))
  dt_agg_post <- data.table(itr=itrs_tracked, name="post", metric="log_score",
                            val=log_scores_post)
  dt_agg_prior <- data.table(itr=itrs_tracked, name="prior", metric="log_score",
                             val=log_scores_post)
  dt <- rbindlist(list(dt, dt_agg_post, dt_agg_prior), use.names=TRUE)
  
  # Pointwise metrics (which have already been aggregated).
  group_names <- c("pw_prior", "pw_post")
  
  for(i in seq_along(tracked_quantities)) {
    itr <- itrs_tracked[i]
    itr_quantities <- tracked_quantities[[i]]
    
    for(nm in group_names) {
      dt_curr <- copy(itr_quantities[[nm]])
      setnames(dt_curr, c("func", "mean"), c("metric", "val"))
      nm_short <- ifelse(nm=="pw_post", "post", "prior")
      dt_curr[, `:=`(name=nm_short, itr=itr)]
      dt_curr[metric=="mse", `:=`(metric="rmse", val=sqrt(val))]
      dt <- rbindlist(list(dt, dt_curr), use.names=TRUE)
    }
  }
  
  return(dt)  
}


extract_acq_itr_info <- function(acq_results) {
  # Stores data.table with columns: itr, response, acq_val.
  
  data.table(itr = seq_along(acq_results$responses),
             response = acq_results$responses,
             acq_val = acq_results$tracking_list$acq_val)
}


process_acq_data_reps <- function(experiment_dir, round, acq_ids, mcmc_tags_prev,
                                  design_tag_prev, em_tag_prev) {
  # Loads acquisition results for all MCMC IDs that are associated with 
  # the given round, acq IDs, MCMC tags (from the previous round), design tag
  # (from the previous round) and emulator tag (from the previous round), all
  # within the specified `experiment_dir`. This is typically intended to 
  # identify "replications" of the same experimental setup. Note that 
  # `mcmc_tags_prev` and `acq_ids` can be vectors with multiple elements, 
  # while the other arguments should only specify one tag/ID each.
  #
  # Returns the same two data.tables as `process_acq_results()`, where the
  # information from all of the loaded runs have been stacked together. 
  # The following columns are also added: acq_id, mcmc_tag, mcmc_id.
  # Also returns the ID map produced by `get_mcmc_rep_ids()`.
  
  dt_track <- dt_itr <- NULL
  
  # Fetch the MCMC IDs.
  id_map <- get_mcmc_rep_ids(experiment_dir, round, mcmc_tags_prev, 
                             design_tag_prev, em_tag_prev)
  mcmc_tag_id <- unique(id_map[, .(mcmc_tag, mcmc_id)])
  
  for(acq_id in acq_ids) {
    for(i in 1:nrow(mcmc_tag_id)) {
      mcmc_tag_prev <- mcmc_tag_id[i,mcmc_tag]
      mcmc_id_prev <- mcmc_tag_id[i,mcmc_id]
      
      # Process results from the acquisition run.
      acq_results <- load_acq_data(experiment_dir, round, acq_id, 
                                   mcmc_tag_prev, mcmc_id_prev)
      results_list <- process_acq_results(acq_results)
      
      # Append to tracked quantities table.
      tracked_quantities <- results_list$tracked_quantities
      tracked_quantities[, `:=`(acq_id=acq_id, mcmc_tag=mcmc_tag_prev,
                                mcmc_id=mcmc_id_prev)]
      if(is.null(dt_track)) dt_track <- copy(tracked_quantities)
      else dt_track <- rbindlist(list(dt_track, tracked_quantities))
      
      # Append to iteration info table.
      itr_info <- results_list$itr_info
      itr_info[, `:=`(acq_id=acq_id, mcmc_tag=mcmc_tag_prev, mcmc_id=mcmc_id_prev)]
      if(is.null(dt_itr)) dt_itr <- copy(itr_info)
      else dt_itr <- rbindlist(list(dt_itr, itr_info))
    }
  }
  
  return(list(dt_track=dt_track, dt_itr=dt_itr, id_map=id_map,
              design_tag_prev=design_tag_prev, em_tag_prev=em_tag_prev))
}


# ------------------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------------------








