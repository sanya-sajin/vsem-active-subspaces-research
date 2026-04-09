# summarize_acq_design.r
#
# Currently, this is very rough code for reading sequential design results,
# conditioning the new emulators, and summarizing emulator error metrics.
# This file should almost be certainly split into multiple steps (e.g., the 
# emulators should probably be updated at the end of the sequential design
# step).
#
# Andrew Roberts

library(data.table)
library(ggplot2)

# Settings
experiment_tag <- "vsem"
round <- 2L
acq_ids <- c(1L, 2L, 3L, 11L, 12L)
design_tag_prev <- "LHS_200"
em_tag_prev <- "llik_quad_mean"

# All directories relative to base_dir.
base_dir <- file.path("/projectnb", "dietzelab", "arober", "gp-calibration")

# Directory to source code.
src_dir <- file.path(base_dir, "src")

# Base experiment directory.
experiment_dir <- file.path(base_dir, "output", "gp_inv_prob", experiment_tag)

# Output directories.
summary_files_dir <- file.path(experiment_dir, paste0("round", round), 
                               "design", "summary_files")
plot_dir <- file.path(summary_files_dir, "plots")
dir.create(plot_dir, showWarnings=FALSE, recursive=TRUE)

# Source required scripts.
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
source(file.path(src_dir, "seq_design.r"))
source(file.path(src_dir, "seq_design_gp.r"))
source(file.path(src_dir, "seq_design_for_post_approx.r"))
source(file.path(src_dir, "sim_study_functions.r"))

# ------------------------------------------------------------------------------
# Assemble table of acquisition run results.
# ------------------------------------------------------------------------------

acq_reps_info <- process_acq_data_reps(experiment_dir, round, acq_ids=acq_ids, 
                                       mcmc_tags_prev="mcwmh-joint-rect",
                                       design_tag_prev=design_tag_prev, 
                                       em_tag_prev=em_tag_prev)

saveRDS(acq_reps_info, file.path(summary_files_dir, "acq_reps_info.rds"))
dt_track_reps <- acq_reps_info$dt_track
dt_itr_reps <- acq_reps_info$dt_itr

# ------------------------------------------------------------------------------
# Generate plots.
# ------------------------------------------------------------------------------

# We want to aggregate over `mcmc_id`.
group_cols <- c("name", "metric", "acq_id", "mcmc_tag", "itr")

# Summarize distribution over replications.
agg_funcs <- list(
  mean = mean,
  median = median,
  q_upper = function(x) quantile(x, 0.8),
  q_lower = function(x) quantile(x, 0.2)
)

dt_track_reps[, acq_id := as.factor(acq_id)]
dt_track_reps_agg <- agg_dt_by_func_list(dt_track_reps, value_col="val",
                                         agg_funcs=agg_funcs,
                                         group_cols=group_cols)


# scale_fill_manual(values = c("Group A" = "blue", "Group B" = "red"))

#
# Comparing rep distribution for two methods.
# 

color_map <- c("3"="blue", "11"="orange")

plt_data <- dt_track_reps_agg[(metric=="rmse") & 
                              (mcmc_tag=="mcwmh-joint-rect") & 
                              (name=="post") &
                              (acq_id %in% names(color_map))]

plt <- ggplot(plt_data, aes(x=itr)) +
        geom_ribbon(aes(ymin=q_lower, ymax=q_upper, fill=acq_id), alpha=0.5) +
        geom_line(aes(y=median, color=acq_id), linetype="dashed") +
        scale_fill_manual(values=color_map) +
        scale_color_manual(values=color_map) + 
        theme_minimal()

plot(plt)


#
# Comparing median run for a bunch of methods.
#

plt_med_data <- dt_track_reps_agg[(metric=="crps") & 
                                  (mcmc_tag=="mcwmh-joint-rect")] # & 
                                  # (name=="post")]
  
plt_med_data[acq_id=="1", acq_id := "IVAR"]
plt_med_data[acq_id=="2", acq_id := "IVAR-post"]
plt_med_data[acq_id=="3", acq_id := "IVAR-mix"]
plt_med_data[acq_id=="11", acq_id := "max-var"]
plt_med_data[acq_id=="12", acq_id := "max-lik-var"]


plt_med <- ggplot(plt_med_data, aes(x=itr)) +
            geom_line(aes(y=median, color=acq_id, linetype=name), linewidth=1.5) +
            theme_minimal() + 
            ggtitle("Log-Lik Error: Prior vs. Post Test Points") +
            labs(color="Acq Func", linetype="Validation Set") +
            xlab("Iteration") + ylab("median crps (100 reps)") +
            theme(
              axis.title.x = element_text(size=16),
              axis.title.y = element_text(size=16),
              axis.text.x = element_text(size=14),
              axis.text.y = element_text(size=14)
            )
  
plot(plt_med)  
  
ggsave(file.path(plot_dir, "seq_design_plt_for_poster.png"))




