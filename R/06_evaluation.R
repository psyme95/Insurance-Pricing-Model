# =============================================================================
# 06_evaluation.R
# Model Evaluation — Gini Coefficient, Double Lift Chart, Calibration
#
# Objectives:
#   1. Load all model predictions and assemble evaluation dataset
#   2. Apply large loss loading to XGBoost predictions (matching script 04)
#   3. Compute normalised Gini coefficient for each model
#   4. Produce double lift chart (key actuarial diagnostic)
#   5. Calibration by predicted decile — all three models
#   6. Summary comparison table
#
# Evaluation framework:
#   Standard classification metrics (AUC, accuracy) are not appropriate
#   for insurance pricing models. The goal is RISK RANK-ORDERING — correctly
#   identifying which policies are more or less expensive to insure.
#
#   Three insurance-specific metrics are used:
#
#   1. Normalised Gini coefficient
#      Derived from the Lorenz curve of predicted vs actual loss.
#      Measures discrimination: how well the model separates high from low risk.
#      Range: 0 (no discrimination) to 1 (perfect rank-ordering).
#      Typical range for well-performing motor models: 0.25–0.45.
#
#   2. Double lift chart
#      Policies sorted by predicted pure premium, binned into deciles.
#      For each decile: ratio of (actual loss) / (model predicted loss).
#      A well-calibrated model produces ratios close to 1.0 across all deciles.
#      Systematic departures indicate model bias in specific risk segments.
#
#   3. Calibration by decile
#      Mean predicted vs mean actual pure premium within each predicted decile.
#      Assesses absolute accuracy (not just rank-ordering).
#
# Output files:
#   outputs/figures/eval_*.png             — evaluation figures
#   outputs/tables/eval_summary.csv        — model comparison table
# =============================================================================

library(tidyverse)
library(patchwork)
library(scales)

plot_theme <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold", size = 13),
      plot.subtitle    = element_text(colour = "grey40", size = 10),
      axis.title       = element_text(size = 10),
      panel.grid.minor = element_blank(),
      plot.caption     = element_text(colour = "grey50", size = 8)
    )
}


# =============================================================================
# 1. LOAD AND ASSEMBLE EVALUATION DATASET
# =============================================================================

dat        <- readRDS("outputs/data/model_data.rds")
pp_dat     <- readRDS("outputs/data/pure_premium.rds")
xgb_dat    <- readRDS("outputs/data/xgb_predictions.rds")

LARGE_LOSS_LOADING <- 1.4245  # from script 04

eval_dat <- dat |>
  left_join(
    pp_dat |> select(IDpol, pp_freq_sev, pp_tweedie),
    by = "IDpol"
  ) |>
  left_join(
    xgb_dat |> select(IDpol, pp_xgb, TotalClaim_capped),
    by = "IDpol"
  ) |>
  mutate(
    # Apply large loss loading to XGBoost (attritional → full pure premium)
    pp_xgb_loaded  = pp_xgb * LARGE_LOSS_LOADING,
    # Observed pure premium (full, uncapped — this is what we're trying to predict)
    obs_pp         = TotalClaimAmount / Exposure
  )

cat("Evaluation dataset assembled.\n")
cat(sprintf("Policies: %d\n", nrow(eval_dat)))
cat(sprintf("Mean observed PP:              €%.2f\n",
            weighted.mean(eval_dat$obs_pp, eval_dat$Exposure)))
cat(sprintf("Mean freq-sev PP (loaded):     €%.2f\n",
            weighted.mean(eval_dat$pp_freq_sev, eval_dat$Exposure)))
cat(sprintf("Mean XGBoost PP (loaded):      €%.2f\n",
            weighted.mean(eval_dat$pp_xgb_loaded, eval_dat$Exposure)))
cat(sprintf("Mean Tweedie PP:               €%.2f\n",
            weighted.mean(eval_dat$pp_tweedie, eval_dat$Exposure)))


# =============================================================================
# 2. NORMALISED GINI COEFFICIENT
# =============================================================================
# The normalised Gini is computed from the concentration curve of actual losses
# ordered by predicted pure premium.
#
# Algorithm (exposure-weighted):
#   1. Sort policies by predicted pure premium (ascending)
#   2. Compute cumulative share of exposure (x-axis)
#   3. Compute cumulative share of actual loss (y-axis)
#   4. Gini = 2 * (area under concentration curve - 0.5)
#   5. Normalise by the "oracle" Gini (model with perfect predictions)
#
# A higher normalised Gini = better risk discrimination.

gini_normalised <- function(actual, predicted, weight) {
  # Sort by predicted (ascending)
  ord      <- order(predicted)
  actual   <- actual[ord]
  weight   <- weight[ord]
  
  # Cumulative shares
  cum_wt   <- cumsum(weight) / sum(weight)
  cum_loss <- cumsum(actual * weight) / sum(actual * weight)
  
  # Area under concentration curve (trapezoidal rule)
  auc <- sum(diff(c(0, cum_wt)) * (c(0, cum_loss[-length(cum_loss)]) +
                                     c(cum_loss)) / 2)
  gini_model <- 2 * auc - 1
  
  # Oracle Gini: sort by actual loss (best possible ordering)
  ord_oracle    <- order(actual)
  actual_oracle <- actual[ord_oracle]
  weight_oracle <- weight[ord_oracle]
  cum_wt_o      <- cumsum(weight_oracle) / sum(weight_oracle)
  cum_loss_o    <- cumsum(actual_oracle * weight_oracle) /
    sum(actual_oracle * weight_oracle)
  auc_oracle    <- sum(diff(c(0, cum_wt_o)) *
                         (c(0, cum_loss_o[-length(cum_loss_o)]) +
                            c(cum_loss_o)) / 2)
  gini_oracle   <- 2 * auc_oracle - 1
  
  gini_model / gini_oracle
}

cat("\n--- Normalised Gini coefficients ---\n")

gini_fs  <- gini_normalised(eval_dat$obs_pp, eval_dat$pp_freq_sev,
                            eval_dat$Exposure)
gini_xgb <- gini_normalised(eval_dat$obs_pp, eval_dat$pp_xgb_loaded,
                            eval_dat$Exposure)
gini_tw  <- gini_normalised(eval_dat$obs_pp, eval_dat$pp_tweedie,
                            eval_dat$Exposure)

cat(sprintf("Freq-Sev GLM:  %.4f\n", gini_fs))
cat(sprintf("XGBoost:       %.4f\n", gini_xgb))
cat(sprintf("Tweedie GLM:   %.4f\n", gini_tw))


# =============================================================================
# 3. LORENZ / CONCENTRATION CURVES
# =============================================================================
# Visual representation of the Gini computation.
# Each model's curve shows how well it concentrates actual losses
# among predicted high-risk policies.
# A curve closer to the top-left = better discrimination.

lorenz_curve <- function(actual, predicted, weight, model_name) {
  ord      <- order(predicted)
  actual   <- actual[ord]
  weight   <- weight[ord]
  cum_wt   <- cumsum(weight) / sum(weight)
  cum_loss <- cumsum(actual * weight) / sum(actual * weight)
  tibble(
    cum_exposure = c(0, cum_wt),
    cum_loss     = c(0, cum_loss),
    model        = model_name
  )
}

# Sample for plotting speed (1% of data, preserving ordering)
set.seed(42)
sample_idx <- sort(sample(nrow(eval_dat), size = nrow(eval_dat) %/% 100))
ev_s       <- eval_dat[sample_idx, ]

lorenz_dat <- bind_rows(
  lorenz_curve(ev_s$obs_pp, ev_s$pp_freq_sev,    ev_s$Exposure, "Freq-Sev GLM"),
  lorenz_curve(ev_s$obs_pp, ev_s$pp_xgb_loaded,  ev_s$Exposure, "XGBoost"),
  lorenz_curve(ev_s$obs_pp, ev_s$pp_tweedie,      ev_s$Exposure, "Tweedie GLM")
)

p_lorenz <- ggplot(lorenz_dat,
                   aes(x = cum_exposure, y = cum_loss,
                       colour = model, linetype = model)) +
  geom_line(linewidth = 0.9) +
  geom_abline(slope = 1, intercept = 0,
              colour = "grey60", linetype = "dotted") +
  scale_x_continuous(labels = percent) +
  scale_y_continuous(labels = percent) +
  scale_colour_manual(values = c(
    "Freq-Sev GLM" = "#2166ac",
    "XGBoost"      = "#d6604d",
    "Tweedie GLM"  = "#4dac26"
  )) +
  scale_linetype_manual(values = c(
    "Freq-Sev GLM" = "solid",
    "XGBoost"      = "dashed",
    "Tweedie GLM"  = "dotdash"
  )) +
  annotate("text", x = 0.65, y = 0.35, label = "Random (Gini = 0)",
           colour = "grey50", size = 3.2, angle = 35) +
  labs(
    title    = "Concentration Curves — Risk Discrimination",
    subtitle = sprintf(
      "Normalised Gini: Freq-Sev = %.3f | XGBoost = %.3f | Tweedie = %.3f",
      gini_fs, gini_xgb, gini_tw),
    x        = "Cumulative share of exposure (sorted by predicted risk)",
    y        = "Cumulative share of actual loss",
    colour   = NULL, linetype = NULL
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/eval_lorenz.png", p_lorenz,
       width = 8, height = 7, dpi = 150)


# =============================================================================
# 4. DOUBLE LIFT CHART
# =============================================================================
# The double lift chart is the primary actuarial diagnostic for pricing models.
#
# Construction:
#   1. Sort policies by predicted pure premium
#   2. Bin into deciles
#   3. For each decile: compute actual / predicted loss ratio
#   4. Plot ratio by decile — flat line at 1.0 = perfect calibration
#
# A well-behaved model should show:
#   - Ratios close to 1.0 across all deciles
#   - No systematic over/underprediction at either tail
#   - The highest decile (most predicted risk) should have ratio near 1.0

double_lift <- function(data, pred_col, model_name, n_bins = 10) {
  data |>
    mutate(pred_decile = ntile(.data[[pred_col]], n_bins)) |>
    group_by(pred_decile) |>
    summarise(
      exposure   = sum(Exposure),
      obs_loss   = sum(TotalClaimAmount),
      pred_loss  = sum(.data[[pred_col]] * Exposure),
      obs_pp     = obs_loss / exposure,
      pred_pp    = pred_loss / exposure,
      ratio      = obs_loss / pred_loss,
      .groups    = "drop"
    ) |>
    mutate(model = model_name)
}

dl_fs  <- double_lift(eval_dat, "pp_freq_sev",    "Freq-Sev GLM")
dl_xgb <- double_lift(eval_dat, "pp_xgb_loaded",  "XGBoost")
dl_tw  <- double_lift(eval_dat, "pp_tweedie",      "Tweedie GLM")

dl_all <- bind_rows(dl_fs, dl_xgb, dl_tw)

cat("\n--- Double lift ratios by decile ---\n")
dl_all |>
  select(model, pred_decile, obs_pp, pred_pp, ratio) |>
  print(n = 30)

p_dl <- ggplot(dl_all,
               aes(x = pred_decile, y = ratio,
                   colour = model, linetype = model)) +
  geom_hline(yintercept = 1, colour = "grey40", linetype = "dashed",
             linewidth = 0.7) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = 1:10) +
  scale_colour_manual(values = c(
    "Freq-Sev GLM" = "#2166ac",
    "XGBoost"      = "#d6604d",
    "Tweedie GLM"  = "#4dac26"
  )) +
  scale_linetype_manual(values = c(
    "Freq-Sev GLM" = "solid",
    "XGBoost"      = "dashed",
    "Tweedie GLM"  = "dotdash"
  )) +
  labs(
    title    = "Double Lift Chart — Actual / Predicted Loss Ratio by Decile",
    subtitle = "Dashed line = perfect calibration (ratio = 1.0). Sorted by each model's predicted risk.",
    x        = "Predicted pure premium decile (1 = lowest risk)",
    y        = "Actual / predicted loss ratio",
    colour   = NULL, linetype = NULL
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/eval_double_lift.png", p_dl,
       width = 9, height = 6, dpi = 150)


# =============================================================================
# 5. CALIBRATION BY DECILE
# =============================================================================
# Absolute calibration: predicted vs actual pure premium within each decile.
# Complements the double lift chart — shows not just ratio but scale.

calib_plot <- function(data, pred_col, model_name, n_bins = 10) {
  data |>
    mutate(pred_decile = ntile(.data[[pred_col]], n_bins)) |>
    group_by(pred_decile) |>
    summarise(
      obs_pp  = sum(TotalClaimAmount) / sum(Exposure),
      pred_pp = weighted.mean(.data[[pred_col]], Exposure),
      .groups = "drop"
    ) |>
    pivot_longer(c(obs_pp, pred_pp),
                 names_to = "type", values_to = "pp") |>
    mutate(
      type  = recode(type, "obs_pp" = "Observed", "pred_pp" = "Predicted"),
      model = model_name
    )
}

calib_all <- bind_rows(
  calib_plot(eval_dat, "pp_freq_sev",   "Freq-Sev GLM"),
  calib_plot(eval_dat, "pp_xgb_loaded", "XGBoost"),
  calib_plot(eval_dat, "pp_tweedie",    "Tweedie GLM")
)

p_calib <- ggplot(calib_all,
                  aes(x = pred_decile, y = pp,
                      colour = type, linetype = type)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  facet_wrap(~ model, ncol = 3) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = comma) +
  scale_colour_manual(values = c("Observed" = "#d6604d",
                                 "Predicted" = "#2166ac")) +
  scale_linetype_manual(values = c("Observed" = "dotted",
                                   "Predicted" = "solid")) +
  labs(
    title    = "Calibration by Predicted Decile — All Models",
    subtitle = "Observed vs predicted pure premium. Lines should overlap.",
    x        = "Predicted decile (1 = lowest risk)",
    y        = "Pure premium (€/year)",
    colour   = NULL, linetype = NULL
  ) +
  plot_theme() +
  theme(
    legend.position = "bottom",
    strip.text      = element_text(face = "bold")
  )

ggsave("outputs/figures/eval_calibration.png", p_calib,
       width = 12, height = 5, dpi = 150)


# =============================================================================
# 6. TOP DECILE RATIO
# =============================================================================
# The top decile ratio (TDR) summarises discrimination in a single number:
# the ratio of mean predicted pure premium in the top decile vs the bottom.
# Higher TDR = model identifies a wider spread between high and low risk.

tdr <- function(data, pred_col) {
  deciles <- data |>
    mutate(pred_decile = ntile(.data[[pred_col]], 10)) |>
    group_by(pred_decile) |>
    summarise(pred_pp = weighted.mean(.data[[pred_col]], Exposure),
              .groups = "drop")
  deciles$pred_pp[10] / deciles$pred_pp[1]
}

tdr_fs  <- tdr(eval_dat, "pp_freq_sev")
tdr_xgb <- tdr(eval_dat, "pp_xgb_loaded")
tdr_tw  <- tdr(eval_dat, "pp_tweedie")

cat("\n--- Top decile ratios ---\n")
cat(sprintf("Freq-Sev GLM:  %.2f\n", tdr_fs))
cat(sprintf("XGBoost:       %.2f\n", tdr_xgb))
cat(sprintf("Tweedie GLM:   %.2f\n", tdr_tw))


# =============================================================================
# 7. MEAN ABSOLUTE ERROR BY SEGMENT
# =============================================================================
# MAE on the pure premium scale, exposure-weighted.
# Note: MAE is less meaningful than Gini for pricing but useful for
# communicating model accuracy to non-actuarial stakeholders.

wmae <- function(actual, predicted, weight) {
  weighted.mean(abs(actual - predicted), weight)
}

cat("\n--- Exposure-weighted MAE ---\n")
cat(sprintf("Freq-Sev GLM:  €%.2f\n",
            wmae(eval_dat$obs_pp, eval_dat$pp_freq_sev,   eval_dat$Exposure)))
cat(sprintf("XGBoost:       €%.2f\n",
            wmae(eval_dat$obs_pp, eval_dat$pp_xgb_loaded, eval_dat$Exposure)))
cat(sprintf("Tweedie GLM:   €%.2f\n",
            wmae(eval_dat$obs_pp, eval_dat$pp_tweedie,     eval_dat$Exposure)))


# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

summary_tab <- tibble(
  Model = c("Freq-Sev GLM", "XGBoost", "Tweedie GLM"),
  Mean_predicted_PP = c(
    round(weighted.mean(eval_dat$pp_freq_sev,   eval_dat$Exposure), 2),
    round(weighted.mean(eval_dat$pp_xgb_loaded, eval_dat$Exposure), 2),
    round(weighted.mean(eval_dat$pp_tweedie,     eval_dat$Exposure), 2)
  ),
  Mean_observed_PP = round(weighted.mean(eval_dat$obs_pp, eval_dat$Exposure), 2),
  Ratio = round(c(
    weighted.mean(eval_dat$pp_freq_sev,   eval_dat$Exposure),
    weighted.mean(eval_dat$pp_xgb_loaded, eval_dat$Exposure),
    weighted.mean(eval_dat$pp_tweedie,     eval_dat$Exposure)
  ) / weighted.mean(eval_dat$obs_pp, eval_dat$Exposure), 4),
  Normalised_Gini = round(c(gini_fs, gini_xgb, gini_tw), 4),
  Top_Decile_Ratio = round(c(tdr_fs, tdr_xgb, tdr_tw), 2),
  WMAE = round(c(
    wmae(eval_dat$obs_pp, eval_dat$pp_freq_sev,   eval_dat$Exposure),
    wmae(eval_dat$obs_pp, eval_dat$pp_xgb_loaded, eval_dat$Exposure),
    wmae(eval_dat$obs_pp, eval_dat$pp_tweedie,     eval_dat$Exposure)
  ), 2)
)

cat("\n--- Model comparison summary ---\n")
print(summary_tab)

write_csv(summary_tab, "outputs/tables/eval_summary.csv")


# =============================================================================
# 9. COMBINED SUMMARY FIGURE
# =============================================================================

p_summary <- p_lorenz / p_dl +
  plot_annotation(
    title    = "Insurance Pricing Model Evaluation",
    subtitle = "French Motor MTPL — Freq-Sev GLM vs XGBoost vs Tweedie GLM",
    theme    = theme(
      plot.title    = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(colour = "grey40", size = 11)
    )
  )

ggsave("outputs/figures/eval_summary.png", p_summary,
       width = 9, height = 13, dpi = 150)

cat("\nEvaluation complete.\n")
cat("Figures written to outputs/figures/\n")
cat("Summary table written to outputs/tables/eval_summary.csv\n")