# =============================================================================
# 06_evaluation.R
# Model Evaluation — Gini Coefficient, Double Lift Chart, Calibration
#
# Objectives:
#   1. Load all model predictions and assemble evaluation dataset
#   2. Compute normalised Gini coefficient for each model
#   3. Produce double lift chart (key actuarial diagnostic)
#   4. Calibration by predicted decile — all three models
#   5. Summary comparison table
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
# Evaluation basis — predicted vs observed pairing per model:
#
#   Freq-Sev GLM : pp_freq_sev (loaded, full PP) vs obs_pp (uncapped losses)
#                  In-sample. GLM fitted on full dataset.
#
#   XGBoost      : pp_xgb (attritional, no loading) vs obs_pp_capped (capped losses)
#                  Out-of-sample — 20% holdout test set loaded from test_idx.rds.
#                  XGBoost was trained on capped pure premium; the large loss loading
#                  is not applied here because observed losses are also capped.
#                  Both predicted and observed are on the same attritional scale.
#                  Caveat: early stopping used the test set loss to select rounds (416),
#                  so the Gini may be very slightly optimistic. A three-way split
#                  would eliminate this; documented as a limitation in the README.
#
#   Tweedie GLM  : pp_tweedie (full PP) vs obs_pp (uncapped losses)
#                  In-sample. Tweedie GLM fitted on full dataset.
#
#   Note: Gini figures are not directly comparable across models because
#   XGBoost is evaluated on a different loss basis (capped) than the GLMs
#   (uncapped). This is the honest representation of how the models were built.
#
# Output files:
#   outputs/figures/eval_*.png             — evaluation figures
#   outputs/tables/eval_summary.csv        — model comparison table
# =============================================================================

# Set working directory to project root if running from R/ subfolder
if (basename(getwd()) == "R") setwd("..")

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

dat     <- readRDS("outputs/data/model_data.rds")
pp_dat  <- readRDS("outputs/data/pure_premium.rds")
xgb_dat <- readRDS("outputs/data/xgb_predictions.rds")

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
    # pp_freq_sev and pp_tweedie already include the large loss loading (script 04)
    # pp_xgb is attritional only — no loading applied here; evaluated against capped obs
    # pp_xgb_loaded retained for reference only (not used in primary evaluation)
    pp_xgb_loaded = pp_xgb * LARGE_LOSS_LOADING,
    
    # Observed PP — two versions:
    #   obs_pp        : uncapped, used for GLM evaluation
    #   obs_pp_capped : capped at 99th pct severity threshold, used for XGBoost evaluation
    obs_pp        = TotalClaimAmount / Exposure,
    obs_pp_capped = TotalClaim_capped / Exposure
  )

# Load saved test indices from script 05 — guarantees identical 80/20 split.
test_idx  <- readRDS("outputs/data/test_idx.rds")
eval_test <- eval_dat[test_idx, ]   # XGBoost out-of-sample evaluation set
eval_full <- eval_dat               # GLMs evaluated on full dataset (in-sample)

cat("Evaluation dataset assembled.\n")
cat(sprintf("Full dataset:     %d policies\n", nrow(eval_full)))
cat(sprintf("XGBoost test set: %d policies (%.0f%% holdout)\n",
            nrow(eval_test), 100 * nrow(eval_test) / nrow(eval_full)))
cat(sprintf("\nMean obs PP uncapped  (full):             €%.2f\n",
            weighted.mean(eval_full$obs_pp,        eval_full$Exposure)))
cat(sprintf("Mean obs PP capped    (test set):         €%.2f\n",
            weighted.mean(eval_test$obs_pp_capped, eval_test$Exposure)))
cat(sprintf("Mean freq-sev PP      (full, in-sample):  €%.2f\n",
            weighted.mean(eval_full$pp_freq_sev,   eval_full$Exposure)))
cat(sprintf("Mean XGBoost PP attrn (test, OOS):        €%.2f\n",
            weighted.mean(eval_test$pp_xgb,        eval_test$Exposure)))
cat(sprintf("Mean Tweedie PP       (full, in-sample):  €%.2f\n",
            weighted.mean(eval_full$pp_tweedie,    eval_full$Exposure)))


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

gini_normalised <- function(actual, predicted, weight) {
  ord      <- order(predicted)
  actual   <- actual[ord]
  weight   <- weight[ord]
  
  cum_wt   <- cumsum(weight) / sum(weight)
  cum_loss <- cumsum(actual * weight) / sum(actual * weight)
  
  auc <- sum(diff(c(0, cum_wt)) * (c(0, cum_loss[-length(cum_loss)]) +
                                     c(cum_loss)) / 2)
  gini_model <- 2 * auc - 1
  
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
cat("(XGBoost: OOS, attritional pred vs capped obs)\n")
cat("(GLMs: in-sample, full pred vs uncapped obs)\n")

gini_fs  <- gini_normalised(eval_full$obs_pp,        eval_full$pp_freq_sev, eval_full$Exposure)
gini_xgb <- gini_normalised(eval_test$obs_pp_capped, eval_test$pp_xgb,      eval_test$Exposure)
gini_tw  <- gini_normalised(eval_full$obs_pp,        eval_full$pp_tweedie,  eval_full$Exposure)

cat(sprintf("Freq-Sev GLM (in-sample,    uncapped obs): %.4f\n", gini_fs))
cat(sprintf("XGBoost      (out-of-sample, capped obs):  %.4f\n", gini_xgb))
cat(sprintf("Tweedie GLM  (in-sample,    uncapped obs): %.4f\n", gini_tw))


# =============================================================================
# 3. LORENZ / CONCENTRATION CURVES
# =============================================================================

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

set.seed(123)
sample_full <- sort(sample(nrow(eval_full), size = nrow(eval_full) %/% 100))
sample_test <- sort(sample(nrow(eval_test), size = nrow(eval_test) %/% 100))
ev_full_s   <- eval_full[sample_full, ]
ev_test_s   <- eval_test[sample_test, ]

lorenz_dat <- bind_rows(
  lorenz_curve(ev_full_s$obs_pp,        ev_full_s$pp_freq_sev, ev_full_s$Exposure,
               "Freq-Sev GLM (in-sample)"),
  lorenz_curve(ev_test_s$obs_pp_capped, ev_test_s$pp_xgb,      ev_test_s$Exposure,
               "XGBoost (out-of-sample)"),
  lorenz_curve(ev_full_s$obs_pp,        ev_full_s$pp_tweedie,  ev_full_s$Exposure,
               "Tweedie GLM (in-sample)")
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
    "Freq-Sev GLM (in-sample)"  = "#2166ac",
    "XGBoost (out-of-sample)"   = "#d6604d",
    "Tweedie GLM (in-sample)"   = "#4dac26"
  )) +
  scale_linetype_manual(values = c(
    "Freq-Sev GLM (in-sample)"  = "solid",
    "XGBoost (out-of-sample)"   = "dashed",
    "Tweedie GLM (in-sample)"   = "dotdash"
  )) +
  annotate("text", x = 0.65, y = 0.35, label = "Random (Gini = 0)",
           colour = "grey50", size = 3.2, angle = 35) +
  labs(
    title    = "Concentration Curves — Risk Discrimination",
    subtitle = sprintf(
      "Normalised Gini: Freq-Sev = %.3f (IS) | XGBoost = %.3f (OOS) | Tweedie = %.3f (IS)",
      gini_fs, gini_xgb, gini_tw),
    x        = "Cumulative share of exposure (sorted by predicted risk)",
    y        = "Cumulative share of actual loss",
    colour   = NULL, linetype = NULL,
    caption  = paste(
      "GLMs: in-sample, uncapped observed losses.",
      "XGBoost: out-of-sample 20% holdout, capped observed losses.",
      "Gini figures not directly comparable across models."
    )
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/eval_lorenz.png", p_lorenz,
       width = 8, height = 7, dpi = 150)


# =============================================================================
# 4. DOUBLE LIFT CHART
# =============================================================================
# obs_col argument allows each model to be compared against its own
# appropriate observed loss basis.

double_lift <- function(data, pred_col, model_name, obs_col = "obs_pp", n_bins = 10) {
  data |>
    mutate(pred_decile = ntile(.data[[pred_col]], n_bins)) |>
    group_by(pred_decile) |>
    summarise(
      exposure  = sum(Exposure),
      obs_loss  = sum(.data[[obs_col]] * Exposure),
      pred_loss = sum(.data[[pred_col]] * Exposure),
      obs_pp    = obs_loss / exposure,
      pred_pp   = pred_loss / exposure,
      ratio     = obs_loss / pred_loss,
      .groups   = "drop"
    ) |>
    mutate(model = model_name)
}

cat("\n--- Double lift ratios by decile ---\n")
cat("(XGBoost: OOS, attritional vs capped obs | GLMs: in-sample vs uncapped obs)\n")

dl_fs  <- double_lift(eval_full, "pp_freq_sev", "Freq-Sev GLM (in-sample)",  obs_col = "obs_pp")
dl_xgb <- double_lift(eval_test, "pp_xgb",      "XGBoost (out-of-sample)",   obs_col = "obs_pp_capped")
dl_tw  <- double_lift(eval_full, "pp_tweedie",   "Tweedie GLM (in-sample)",   obs_col = "obs_pp")

dl_all <- bind_rows(dl_fs, dl_xgb, dl_tw)

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
    "Freq-Sev GLM (in-sample)"  = "#2166ac",
    "XGBoost (out-of-sample)"   = "#d6604d",
    "Tweedie GLM (in-sample)"   = "#4dac26"
  )) +
  scale_linetype_manual(values = c(
    "Freq-Sev GLM (in-sample)"  = "solid",
    "XGBoost (out-of-sample)"   = "dashed",
    "Tweedie GLM (in-sample)"   = "dotdash"
  )) +
  labs(
    title    = "Double Lift Chart — Actual / Predicted Loss Ratio by Decile",
    subtitle = "Dashed line = perfect calibration (ratio = 1.0). Sorted by each model's predicted risk.",
    x        = "Predicted pure premium decile (1 = lowest risk)",
    y        = "Actual / predicted loss ratio",
    colour   = NULL, linetype = NULL,
    caption  = paste(
      "GLMs: in-sample, uncapped observed losses.",
      "XGBoost: out-of-sample 20% holdout, capped observed losses (attritional layer only)."
    )
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/eval_double_lift.png", p_dl,
       width = 9, height = 6, dpi = 150)


# =============================================================================
# 5. CALIBRATION BY DECILE
# =============================================================================

calib_plot <- function(data, pred_col, model_name, obs_col = "obs_pp", n_bins = 10) {
  data |>
    mutate(pred_decile = ntile(.data[[pred_col]], n_bins)) |>
    group_by(pred_decile) |>
    summarise(
      obs_pp  = weighted.mean(.data[[obs_col]], Exposure),
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
  calib_plot(eval_full, "pp_freq_sev", "Freq-Sev GLM (in-sample)",  obs_col = "obs_pp"),
  calib_plot(eval_test, "pp_xgb",      "XGBoost (out-of-sample)",   obs_col = "obs_pp_capped"),
  calib_plot(eval_full, "pp_tweedie",  "Tweedie GLM (in-sample)",   obs_col = "obs_pp")
)

p_calib <- ggplot(calib_all,
                  aes(x = pred_decile, y = pp,
                      colour = type, linetype = type)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2) +
  facet_wrap(~ model, ncol = 3) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = comma) +
  scale_colour_manual(values = c("Observed" = "#d6604d", "Predicted" = "#2166ac")) +
  scale_linetype_manual(values = c("Observed" = "dotted", "Predicted" = "solid")) +
  labs(
    title    = "Calibration by Predicted Decile — All Models",
    subtitle = "Observed vs predicted pure premium. Lines should overlap.",
    x        = "Predicted decile (1 = lowest risk)",
    y        = "Pure premium (€/year)",
    colour   = NULL, linetype = NULL,
    caption  = paste(
      "GLMs: uncapped observed losses.",
      "XGBoost: capped observed losses (attritional layer only, no large loss loading)."
    )
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
# Ratio of mean predicted PP in decile 10 vs decile 1.
# Measures spread between highest and lowest predicted risk.
# Computed on predicted values only — same basis across all models.

tdr <- function(data, pred_col) {
  deciles <- data |>
    mutate(pred_decile = ntile(.data[[pred_col]], 10)) |>
    group_by(pred_decile) |>
    summarise(pred_pp = weighted.mean(.data[[pred_col]], Exposure),
              .groups = "drop")
  deciles$pred_pp[10] / deciles$pred_pp[1]
}

cat("\n--- Top decile ratios ---\n")

tdr_fs  <- tdr(eval_full, "pp_freq_sev")
tdr_xgb <- tdr(eval_test, "pp_xgb")
tdr_tw  <- tdr(eval_full, "pp_tweedie")

cat(sprintf("Freq-Sev GLM (in-sample):     %.2f\n", tdr_fs))
cat(sprintf("XGBoost      (out-of-sample):  %.2f\n", tdr_xgb))
cat(sprintf("Tweedie GLM  (in-sample):     %.2f\n", tdr_tw))


# =============================================================================
# 7. MEAN ABSOLUTE ERROR
# =============================================================================

wmae <- function(actual, predicted, weight) {
  weighted.mean(abs(actual - predicted), weight)
}

cat("\n--- Exposure-weighted MAE ---\n")
cat(sprintf("Freq-Sev GLM (in-sample,    uncapped obs): €%.2f\n",
            wmae(eval_full$obs_pp,        eval_full$pp_freq_sev, eval_full$Exposure)))
cat(sprintf("XGBoost      (out-of-sample, capped obs):  €%.2f\n",
            wmae(eval_test$obs_pp_capped, eval_test$pp_xgb,      eval_test$Exposure)))
cat(sprintf("Tweedie GLM  (in-sample,    uncapped obs): €%.2f\n",
            wmae(eval_full$obs_pp,        eval_full$pp_tweedie,  eval_full$Exposure)))


# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

summary_tab <- tibble(
  Model            = c("Freq-Sev GLM", "XGBoost", "Tweedie GLM"),
  Eval_set         = c("In-sample", "Out-of-sample (20% holdout)", "In-sample"),
  Obs_basis        = c("Uncapped losses", "Capped losses (attritional)", "Uncapped losses"),
  Mean_predicted_PP = c(
    round(weighted.mean(eval_full$pp_freq_sev, eval_full$Exposure), 2),
    round(weighted.mean(eval_test$pp_xgb,      eval_test$Exposure), 2),
    round(weighted.mean(eval_full$pp_tweedie,  eval_full$Exposure), 2)
  ),
  Mean_observed_PP = c(
    round(weighted.mean(eval_full$obs_pp,        eval_full$Exposure), 2),
    round(weighted.mean(eval_test$obs_pp_capped, eval_test$Exposure), 2),
    round(weighted.mean(eval_full$obs_pp,        eval_full$Exposure), 2)
  ),
  Ratio = round(c(
    weighted.mean(eval_full$pp_freq_sev, eval_full$Exposure) /
      weighted.mean(eval_full$obs_pp,        eval_full$Exposure),
    weighted.mean(eval_test$pp_xgb,      eval_test$Exposure) /
      weighted.mean(eval_test$obs_pp_capped, eval_test$Exposure),
    weighted.mean(eval_full$pp_tweedie,  eval_full$Exposure) /
      weighted.mean(eval_full$obs_pp,        eval_full$Exposure)
  ), 4),
  Normalised_Gini  = round(c(gini_fs, gini_xgb, gini_tw), 4),
  Top_Decile_Ratio = round(c(tdr_fs, tdr_xgb, tdr_tw), 2),
  WMAE = round(c(
    wmae(eval_full$obs_pp,        eval_full$pp_freq_sev, eval_full$Exposure),
    wmae(eval_test$obs_pp_capped, eval_test$pp_xgb,      eval_test$Exposure),
    wmae(eval_full$obs_pp,        eval_full$pp_tweedie,  eval_full$Exposure)
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