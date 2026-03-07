# =============================================================================
# 03_severity_model.R
# Claim Severity Modelling — Gamma GLM
#
# Objectives:
#   1. Load cleaned data; subset to policies with at least one claim
#   2. Fit a baseline Gamma GLM (log link) for mean severity
#   3. Check distributional assumptions
#   4. Refine via likelihood ratio tests
#   5. Diagnostic plots and one-way lift checks
#   6. Save severity predictions for ALL policies (for pure premium assembly)
#
# Key modelling principles:
#   - Severity is modelled ONLY on policies with at least one claim.
#     Including zero-claim policies would bias the mean upward and
#     conflate frequency and severity signals.
#   - Claim COUNT is used as a frequency weight (policies with 2 claims
#     contribute twice as much information about severity as policies with 1).
#   - No exposure offset: severity is a per-claim quantity, not a rate.
#   - The Gamma GLM with log link handles right-skewed, strictly positive
#     responses without requiring back-transformation corrections
#     (unlike lognormal, which needs a smearing factor).
#
# Output files:
#   outputs/models/sev_model.rds           — final model object
#   outputs/data/sev_predictions.rds       — severity predictions for all policies
#   outputs/figures/sev_*.png              — diagnostic figures
#   outputs/tables/sev_model_summary.csv   — coefficient table
# =============================================================================

library(tidyverse)
library(patchwork)
library(scales)
library(broom)

# Plot theme
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
# 1. LOAD AND PREPARE DATA
# =============================================================================

dat <- readRDS("outputs/data/model_data.rds")

dat <- dat |>
  mutate(
    log_BonusMalus = log(BonusMalus),
    log_Density    = log(Density),
    Area           = fct_relevel(Area, "A"),
    VehBrand       = fct_infreq(VehBrand)
  )

# Severity dataset: policies with at least one claim
# TotalClaimAmount is total paid across all claims on the policy.
# We model mean severity per claim, so divide by ClaimNb where ClaimNb > 1.
dat_sev <- dat |>
  filter(HasClaim) |>
  mutate(
    MeanSeverity = TotalClaimAmount / ClaimNb  # per-claim severity
  )

cat("--- Severity modelling dataset ---\n")
cat("Policies with claims:", nrow(dat_sev), "\n")
cat("Total claims:", sum(dat_sev$ClaimNb), "\n")
cat("Mean severity per claim: €", round(mean(dat_sev$MeanSeverity), 0), "\n")
cat("Median severity per claim: €", round(median(dat_sev$MeanSeverity), 0), "\n")
cat(sprintf("Skewness check: mean/median ratio = %.2f (>1 confirms right skew)\n",
            mean(dat_sev$MeanSeverity) / median(dat_sev$MeanSeverity)))


# =============================================================================
# 2. DISTRIBUTIONAL CHECK
# =============================================================================
# Before fitting, confirm that Gamma is a reasonable distributional choice.
# Gamma is appropriate when:
#   - Response is strictly positive (no zeros)
#   - Variance increases with the mean (heteroscedastic)
#   - Distribution is right-skewed
#
# A log-normal is an alternative. The key practical difference:
#   - Gamma GLM: models E[Y] directly on the original scale
#   - Log-normal: models E[log(Y)], requires smearing correction for E[Y]
# Gamma is preferred here to avoid back-transformation bias.

cat("\n--- Distributional checks ---\n")
cat("Min severity: €", round(min(dat_sev$MeanSeverity), 2), "\n")
cat("Max severity: €", round(max(dat_sev$MeanSeverity), 0), "\n")
cat("Any zeros:", any(dat_sev$MeanSeverity == 0), "\n")

# Coefficient of variation: CV = sd/mean
# For Gamma, CV is constant across risk groups (key Gamma property)
cv_overall <- sd(dat_sev$MeanSeverity) / mean(dat_sev$MeanSeverity)
cat(sprintf("Overall CV: %.3f\n", cv_overall))

# Check CV stability across driver age bands (should be roughly constant)
cv_by_age <- dat_sev |>
  mutate(age_band = cut(DrivAge, breaks = c(17, 30, 45, 60, Inf),
                        labels = c("18-30", "31-45", "46-60", "61+"))) |>
  group_by(age_band) |>
  summarise(
    n    = n(),
    mean = mean(MeanSeverity),
    sd   = sd(MeanSeverity),
    cv   = sd / mean,
    .groups = "drop"
  )

cat("\nCV by driver age band (roughly constant = Gamma appropriate):\n")
print(cv_by_age)

# Q-Q plot on log scale (log-normal would be perfectly linear)
p_qq <- ggplot(dat_sev |> slice_sample(n = 5000),
               aes(sample = log(MeanSeverity))) +
  stat_qq(alpha = 0.3, size = 0.8, colour = "#2166ac") +
  stat_qq_line(colour = "#d6604d", linewidth = 0.9) +
  labs(
    title    = "Q-Q Plot of log(Severity)",
    subtitle = "Sample of 5,000 claims. Departures from line indicate heavy tails.",
    x        = "Theoretical quantiles",
    y        = "Sample quantiles"
  ) +
  plot_theme()

ggsave("outputs/figures/sev_qqplot.png", p_qq,
       width = 7, height = 5, dpi = 150)


# =============================================================================
# 3. LARGE LOSS TREATMENT
# =============================================================================
# From EDA: top 1% of claims accounts for ~38% of total paid loss.
# Large losses are driven by different processes (e.g. catastrophic injury,
# litigation) and can destabilise the Gamma GLM.
# Standard actuarial practice: cap severity at a large loss threshold,
# model the capped distribution, and note the limitation.
#
# We cap at the 99th percentile. This is a modelling choice — document it.

cap_threshold <- quantile(dat_sev$MeanSeverity, 0.99)
cat(sprintf("\n--- Large loss treatment ---\n"))
cat(sprintf("99th percentile threshold: €%.0f\n", cap_threshold))
cat(sprintf("Claims above threshold: %d (%.1f%%)\n",
            sum(dat_sev$MeanSeverity > cap_threshold),
            100 * mean(dat_sev$MeanSeverity > cap_threshold)))
cat(sprintf("Loss ceded to large loss layer: %.1f%%\n",
            100 * sum(pmax(dat_sev$MeanSeverity - cap_threshold, 0)) /
              sum(dat_sev$MeanSeverity)))

dat_sev <- dat_sev |>
  mutate(
    MeanSeverity_capped = pmin(MeanSeverity, cap_threshold),
    is_large_loss       = MeanSeverity > cap_threshold
  )

cat(sprintf("Mean severity after capping: €%.0f (vs €%.0f uncapped)\n",
            mean(dat_sev$MeanSeverity_capped),
            mean(dat_sev$MeanSeverity)))


# =============================================================================
# 4. BASELINE GAMMA GLM
# =============================================================================
# weights = ClaimNb: policies with multiple claims contribute proportionally
# more information about mean severity per claim.
# family = Gamma(link = "log"): log link ensures predictions are positive
# and gives multiplicative interpretation of coefficients.

cat("\n--- Fitting baseline Gamma GLM ---\n")

m_sev_base <- glm(
  MeanSeverity_capped ~ log_BonusMalus +
    DrivAge +
    VehAge +
    VehPower +
    VehGas +
    Area +
    log_Density,
  family  = Gamma(link = "log"),
  weights = ClaimNb,
  data    = dat_sev
)

cat("Baseline model fitted.\n")
print(summary(m_sev_base))


# =============================================================================
# 5. NON-LINEAR TERMS — LRT
# =============================================================================

cat("\n--- Testing non-linear terms ---\n")

# DrivAge quadratic
m_sev_drivage_quad <- glm(
  MeanSeverity_capped ~ log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge +
    VehPower +
    VehGas +
    Area +
    log_Density,
  family  = Gamma(link = "log"),
  weights = ClaimNb,
  data    = dat_sev
)

lrt_sev_drivage <- anova(m_sev_base, m_sev_drivage_quad, test = "LRT")
cat("\nLRT: DrivAge linear vs quadratic\n")
print(lrt_sev_drivage)

# VehAge quadratic
m_sev_vehage_quad <- glm(
  MeanSeverity_capped ~ log_BonusMalus +
    DrivAge +
    VehAge + I(VehAge^2) +
    VehPower +
    VehGas +
    Area +
    log_Density,
  family  = Gamma(link = "log"),
  weights = ClaimNb,
  data    = dat_sev
)

lrt_sev_vehage <- anova(m_sev_base, m_sev_vehage_quad, test = "LRT")
cat("\nLRT: VehAge linear vs quadratic\n")
print(lrt_sev_vehage)

# VehPower quadratic
m_sev_vehpow_quad <- glm(
  MeanSeverity_capped ~ log_BonusMalus +
    DrivAge +
    VehAge +
    VehPower + I(VehPower^2) +
    VehGas +
    Area +
    log_Density,
  family  = Gamma(link = "log"),
  weights = ClaimNb,
  data    = dat_sev
)

lrt_sev_vehpow <- anova(m_sev_base, m_sev_vehpow_quad, test = "LRT")
cat("\nLRT: VehPower linear vs quadratic\n")
print(lrt_sev_vehpow)

# AIC comparison
cat(sprintf("\nBaseline AIC:          %.1f\n", AIC(m_sev_base)))
cat(sprintf("+ DrivAge^2 AIC:       %.1f\n",  AIC(m_sev_drivage_quad)))
cat(sprintf("+ VehAge^2 AIC:        %.1f\n",  AIC(m_sev_vehage_quad)))
cat(sprintf("+ VehPower^2 AIC:      %.1f\n",  AIC(m_sev_vehpow_quad)))


# =============================================================================
# 6. FINAL SEVERITY MODEL
# =============================================================================
# Include quadratic terms that are significant at p < 0.05 in LRT above.
# Update formula below based on LRT results before running.
# Template includes all three — remove non-significant ones.

cat("\n--- Fitting final Gamma GLM ---\n")

m_sev_final <- glm(
  MeanSeverity_capped ~ log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge +
    Area +
    log_Density,
  family  = Gamma(link = "log"),
  weights = ClaimNb,
  data    = dat_sev
)

print(summary(m_sev_final))

# Coefficient table — Wald CIs (profile likelihood too slow for Gamma)
coef_tab_sev <- tidy(m_sev_final, conf.int = TRUE, conf.method = "Wald") |>
  mutate(
    exp_est  = exp(estimate),     # multiplicative effect on mean severity
    exp_low  = exp(conf.low),
    exp_high = exp(conf.high),
    signif   = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      p.value < 0.1   ~ ".",
      TRUE            ~ ""
    )
  ) |>
  select(term, estimate, exp_est, exp_low, exp_high, std.error, p.value, signif)

cat("\n--- Coefficient table ---\n")
print(coef_tab_sev, n = Inf)

write_csv(coef_tab_sev, "outputs/tables/sev_model_summary.csv")


# =============================================================================
# 7. DIAGNOSTIC PLOTS
# =============================================================================

# --- 7a. One-way lift: observed vs predicted severity by key variables ---
# Predicted severity = predict(m_sev_final) for claims-only subset,
# weighted back to policy level

dat_sev <- dat_sev |>
  mutate(sev_pred = predict(m_sev_final, type = "response"))

sev_lift_plot <- function(data, var, var_label, nbins = 10) {
  data |>
    mutate(bin = ntile(.data[[var]], nbins)) |>
    group_by(bin) |>
    summarise(
      midpoint  = mean(.data[[var]]),
      obs_sev   = sum(MeanSeverity_capped * ClaimNb) / sum(ClaimNb),
      pred_sev  = sum(sev_pred * ClaimNb) / sum(ClaimNb),
      .groups   = "drop"
    ) |>
    pivot_longer(c(obs_sev, pred_sev),
                 names_to = "type", values_to = "severity") |>
    mutate(type = recode(type,
                         "obs_sev"  = "Observed",
                         "pred_sev" = "Predicted"
    )) |>
    ggplot(aes(x = midpoint, y = severity, colour = type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    scale_y_continuous(labels = comma) +
    scale_colour_manual(values = c("Observed" = "#d6604d",
                                   "Predicted" = "#2166ac")) +
    labs(x = var_label, y = "Mean severity per claim (€)", colour = NULL) +
    plot_theme() +
    theme(legend.position = "bottom")
}

p_sev_drivage <- sev_lift_plot(dat_sev, "DrivAge",    "Driver age") +
  labs(title = "Observed vs Predicted Severity — Driver Age")
p_sev_bm      <- sev_lift_plot(dat_sev, "BonusMalus", "Bonus-Malus score") +
  labs(title = "Observed vs Predicted Severity — Bonus-Malus")
p_sev_vehage  <- sev_lift_plot(dat_sev, "VehAge",     "Vehicle age") +
  labs(title = "Observed vs Predicted Severity — Vehicle Age")
p_sev_vehpow  <- sev_lift_plot(dat_sev, "VehPower",   "Vehicle power") +
  labs(title = "Observed vs Predicted Severity — Vehicle Power")

p_sev_lift <- (p_sev_drivage + p_sev_bm) / (p_sev_vehage + p_sev_vehpow)

ggsave("outputs/figures/sev_lift_plots.png", p_sev_lift,
       width = 12, height = 10, dpi = 150)

# --- 7b. Deviance residuals vs fitted ---
resid_sev <- tibble(
  fitted    = fitted(m_sev_final),
  residuals = residuals(m_sev_final, type = "deviance")
)

p_sev_resid <- ggplot(resid_sev, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.2, size = 0.8, colour = "#2166ac") +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey30") +
  geom_smooth(method = "loess", se = FALSE, colour = "#d6604d",
              linewidth = 0.8) +
  scale_x_log10(labels = comma) +
  labs(
    title    = "Severity Model — Deviance Residuals vs Fitted",
    subtitle = "Loess smoother — should be flat at 0. Fanning indicates heteroscedasticity.",
    x        = "Fitted values (log scale, €)",
    y        = "Deviance residuals"
  ) +
  plot_theme()

ggsave("outputs/figures/sev_residuals.png", p_sev_resid,
       width = 8, height = 5, dpi = 150)

# --- 7c. Coefficient plot ---
p_sev_coef <- coef_tab_sev |>
  filter(term != "(Intercept)") |>
  mutate(term = fct_reorder(term, exp_est)) |>
  ggplot(aes(x = exp_est, xmin = exp_low, xmax = exp_high, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
  geom_errorbar(aes(xmin = exp_low, xmax = exp_high),
                width = 0.3, colour = "grey50",
                orientation = "y") +
  geom_point(size = 2.5, colour = "#d6604d") +
  labs(
    title    = "Severity Model — Multiplicative Effects on Mean Claim Cost",
    subtitle = "exp(coefficient) > 1 = higher severity than reference",
    x        = "Multiplicative effect on mean severity",
    y        = NULL
  ) +
  plot_theme()

ggsave("outputs/figures/sev_coefplot.png", p_sev_coef,
       width = 9, height = 7, dpi = 150)


# =============================================================================
# 8. CALIBRATION CHECK
# =============================================================================

total_obs_sev  <- sum(dat_sev$MeanSeverity_capped * dat_sev$ClaimNb)
total_pred_sev <- sum(dat_sev$sev_pred * dat_sev$ClaimNb)

cat(sprintf("\n--- Global calibration (capped severity) ---\n"))
cat(sprintf("Observed total loss (capped): €%.0f\n", total_obs_sev))
cat(sprintf("Predicted total loss:         €%.0f\n", total_pred_sev))
cat(sprintf("Ratio:                        %.4f\n", total_pred_sev / total_obs_sev))

# Calibration by predicted severity decile
calib_sev <- dat_sev |>
  mutate(
    pred_decile = ntile(sev_pred, 10)
  ) |>
  group_by(pred_decile) |>
  summarise(
    n_claims   = sum(ClaimNb),
    obs_sev    = sum(MeanSeverity_capped * ClaimNb) / sum(ClaimNb),
    pred_sev   = sum(sev_pred * ClaimNb) / sum(ClaimNb),
    ratio      = obs_sev / pred_sev,
    .groups    = "drop"
  )

cat("\n--- Calibration by predicted severity decile ---\n")
print(calib_sev)

p_sev_calib <- calib_sev |>
  pivot_longer(c(obs_sev, pred_sev),
               names_to = "type", values_to = "severity") |>
  mutate(type = recode(type,
                       "obs_sev"  = "Observed",
                       "pred_sev" = "Predicted"
  )) |>
  ggplot(aes(x = pred_decile, y = severity, colour = type)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(labels = comma) +
  scale_colour_manual(values = c("Observed" = "#d6604d",
                                 "Predicted" = "#2166ac")) +
  labs(
    title    = "Severity Model Calibration by Predicted Decile",
    subtitle = "Observed vs predicted mean severity — lines should overlap",
    x        = "Predicted severity decile (1 = lowest severity)",
    y        = "Mean severity per claim (€)",
    colour   = NULL
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/sev_calibration.png", p_sev_calib,
       width = 8, height = 5, dpi = 150)


# =============================================================================
# 9. GENERATE PREDICTIONS FOR ALL POLICIES
# =============================================================================
# Severity predictions are needed for all policies (not just claims-only)
# to assemble pure premium in script 04.
# For non-claim policies, predicted severity is the model's estimate of
# what severity WOULD BE if a claim occurred — this is correct.
# Pure premium = predicted frequency × predicted severity for ALL policies.

sev_preds_all <- dat |>
  mutate(
    sev_pred = predict(m_sev_final, newdata = dat, type = "response")
  ) |>
  select(IDpol, ClaimNb, TotalClaimAmount, HasClaim, sev_pred)

cat(sprintf("\n--- Severity predictions (all policies) ---\n"))
cat(sprintf("Rows: %d\n", nrow(sev_preds_all)))
cat(sprintf("Mean predicted severity (all policies): €%.0f\n",
            mean(sev_preds_all$sev_pred)))
cat(sprintf("Mean predicted severity (claims only):  €%.0f\n",
            mean(sev_preds_all$sev_pred[sev_preds_all$HasClaim])))


# =============================================================================
# 10. SAVE MODEL AND PREDICTIONS
# =============================================================================

saveRDS(m_sev_final, "outputs/models/sev_model.rds")
cat("\nSeverity model saved to outputs/models/sev_model.rds\n")

saveRDS(sev_preds_all, "outputs/data/sev_predictions.rds")
cat("Severity predictions saved to outputs/data/sev_predictions.rds\n")
cat("\nSeverity modelling complete.\n")