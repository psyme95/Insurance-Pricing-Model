# =============================================================================
# 02_frequency_model.R
# Claim Frequency Modelling — Poisson GLM
#
# Objectives:
#   1. Load cleaned data from 01_eda.R
#   2. Fit a baseline Poisson GLM with log(Exposure) offset
#   3. Check for overdispersion and fit quasi-Poisson if needed
#   4. Refine via likelihood ratio tests
#   5. Diagnostic plots
#   6. Generate and save frequency predictions for all policies
#
# Key modelling principle:
#   Claim counts follow a Poisson process scaled by exposure time.
#   log(Exposure) enters as an OFFSET (coefficient fixed at 1), not a predictor.
#   This ensures we model the RATE (claims per policy-year), not raw counts.
#
# Output files:
#   outputs/models/freq_model.rds          — final model object
#   outputs/data/freq_predictions.rds      — predicted frequencies for all policies
#   outputs/figures/freq_*.png             — diagnostic figures
#   outputs/tables/freq_model_summary.csv  — coefficient table
# =============================================================================

library(tidyverse)
library(patchwork)
library(scales)
library(broom)

dir.create("outputs/models",  recursive = TRUE, showWarnings = FALSE)

# Set common plot theme
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
# 1. LOAD DATA
# =============================================================================

dat <- readRDS("outputs/data/model_data.rds")

cat("Loaded:", nrow(dat), "policies\n")
cat("Total claims:", sum(dat$ClaimNb), "\n")
cat("Total exposure:", round(sum(dat$Exposure), 0), "policy-years\n")

# Feature preparation
# BonusMalus: log-transform to compress the long right tail
# DrivAge, VehAge, VehPower: keep numeric; spline terms tested below
# Region, VehBrand, VehGas, Area: factors — confirm levels

dat <- dat |>
  mutate(
    log_BonusMalus = log(BonusMalus),
    Region         = fct_infreq(Region),    # most frequent level = reference
    VehBrand       = fct_infreq(VehBrand),
    Area           = fct_relevel(Area, "A") # Area A = reference (rural, lowest density)
  )

cat("\nRegion levels (reference =", levels(dat$Region)[1], ")\n")
cat("VehBrand levels (reference =", levels(dat$VehBrand)[1], ")\n")


# =============================================================================
# 2. BASELINE POISSON GLM
# =============================================================================
# offset = log(Exposure) fixes its coefficient to 1 on the log scale.
# This models log(E[ClaimNb] / Exposure) as a linear function of predictors,
# i.e. we are modelling the RATE, not the count.

cat("\n--- Fitting baseline Poisson GLM ---\n")

m_pois_base <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge +
    VehAge +
    VehPower +
    VehGas +
    Area +
    Region,
  family = poisson(link = "log"),
  data = dat
)

cat("Baseline model fitted.\n")
print(summary(m_pois_base))


# =============================================================================
# 3. OVERDISPERSION CHECK
# =============================================================================
# The Poisson assumption requires Var(Y) = E(Y).
# In practice, claim counts are often overdispersed: Var(Y) > E(Y).
# Overdispersion doesn't bias coefficient estimates but underestimates SEs,
# leading to spuriously significant predictors.
#
# Simple check: residual deviance / residual df >> 1 indicates overdispersion.
# Formal test: compare Poisson vs quasi-Poisson dispersion parameter.

dispersion_ratio <- m_pois_base$deviance / m_pois_base$df.residual
cat(sprintf("\n--- Overdispersion check ---\n"))
cat(sprintf("Residual deviance: %.1f\n", m_pois_base$deviance))
cat(sprintf("Residual df:       %d\n",   m_pois_base$df.residual))
cat(sprintf("Dispersion ratio:  %.4f\n", dispersion_ratio))

if (dispersion_ratio > 1.1) {
  cat("Dispersion ratio > 1.1: overdispersion present.\n")
  cat("Fitting quasi-Poisson to obtain corrected standard errors.\n")
  USE_QUASI <- TRUE
} else {
  cat("Dispersion ratio close to 1: Poisson assumption reasonable.\n")
  USE_QUASI <- FALSE
}

# Fit quasi-Poisson regardless for comparison
m_qpois_base <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge +
    VehAge +
    VehPower +
    VehGas +
    Area +
    Region,
  family  = quasipoisson(link = "log"),
  data    = dat
)

cat("\n--- Quasi-Poisson dispersion parameter ---\n")
cat(sprintf("Estimated dispersion: %.4f\n",
            summary(m_qpois_base)$dispersion))

# Select working model based on overdispersion check
m_freq_working <- if (USE_QUASI) m_qpois_base else m_pois_base
cat(sprintf("\nWorking model: %s\n",
            if (USE_QUASI) "quasi-Poisson" else "Poisson"))


# =============================================================================
# 4. NON-LINEAR TERMS FOR AGE VARIABLES
# =============================================================================
# DrivAge typically has a U-shaped relationship with claim frequency:
# young and very old drivers have higher rates. A linear term misses this.
# VehAge may similarly have a non-linear relationship.
# Test quadratic terms via likelihood ratio tests.

# Note: LRT not valid for quasi-Poisson (no likelihood). Use Poisson for LRT,
# then apply quasi-Poisson to the chosen structure.

cat("\n--- Testing non-linear terms (LRT on Poisson) ---\n")

# DrivAge quadratic
m_drivage_quad <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge +
    VehPower +
    VehGas +
    Area +
    Region,
  family = poisson(link = "log"),
  data   = dat
)

lrt_drivage <- anova(m_pois_base, m_drivage_quad, test = "LRT")
cat("\nLRT: DrivAge linear vs quadratic\n")
print(lrt_drivage)

# VehAge quadratic
m_vehage_quad <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge +
    VehAge + I(VehAge^2) +
    VehPower +
    VehGas +
    Area +
    Region,
  family = poisson(link = "log"),
  data   = dat
)

lrt_vehage <- anova(m_pois_base, m_vehage_quad, test = "LRT")
cat("\nLRT: VehAge linear vs quadratic\n")
print(lrt_vehage)

# Build refined model with significant non-linear terms
# (Script will include both quadratics — remove if LRT non-significant)
m_pois_refined <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge + I(VehAge^2) +
    VehPower +
    VehGas +
    Area +
    Region,
  family = poisson(link = "log"),
  data   = dat
)

lrt_refined <- anova(m_pois_base, m_pois_refined, test = "LRT")
cat("\nLRT: baseline vs refined (both quadratics)\n")
print(lrt_refined)

cat(sprintf("\nBaseline AIC:  %.1f\n", AIC(m_pois_base)))
cat(sprintf("Refined AIC:   %.1f\n",  AIC(m_pois_refined)))


# =============================================================================
# 5. FINAL MODEL — QUASI-POISSON WITH REFINED STRUCTURE
# =============================================================================
# Apply the refined term structure to quasi-Poisson for correct SEs.

cat("\n--- Fitting final quasi-Poisson model ---\n")

m_freq_final <- glm(
  ClaimNb ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge + I(VehAge^2) +
    VehPower +
    VehGas +
    Area +
    log(Density),
  family  = quasipoisson(link = "log"),
  data    = dat
)

print(summary(m_freq_final))

# Coefficient table
coef_tab <- tidy(m_freq_final, conf.int = TRUE, conf.method = "Wald") |>
  mutate(
    IRR      = exp(estimate),
    IRR_low  = exp(conf.low),
    IRR_high = exp(conf.high),
    signif   = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      p.value < 0.1   ~ ".",
      TRUE            ~ ""
    )
  ) |>
  select(term, estimate, IRR, IRR_low, IRR_high, std.error, p.value, signif)

cat("\n--- Coefficient table (IRR = exp(estimate)) ---\n")
print(coef_tab, n = Inf)

write_csv(coef_tab, "outputs/tables/freq_model_summary.csv")


# =============================================================================
# 6. DIAGNOSTIC PLOTS
# =============================================================================

# --- 6a. Observed vs predicted frequency by key variables ---
# Bin continuous variables and compare mean observed vs mean predicted rate.
# This is the standard actuarial "one-way lift" diagnostic.

dat <- dat |>
  mutate(
    freq_pred = predict(m_freq_final, type = "response") / Exposure
  )

# Helper: observed vs predicted lift plot
lift_plot <- function(data, var, var_label, nbins = 10) {
  data |>
    mutate(bin = ntile(.data[[var]], nbins)) |>
    group_by(bin) |>
    summarise(
      midpoint   = mean(.data[[var]]),
      obs_freq   = sum(ClaimNb) / sum(Exposure),
      pred_freq  = sum(freq_pred * Exposure) / sum(Exposure),
      .groups    = "drop"
    ) |>
    pivot_longer(c(obs_freq, pred_freq),
                 names_to = "type", values_to = "frequency") |>
    mutate(type = recode(type,
                         "obs_freq"  = "Observed",
                         "pred_freq" = "Predicted"
    )) |>
    ggplot(aes(x = midpoint, y = frequency, colour = type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 2) +
    scale_colour_manual(values = c("Observed" = "#d6604d",
                                   "Predicted" = "#2166ac")) +
    labs(x = var_label, y = "Claims per policy-year",
         colour = NULL) +
    plot_theme() +
    theme(legend.position = "bottom")
}

p_lift_drivage <- lift_plot(dat, "DrivAge",    "Driver age")      +
  labs(title = "Observed vs Predicted Frequency — Driver Age")
p_lift_bm      <- lift_plot(dat, "BonusMalus", "Bonus-Malus score") +
  labs(title = "Observed vs Predicted Frequency — Bonus-Malus")
p_lift_vehage  <- lift_plot(dat, "VehAge",     "Vehicle age")     +
  labs(title = "Observed vs Predicted Frequency — Vehicle Age")
p_lift_vehpow  <- lift_plot(dat, "VehPower",   "Vehicle power")   +
  labs(title = "Observed vs Predicted Frequency — Vehicle Power")

p_lift_combined <- (p_lift_drivage + p_lift_bm) / (p_lift_vehage + p_lift_vehpow)

ggsave("outputs/figures/freq_lift_plots.png", p_lift_combined,
       width = 12, height = 10, dpi = 150)

# --- 6b. Deviance residuals vs fitted ---
# Should show no strong pattern if the model is well specified.

resid_df <- tibble(
  fitted    = fitted(m_freq_final),
  residuals = residuals(m_freq_final, type = "deviance")
)

p_resid <- ggplot(resid_df |> slice_sample(n = 10000),
                  aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.15, size = 0.6, colour = "#2166ac") +
  geom_hline(yintercept = 0, colour = "grey30", linetype = "dashed") +
  geom_smooth(method = "loess", se = FALSE, colour = "#d6604d",
              linewidth = 0.8) +
  scale_x_log10() +
  labs(
    title    = "Deviance Residuals vs Fitted Values",
    subtitle = "Sample of 10,000 policies. Loess smoother — should be flat at 0.",
    x        = "Fitted values (log scale)",
    y        = "Deviance residuals"
  ) +
  plot_theme()

ggsave("outputs/figures/freq_residuals.png", p_resid,
       width = 8, height = 5, dpi = 150)

# --- 6c. Coefficient plot (IRR) ---
# Visualise exponentiated coefficients (incidence rate ratios).
# IRR > 1 = higher claim rate than reference; IRR < 1 = lower.

p_coef <- coef_tab |>
  filter(term != "(Intercept)") |>
  mutate(term = fct_reorder(term, IRR)) |>
  ggplot(aes(x = IRR, xmin = IRR_low, xmax = IRR_high, y = term)) +
  geom_vline(xintercept = 1, linetype = "dashed", colour = "grey40") +
  geom_errorbar(aes(xmin = IRR_low, xmax = IRR_high), 
                height = 0.3, colour = "grey50",
                orientation = "y") +
  geom_point(size = 2.5, colour = "#2166ac") +
  labs(
    title    = "Frequency Model — Incidence Rate Ratios",
    subtitle = "IRR > 1 = higher claim rate than reference category",
    x        = "Incidence Rate Ratio (IRR)",
    y        = NULL
  ) +
  plot_theme()

ggsave("outputs/figures/freq_coefplot.png", p_coef,
       width = 9, height = 7, dpi = 150)


# =============================================================================
# 7. MODEL CALIBRATION CHECK
# =============================================================================
# Compare sum of predicted claims vs actual claims overall and by segment.
# A well-calibrated model should have sum(predicted) ≈ sum(observed).

total_obs  <- sum(dat$ClaimNb)
total_pred <- sum(predict(m_freq_final, type = "response"))

cat(sprintf("\n--- Global calibration ---\n"))
cat(sprintf("Observed claims:  %d\n", total_obs))
cat(sprintf("Predicted claims: %.1f\n", total_pred))
cat(sprintf("Ratio:            %.4f\n", total_pred / total_obs))

# Calibration by decile of predicted frequency
dat <- dat |>
  mutate(
    pred_rate  = predict(m_freq_final, type = "response") / Exposure,
    pred_decile = ntile(pred_rate, 10)
  )

calib_tab <- dat |>
  group_by(pred_decile) |>
  summarise(
    n_policies   = n(),
    exposure     = sum(Exposure),
    obs_claims   = sum(ClaimNb),
    pred_claims  = sum(predict(m_freq_final, type = "response")[pred_decile == cur_group_id()]),
    obs_rate     = obs_claims / exposure,
    pred_rate    = mean(pred_rate),
    .groups      = "drop"
  )

# Recompute cleanly — the nested predict above is fragile
calib_tab <- dat |>
  mutate(pred_claims_i = predict(m_freq_final, type = "response")) |>
  group_by(pred_decile) |>
  summarise(
    n_policies  = n(),
    exposure    = sum(Exposure),
    obs_claims  = sum(ClaimNb),
    pred_claims = sum(pred_claims_i),
    obs_rate    = obs_claims / exposure,
    pred_rate   = pred_claims / exposure,
    .groups     = "drop"
  )

cat("\n--- Calibration by predicted frequency decile ---\n")
print(calib_tab)

p_calib <- calib_tab |>
  pivot_longer(c(obs_rate, pred_rate),
               names_to = "type", values_to = "rate") |>
  mutate(type = recode(type,
                       "obs_rate"  = "Observed",
                       "pred_rate" = "Predicted"
  )) |>
  ggplot(aes(x = pred_decile, y = rate, colour = type)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.5) +
  scale_x_continuous(breaks = 1:10) +
  scale_colour_manual(values = c("Observed" = "#d6604d",
                                 "Predicted" = "#2166ac")) +
  labs(
    title    = "Frequency Model Calibration by Predicted Decile",
    subtitle = "Observed vs predicted claim rate — lines should overlap if well calibrated",
    x        = "Predicted frequency decile (1 = lowest risk)",
    y        = "Claims per policy-year",
    colour   = NULL
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/freq_calibration.png", p_calib,
       width = 8, height = 5, dpi = 150)


# =============================================================================
# 8. SAVE MODEL AND PREDICTIONS
# =============================================================================

saveRDS(m_freq_final, "outputs/models/freq_model.rds")
cat("\nFrequency model saved to outputs/models/freq_model.rds\n")

freq_preds <- dat |>
  select(IDpol, Exposure, ClaimNb, HasClaim) |>
  mutate(
    freq_pred_rate   = predict(m_freq_final, type = "response") / Exposure,
    freq_pred_count  = predict(m_freq_final, type = "response")
  )

saveRDS(freq_preds, "outputs/data/freq_predictions.rds")
cat("Frequency predictions saved to outputs/data/freq_predictions.rds\n")
cat("\nFrequency modelling complete.\n")