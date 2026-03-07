# =============================================================================
# 04_pure_premium.R
# Pure Premium Assembly — Frequency × Severity
#
# Objectives:
#   1. Load frequency and severity predictions
#   2. Assemble pure premium = predicted frequency rate × predicted severity
#   3. Apply large loss loading to account for claims above the cap threshold
#   4. Compare against Tweedie GLM single-model baseline
#   5. Produce portfolio-level and segment-level diagnostics
#   6. Save combined predictions for evaluation in script 06
#
# Key modelling principles:
#
#   Pure Premium = E[frequency] × E[severity]
#                = (predicted claims per policy-year) × (predicted cost per claim)
#
#   This multiplicative assembly is valid under the assumption that frequency
#   and severity are INDEPENDENT conditional on the covariates. This is a
#   standard actuarial assumption — violations (e.g. high-risk drivers also
#   tend to have more expensive claims) would require a joint model.
#
#   Large loss treatment:
#   The severity model was fitted on claims capped at the 99th percentile
#   (€16,327). This produces stable Gamma GLM estimates for the attritional
#   loss layer but systematically understates total pure premium, since
#   29.8% of total paid loss sits above the cap.
#
#   A multiplicative large loss loading is applied to restore the full
#   pure premium:
#
#     loading = 1 / (1 - 0.298) = 1.424
#
#   In production, the large loss layer would be modelled separately
#   using a Pareto or other heavy-tailed distribution and combined with
#   the attritional model. The loading used here is a simplification
#   consistent with the data and documented as a limitation.
#
# Output files:
#   outputs/data/pure_premium.rds          — full prediction dataset
#   outputs/models/tweedie_model.rds       — Tweedie GLM baseline
#   outputs/figures/pp_*.png               — pure premium figures
#   outputs/tables/pp_summary.csv          — segment summary table
# =============================================================================

library(tidyverse)
library(patchwork)
library(scales)
library(statmod)
library(tweedie)

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
# 1. LOAD DATA AND PREDICTIONS
# =============================================================================

dat        <- readRDS("outputs/data/model_data.rds")
freq_preds <- readRDS("outputs/data/freq_predictions.rds")
sev_preds  <- readRDS("outputs/data/sev_predictions.rds")

m_freq <- readRDS("outputs/models/freq_model.rds")
m_sev  <- readRDS("outputs/models/sev_model.rds")

cat("Policies:", nrow(dat), "\n")
cat("Freq predictions:", nrow(freq_preds), "\n")
cat("Sev predictions:", nrow(sev_preds), "\n")

# Join predictions to main dataset
dat <- dat |>
  mutate(log_Density = log(Density)) |>
  left_join(
    freq_preds |> select(IDpol, freq_pred_rate, freq_pred_count),
    by = "IDpol"
  ) |>
  left_join(
    sev_preds |> select(IDpol, sev_pred),
    by = "IDpol"
  )


# =============================================================================
# 2. LARGE LOSS LOADING
# =============================================================================
# The severity model was fitted on capped severity (99th percentile = €16,327).
# From script 03: 29.8% of total paid loss sits above the cap threshold.
# A multiplicative loading restores the full pure premium estimate.
#
# loading = 1 / (1 - 0.298) = 1.424
#
# This is applied uniformly across all policies. In production, the loading
# would vary by risk segment if large loss propensity differs by covariate —
# e.g. young drivers may have a higher large loss share than older drivers.
# Uniform loading is a simplification noted as a limitation.

LARGE_LOSS_PCT     <- 0.298
LARGE_LOSS_LOADING <- 1 / (1 - LARGE_LOSS_PCT)

cat(sprintf("\n--- Large loss loading ---\n"))
cat(sprintf("Loss above 99th pct cap:    %.1f%%\n", 100 * LARGE_LOSS_PCT))
cat(sprintf("Loading factor:             %.4f\n", LARGE_LOSS_LOADING))


# =============================================================================
# 3. ASSEMBLE PURE PREMIUM
# =============================================================================

dat <- dat |>
  mutate(
    pp_attritional = freq_pred_rate * sev_pred,
    pp_freq_sev    = pp_attritional * LARGE_LOSS_LOADING,
    obs_pure_prem  = TotalClaimAmount / Exposure
  )

cat("\n--- Pure premium summary ---\n")
cat(sprintf("Mean predicted PP (attritional only): €%.2f\n",
            weighted.mean(dat$pp_attritional, dat$Exposure)))
cat(sprintf("Mean predicted PP (loaded):           €%.2f\n",
            weighted.mean(dat$pp_freq_sev, dat$Exposure)))
cat(sprintf("Mean observed PP:                     €%.2f\n",
            weighted.mean(dat$obs_pure_prem, dat$Exposure)))
cat(sprintf("Ratio loaded/observed:                %.4f\n",
            weighted.mean(dat$pp_freq_sev, dat$Exposure) /
              weighted.mean(dat$obs_pure_prem, dat$Exposure)))


# =============================================================================
# 4. TWEEDIE GLM BASELINE
# =============================================================================
# The Tweedie distribution is a compound Poisson-Gamma that models pure
# premium directly, handling the mixture of zeros (no claim) and positive
# values (claim occurred) in a single model.
#
# The power parameter p (1 < p < 2) governs the zero mass:
# p = 1 → Poisson; p = 2 → Gamma.
# For motor insurance, p ≈ 1.5–1.8 is typical.
#
# We fix p = 1.5 as a reasonable starting point for motor liability.
# Profile likelihood estimation of p is noted as a refinement for future work.

dat_tw <- dat |>
  mutate(
    log_BonusMalus = log(BonusMalus),
    Area           = fct_relevel(Area, "A")
  )

p_opt <- 1.5
cat(sprintf("\n--- Tweedie GLM ---\n"))
cat(sprintf("Power parameter p fixed at %.1f (typical for motor insurance)\n", p_opt))
cat("Note: profile likelihood estimation of p is a recommended refinement.\n")
cat("Fitting Tweedie GLM...\n")

m_tweedie <- glm(
  TotalClaimAmount ~ offset(log(Exposure)) +
    log_BonusMalus +
    DrivAge + I(DrivAge^2) +
    VehAge +
    VehPower +
    VehGas +
    Area +
    log_Density,
  family = tweedie(var.power = p_opt, link.power = 0),
  data   = dat_tw
)

cat("Tweedie GLM fitted.\n")
print(summary(m_tweedie))

dat <- dat |>
  mutate(
    pp_tweedie = predict(m_tweedie, type = "response") / dat_tw$Exposure
  )

cat(sprintf("\nMean predicted PP (Tweedie):          €%.2f\n",
            weighted.mean(dat$pp_tweedie, dat$Exposure)))
cat(sprintf("Ratio Tweedie/observed:               %.4f\n",
            weighted.mean(dat$pp_tweedie, dat$Exposure) /
              weighted.mean(dat$obs_pure_prem, dat$Exposure)))

saveRDS(m_tweedie, "outputs/models/tweedie_model.rds")


# =============================================================================
# 5. PURE PREMIUM DISTRIBUTION
# =============================================================================

pp_long <- dat |>
  select(IDpol, Exposure, pp_freq_sev, pp_tweedie) |>
  pivot_longer(c(pp_freq_sev, pp_tweedie),
               names_to = "model", values_to = "pure_premium") |>
  mutate(model = recode(model,
                        "pp_freq_sev" = "Freq-Sev GLM (loaded)",
                        "pp_tweedie"  = "Tweedie GLM"
  ))

p99_pp <- quantile(dat$pp_freq_sev, 0.99)

p_pp_dist <- ggplot(pp_long,
                    aes(x = pure_premium, colour = model, fill = model)) +
  geom_density(alpha = 0.15, linewidth = 0.9) +
  scale_x_continuous(labels = comma, limits = c(0, p99_pp)) +
  scale_colour_manual(values = c("Freq-Sev GLM (loaded)" = "#2166ac",
                                 "Tweedie GLM"           = "#d6604d")) +
  scale_fill_manual(values   = c("Freq-Sev GLM (loaded)" = "#2166ac",
                                 "Tweedie GLM"           = "#d6604d")) +
  labs(
    title    = "Distribution of Predicted Pure Premium",
    subtitle = "Truncated at 99th percentile. Freq-Sev includes large loss loading.",
    x        = "Predicted pure premium (€ per policy-year)",
    y        = "Density",
    colour   = NULL, fill = NULL
  ) +
  plot_theme() +
  theme(legend.position = "bottom")

ggsave("outputs/figures/pp_distribution.png", p_pp_dist,
       width = 8, height = 5, dpi = 150)


# =============================================================================
# 6. RISK SEGMENTATION — ONE-WAY PLOTS
# =============================================================================

pp_oneway <- function(data, var, var_label, nbins = 10) {
  data |>
    mutate(bin = ntile(.data[[var]], nbins)) |>
    group_by(bin) |>
    summarise(
      midpoint     = mean(.data[[var]]),
      obs_pp       = sum(TotalClaimAmount) / sum(Exposure),
      pred_freqsev = weighted.mean(pp_freq_sev, Exposure),
      pred_tweedie = weighted.mean(pp_tweedie, Exposure),
      .groups      = "drop"
    ) |>
    pivot_longer(c(obs_pp, pred_freqsev, pred_tweedie),
                 names_to = "type", values_to = "pp") |>
    mutate(type = recode(type,
                         "obs_pp"       = "Observed",
                         "pred_freqsev" = "Freq-Sev GLM (loaded)",
                         "pred_tweedie" = "Tweedie GLM"
    )) |>
    ggplot(aes(x = midpoint, y = pp, colour = type, linetype = type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.8) +
    scale_y_continuous(labels = comma) +
    scale_colour_manual(values = c(
      "Observed"              = "grey40",
      "Freq-Sev GLM (loaded)" = "#2166ac",
      "Tweedie GLM"           = "#d6604d"
    )) +
    scale_linetype_manual(values = c(
      "Observed"              = "dotted",
      "Freq-Sev GLM (loaded)" = "solid",
      "Tweedie GLM"           = "dashed"
    )) +
    labs(x = var_label, y = "Pure premium (€/year)",
         colour = NULL, linetype = NULL) +
    plot_theme() +
    theme(legend.position = "bottom")
}

p_pp_drivage <- pp_oneway(dat, "DrivAge",    "Driver age") +
  labs(title = "Pure Premium by Driver Age")
p_pp_bm      <- pp_oneway(dat, "BonusMalus", "Bonus-Malus score") +
  labs(title = "Pure Premium by Bonus-Malus")
p_pp_vehage  <- pp_oneway(dat, "VehAge",     "Vehicle age") +
  labs(title = "Pure Premium by Vehicle Age")
p_pp_density <- pp_oneway(dat, "Density",    "Population density") +
  labs(title = "Pure Premium by Population Density")

p_pp_oneway <- (p_pp_drivage + p_pp_bm) / (p_pp_vehage + p_pp_density)

ggsave("outputs/figures/pp_oneway.png", p_pp_oneway,
       width = 12, height = 10, dpi = 150)


# =============================================================================
# 7. SEGMENT SUMMARY TABLE
# =============================================================================

seg_summary <- function(data, var) {
  data |>
    group_by(across(all_of(var))) |>
    summarise(
      n_policies = n(),
      exposure   = round(sum(Exposure), 1),
      obs_pp     = round(sum(TotalClaimAmount) / sum(Exposure), 2),
      pred_pp_fs = round(weighted.mean(pp_freq_sev, Exposure), 2),
      pred_pp_tw = round(weighted.mean(pp_tweedie, Exposure), 2),
      loss_ratio = round(sum(TotalClaimAmount) /
                           sum(pp_freq_sev * Exposure), 3),
      .groups    = "drop"
    )
}

seg_area   <- seg_summary(dat, "Area")
seg_vehgas <- seg_summary(dat, "VehGas")

cat("\n--- Segment summary: Area ---\n")
print(seg_area)

cat("\n--- Segment summary: VehGas ---\n")
print(seg_vehgas)

seg_combined <- bind_rows(
  seg_area   |> mutate(segment_var = "Area")   |> rename(level = Area),
  seg_vehgas |> mutate(segment_var = "VehGas") |> rename(level = VehGas)
) |>
  relocate(segment_var, level)

write_csv(seg_combined, "outputs/tables/pp_summary.csv")


# =============================================================================
# 8. MODEL COMPARISON SCATTER
# =============================================================================

p_scatter <- dat |>
  slice_sample(n = 20000) |>
  ggplot(aes(x = pp_tweedie, y = pp_freq_sev)) +
  geom_point(alpha = 0.08, size = 0.6, colour = "#2166ac") +
  geom_abline(slope = 1, intercept = 0,
              colour = "#d6604d", linetype = "dashed", linewidth = 0.8) +
  scale_x_continuous(labels = comma,
                     limits = c(0, quantile(dat$pp_tweedie, 0.995))) +
  scale_y_continuous(labels = comma,
                     limits = c(0, quantile(dat$pp_freq_sev, 0.995))) +
  labs(
    title    = "Freq-Sev GLM vs Tweedie GLM — Predicted Pure Premium",
    subtitle = "Sample of 20,000 policies. Dashed line = perfect agreement.",
    x        = "Tweedie GLM prediction (€/year)",
    y        = "Freq-Sev GLM prediction (€/year, loaded)"
  ) +
  plot_theme()

ggsave("outputs/figures/pp_model_scatter.png", p_scatter,
       width = 7, height = 7, dpi = 150)

cor_pp <- cor(dat$pp_freq_sev, dat$pp_tweedie, method = "spearman")
cat(sprintf("\nSpearman correlation (freq-sev vs Tweedie): %.4f\n", cor_pp))


# =============================================================================
# 9. INDEPENDENCE ASSUMPTION CHECK
# =============================================================================
# Checks whether observed severity varies systematically across
# predicted frequency quintiles. A flat pattern supports independence.
# A monotonic pattern suggests the independence assumption is violated.

indep_check <- dat |>
  filter(HasClaim) |>
  mutate(freq_decile = ntile(freq_pred_rate, 5)) |>
  group_by(freq_decile) |>
  summarise(
    n_claims       = sum(ClaimNb),
    mean_freq_pred = round(mean(freq_pred_rate), 4),
    obs_sev        = round(sum(TotalClaimAmount) / sum(ClaimNb), 2),
    pred_sev       = round(weighted.mean(sev_pred, ClaimNb), 2),
    .groups        = "drop"
  )

cat("\n--- Independence assumption check ---\n")
cat("Observed severity by predicted frequency quintile:\n")
cat("(Flat obs_sev = independence holds; monotonic trend = violation)\n")
print(indep_check)


# =============================================================================
# 10. SAVE COMBINED PREDICTIONS
# =============================================================================

pure_premium <- dat |>
  select(IDpol, Exposure, ClaimNb, TotalClaimAmount, HasClaim,
         freq_pred_rate, freq_pred_count, sev_pred,
         pp_attritional, pp_freq_sev, pp_tweedie, obs_pure_prem)

saveRDS(pure_premium, "outputs/data/pure_premium.rds")
cat("\nPure premium predictions saved to outputs/data/pure_premium.rds\n")
cat("Rows:", nrow(pure_premium), "\n")
cat("\nPure premium assembly complete.\n")