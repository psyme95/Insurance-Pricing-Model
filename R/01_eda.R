# =============================================================================
# 01_eda.R
# Exploratory Data Analysis — French Motor MTPL Dataset
#
# Objectives:
#   1. Load and join frequency + severity tables
#   2. Understand exposure distribution
#   3. Examine claim frequency and zero-inflation
#   4. Examine claim severity distribution
#   5. Explore key risk factors vs observed frequency and severity
#   6. Construct and examine a synthetic loss ratio
#   7. Write cleaned, joined dataset to outputs/ for downstream scripts
#
# Output files:
#   outputs/data/model_data.rds       — joined, cleaned dataset
#   outputs/figures/eda_*.png         — EDA figures
#   outputs/tables/eda_summary.csv    — covariate summary table
# =============================================================================

library(tidyverse)
library(CASdatasets)
library(patchwork)
library(scales)
library(knitr)

# Consistent plot theme throughout the project
plot_theme <- function() {
  theme_minimal(base_size = 12) +
    theme(
      plot.title    = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(colour = "grey40", size = 10),
      axis.title    = element_text(size = 10),
      panel.grid.minor = element_blank(),
      plot.caption  = element_text(colour = "grey50", size = 8)
    )
}

dir.create("outputs/figures", recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/tables",  recursive = TRUE, showWarnings = FALSE)
dir.create("outputs/data",    recursive = TRUE, showWarnings = FALSE)


# =============================================================================
# 1. LOAD DATA
# =============================================================================

data(freMTPL2freq)
data(freMTPL2sev)

freq <- as_tibble(freMTPL2freq)
sev  <- as_tibble(freMTPL2sev)

cat("--- Frequency table ---\n")
glimpse(freq)

cat("\n--- Severity table ---\n")
glimpse(sev)

# Aggregate severity to policy level (total claims paid per policy)
sev_agg <- sev %>%
  group_by(IDpol) %>%
  summarise(
    TotalClaimAmount = sum(ClaimAmount),
    ClaimCount = n(),        # independent check against freq$ClaimNb
    .groups = "drop"
  )

# Join to frequency table
# Policies with no claims get TotalClaimAmount = 0
dat <- freq %>%
  left_join(sev_agg, by = "IDpol") %>%
  mutate(
    TotalClaimAmount = replace_na(TotalClaimAmount, 0),
    ClaimCount       = replace_na(ClaimCount, 0),
    # Pure premium: total claims cost normalised by exposure
    PurePremium      = TotalClaimAmount / Exposure,
    # Binary claim indicator
    HasClaim         = ClaimNb > 0
  )

cat("\n--- Joined dataset ---\n")
glimpse(dat)
cat("Policies:", nrow(dat), "\n")
cat("Policies with at least one claim:", sum(dat$HasClaim), 
    sprintf("(%.1f%%)\n", 100 * mean(dat$HasClaim)))
cat("Total exposure (policy-years):", round(sum(dat$Exposure), 0), "\n")
cat("Total claims paid: £", round(sum(dat$TotalClaimAmount), 0), "\n")


# =============================================================================
# 2. EXPOSURE DISTRIBUTION
# =============================================================================
# Exposure is the fraction of the year a policy was active.
# This is the denominator in all rate calculations.
# Most policies are full-year (Exposure = 1), but a significant tail are not.

p_exposure <- ggplot(dat, aes(x = Exposure)) +
  geom_histogram(bins = 50, fill = "#2166ac", colour = "white", linewidth = 0.2) +
  scale_x_continuous(breaks = seq(0, 1, 0.1)) +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Distribution of Policy Exposure",
    subtitle = "Exposure = fraction of year the policy was active",
    x        = "Exposure (policy-years)",
    y        = "Number of policies",
    caption  = "Source: freMTPL2freq — CASdatasets"
  ) +
  plot_theme()

ggsave("outputs/figures/eda_exposure.png", p_exposure, 
       width = 8, height = 5, dpi = 150)

cat("\nExposure summary:\n")
print(summary(dat$Exposure))


# =============================================================================
# 3. CLAIM FREQUENCY & ZERO-INFLATION
# =============================================================================
# The vast majority of policies have zero claims.
# This is expected — claim rates of 5-15% are typical for motor liability.
# Zero-inflation is the primary distributional feature the Poisson GLM must handle.

claim_tab <- dat %>%
  count(ClaimNb) %>%
  mutate(
    pct     = n / sum(n) * 100,
    cum_pct = cumsum(pct)
  )

cat("\n--- Claim count distribution ---\n")
print(claim_tab)

p_claimcount <- ggplot(claim_tab, aes(x = factor(ClaimNb), y = pct)) +
  geom_col(fill = "#2166ac", colour = "white", linewidth = 0.2) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), 
            vjust = -0.4, size = 3.2) +
  labs(
    title    = "Claim Count Distribution",
    subtitle = sprintf("%.1f%% of policies have zero claims", 
                       claim_tab$pct[claim_tab$ClaimNb == 0]),
    x        = "Number of claims",
    y        = "% of policies",
    caption  = "Source: freMTPL2freq — CASdatasets"
  ) +
  plot_theme()

ggsave("outputs/figures/eda_claimcount.png", p_claimcount,
       width = 8, height = 5, dpi = 150)

# Observed claim frequency (claims per policy-year)
obs_freq <- sum(dat$ClaimNb) / sum(dat$Exposure)
cat(sprintf("\nObserved claim frequency: %.4f claims per policy-year\n", obs_freq))


# =============================================================================
# 4. CLAIM SEVERITY DISTRIBUTION
# =============================================================================
# Severity (cost per claim) is right-skewed with a long tail.
# This justifies a Gamma or lognormal distribution for the severity model.
# Large losses (e.g. >99th percentile) can distort model fitting.

sev_claims <- dat %>% filter(HasClaim)

p_severity <- ggplot(sev_claims, aes(x = TotalClaimAmount)) +
  geom_histogram(bins = 80, fill = "#d6604d", colour = "white", linewidth = 0.2) +
  scale_x_continuous(labels = comma, limits = c(0, quantile(sev_claims$TotalClaimAmount, 0.99))) +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Claim Severity Distribution (policies with ≥1 claim)",
    subtitle = "Truncated at 99th percentile for display. Right-skew justifies Gamma GLM.",
    x        = "Total claim amount (€)",
    y        = "Number of policies",
    caption  = "Source: freMTPL2freq + freMTPL2sev — CASdatasets"
  ) +
  plot_theme()

p_severity_log <- ggplot(sev_claims, aes(x = TotalClaimAmount)) +
  geom_histogram(bins = 80, fill = "#d6604d", colour = "white", linewidth = 0.2) +
  scale_x_log10(labels = comma) +
  scale_y_continuous(labels = comma) +
  labs(
    title    = "Claim Severity — Log Scale",
    subtitle = "Log-scale reveals near-lognormal shape",
    x        = "Total claim amount (€, log scale)",
    y        = "Number of policies"
  ) +
  plot_theme()

p_sev_combined <- patchwork::wrap_plots(p_severity, p_severity_log, ncol = 1)

ggsave("outputs/figures/eda_severity.png", p_sev_combined,
       width = 8, height = 8, dpi = 150)

cat("\nSeverity summary (policies with claims only):\n")
print(summary(sev_claims$TotalClaimAmount))
cat(sprintf("99th percentile: €%.0f\n", 
            quantile(sev_claims$TotalClaimAmount, 0.99)))
cat(sprintf("Large loss threshold (>99th pct) accounts for %.1f%% of total paid loss\n",
            100 * sum(sev_claims$TotalClaimAmount[sev_claims$TotalClaimAmount > 
                                                    quantile(sev_claims$TotalClaimAmount, 0.99)]) / 
              sum(sev_claims$TotalClaimAmount)))


# =============================================================================
# 5. RISK FACTOR EXPLORATION
# =============================================================================
# For each key covariate, compute observed claim frequency and mean severity.
# This mirrors the one-way analyses that pricing actuaries produce before modelling.
# Patterns here motivate feature inclusion and transformation decisions.

# Helper: one-way frequency summary
one_way_freq <- function(data, var) {
  data %>%
    group_by(across(all_of(var))) %>%
    summarise(
      policies      = n(),
      exposure      = sum(Exposure),
      claims        = sum(ClaimNb),
      obs_frequency = claims / exposure,
      .groups = "drop"
    )
}

# Helper: one-way severity summary (claims-only)
one_way_sev <- function(data, var) {
  data %>%
    filter(HasClaim) %>%
    group_by(across(all_of(var))) %>%
    summarise(
      claims     = n(),
      mean_sev   = mean(TotalClaimAmount),
      median_sev = median(TotalClaimAmount),
      .groups = "drop"
    )
}

# --- 5a. Driver Age ---
dat <- dat %>%
  mutate(DrivAgeBand = cut(DrivAge,
                           breaks = c(17, 22, 26, 30, 40, 50, 60, 70, Inf),
                           labels = c("18-22", "23-26", "27-30", "31-40", "41-50", "51-60", "61-70", "71+"),
                           right  = TRUE
  ))

freq_age <- one_way_freq(dat, "DrivAgeBand")
sev_age  <- one_way_sev(dat, "DrivAgeBand")

p_age_freq <- ggplot(freq_age, aes(x = DrivAgeBand, y = obs_frequency)) +
  geom_col(fill = "#4393c3") +
  geom_hline(yintercept = obs_freq, linetype = "dashed", colour = "grey40") +
  labs(title = "Claim Frequency by Driver Age",
       subtitle = "Dashed line = portfolio average",
       x = "Driver age band", y = "Claims per policy-year") +
  plot_theme()

p_age_sev <- ggplot(sev_age, aes(x = DrivAgeBand, y = mean_sev)) +
  geom_col(fill = "#d6604d") +
  labs(title = "Mean Severity by Driver Age",
       x = "Driver age band", y = "Mean claim amount (€)") +
  plot_theme()

ggsave("outputs/figures/eda_drivage.png", p_age_freq / p_age_sev,
       width = 8, height = 8, dpi = 150)

# --- 5b. Vehicle Power ---
dat <- dat %>%
  mutate(VehPowerBand = cut(VehPower,
                            breaks = c(0, 5, 7, 9, 12, 15, Inf),
                            labels = c("≤5", "6-7", "8-9", "10-12", "13-15", "16+"),
                            right  = TRUE
  ))

freq_power <- one_way_freq(dat, "VehPowerBand")
sev_power  <- one_way_sev(dat, "VehPowerBand")

p_power_freq <- ggplot(freq_power, aes(x = VehPowerBand, y = obs_frequency)) +
  geom_col(fill = "#4393c3") +
  geom_hline(yintercept = obs_freq, linetype = "dashed", colour = "grey40") +
  labs(title = "Claim Frequency by Vehicle Power",
       x = "Vehicle power band", y = "Claims per policy-year") +
  plot_theme()

p_power_sev <- ggplot(sev_power, aes(x = VehPowerBand, y = mean_sev)) +
  geom_col(fill = "#d6604d") +
  labs(title = "Mean Severity by Vehicle Power",
       x = "Vehicle power band", y = "Mean claim amount (€)") +
  plot_theme()

ggsave("outputs/figures/eda_vehpower.png", p_power_freq / p_power_sev,
       width = 8, height = 8, dpi = 150)

# --- 5c. Bonus-Malus Score ---
# BonusMalus is an experience-rated score: low = good driver, high = bad.
# Expected to be one of the strongest predictors.
dat <- dat %>%
  mutate(BonusMalusBand = cut(BonusMalus,
                              breaks = c(0, 50, 75, 100, 125, 150, Inf),
                              labels = c("≤50", "51-75", "76-100", "101-125", "126-150", "151+"),
                              right  = TRUE
  ))

freq_bm <- one_way_freq(dat, "BonusMalusBand")

p_bm <- ggplot(freq_bm, aes(x = BonusMalusBand, y = obs_frequency)) +
  geom_col(fill = "#4393c3") +
  geom_hline(yintercept = obs_freq, linetype = "dashed", colour = "grey40") +
  labs(
    title    = "Claim Frequency by Bonus-Malus Score",
    subtitle = "Strong monotonic relationship expected — key pricing variable",
    x        = "Bonus-Malus band", y = "Claims per policy-year"
  ) +
  plot_theme()

ggsave("outputs/figures/eda_bonusmalus.png", p_bm,
       width = 8, height = 5, dpi = 150)

# --- 5d. Vehicle Age ---
dat <- dat %>%
  mutate(VehAgeBand = cut(VehAge,
                          breaks = c(-1, 1, 3, 6, 10, 15, Inf),
                          labels = c("0-1", "2-3", "4-6", "7-10", "11-15", "16+"),
                          right  = TRUE
  ))

freq_vage <- one_way_freq(dat, "VehAgeBand")
sev_vage  <- one_way_sev(dat, "VehAgeBand")

p_vage <- ggplot(freq_vage, aes(x = VehAgeBand, y = obs_frequency)) +
  geom_col(fill = "#4393c3") +
  geom_hline(yintercept = obs_freq, linetype = "dashed", colour = "grey40") +
  labs(title = "Claim Frequency by Vehicle Age",
       x = "Vehicle age band (years)", y = "Claims per policy-year") +
  plot_theme()

ggsave("outputs/figures/eda_vehage.png", p_vage,
       width = 8, height = 5, dpi = 150)

# --- 5e. Region ---
freq_region <- one_way_freq(dat, "Region") %>%
  mutate(Region = fct_reorder(Region, obs_frequency))

p_region <- ggplot(freq_region, aes(x = Region, y = obs_frequency)) +
  geom_col(fill = "#4393c3") +
  geom_hline(yintercept = obs_freq, linetype = "dashed", colour = "grey40") +
  coord_flip() +
  labs(
    title    = "Claim Frequency by Region",
    subtitle = "Spatial risk variation — candidate for spatial random effect in full model",
    x        = NULL, y = "Claims per policy-year"
  ) +
  plot_theme()

ggsave("outputs/figures/eda_region.png", p_region,
       width = 8, height = 6, dpi = 150)


# =============================================================================
# 6. SYNTHETIC LOSS RATIO
# =============================================================================
# In practice, loss ratio = claims paid / premium earned.
# We don't have actual premiums, so we construct a simple risk-relativities-based
# synthetic premium using BonusMalus as a proxy for rate adequacy.
# This is illustrative — it allows us to frame the problem in pricing language.

# Simple synthetic premium: base rate × BonusMalus relativity × Exposure
BASE_RATE <- 300  # €300 base pure premium (illustrative)

dat <- dat %>%
  mutate(
    BM_relativity     = BonusMalus / 100,
    SyntheticPremium  = BASE_RATE * BM_relativity * Exposure,
    LossRatio         = TotalClaimAmount / SyntheticPremium
  )

# Loss ratio by region
lr_region <- dat %>%
  group_by(Region) %>%
  summarise(
    total_claims  = sum(TotalClaimAmount),
    total_premium = sum(SyntheticPremium),
    loss_ratio    = total_claims / total_premium,
    .groups = "drop"
  ) %>%
  mutate(Region = fct_reorder(Region, loss_ratio))

p_lr_region <- ggplot(lr_region, aes(x = Region, y = loss_ratio,
                                     fill = loss_ratio > 1)) +
  geom_col() +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "grey30") +
  scale_fill_manual(values = c("FALSE" = "#4393c3", "TRUE" = "#d6604d"),
                    guide = "none") +
  scale_y_continuous(labels = percent) +
  coord_flip() +
  labs(
    title    = "Synthetic Loss Ratio by Region",
    subtitle = "Red bars = loss ratio >100% (underpriced segments under synthetic premium)\nNote: premium is synthetic — for illustration only",
    x        = NULL, y = "Loss ratio"
  ) +
  plot_theme()

ggsave("outputs/figures/eda_lossratio_region.png", p_lr_region,
       width = 8, height = 6, dpi = 150)

overall_lr <- sum(dat$TotalClaimAmount) / sum(dat$SyntheticPremium)
cat(sprintf("\nOverall synthetic loss ratio: %.1f%%\n", 100 * overall_lr))


# =============================================================================
# 7. COVARIATE CORRELATION / MULTICOLLINEARITY CHECK
# =============================================================================
# Check for multicollinearity among numeric predictors before modelling.
# High correlation between DrivAge and BonusMalus is plausible (older = lower BM).

num_vars <- dat %>%
  select(DrivAge, VehAge, VehPower, BonusMalus, Exposure) %>%
  cor(use = "complete.obs")

cat("\n--- Correlation matrix (numeric predictors) ---\n")
print(round(num_vars, 3))


# =============================================================================
# 8. SUMMARY TABLE
# =============================================================================

summary_tab <- dat %>%
  summarise(
    n_policies       = n(),
    total_exposure   = sum(Exposure),
    n_claims         = sum(ClaimNb),
    pct_with_claim   = mean(HasClaim) * 100,
    obs_freq         = sum(ClaimNb) / sum(Exposure),
    mean_severity    = mean(TotalClaimAmount[HasClaim]),
    median_severity  = median(TotalClaimAmount[HasClaim]),
    total_loss       = sum(TotalClaimAmount),
    mean_pure_prem   = sum(TotalClaimAmount) / sum(Exposure)
  )

write_csv(t(summary_tab) %>% as.data.frame(), "outputs/tables/eda_summary.csv")

cat("\n--- Portfolio summary ---\n")
print(t(summary_tab))


# =============================================================================
# 9. SAVE CLEANED DATASET
# =============================================================================

saveRDS(dat, "outputs/data/model_data.rds")
cat("\nCleaned dataset saved to outputs/data/model_data.rds\n")
cat("Rows:", nrow(dat), "| Columns:", ncol(dat), "\n")
cat("\nEDA complete. Figures written to outputs/figures/\n")