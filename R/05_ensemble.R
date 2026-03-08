# =============================================================================
# 05_ensemble.R
# XGBoost Tweedie Ensemble — Pure Premium Modelling
#
# Objectives:
#   1. Load cleaned data and GLM pure premium predictions
#   2. Prepare feature matrix for XGBoost
#   3. Tune XGBoost (Tweedie objective) via cross-validation
#   4. Fit final XGBoost model
#   5. SHAP values — global importance and example waterfall plots
#   6. Save XGBoost predictions for evaluation in script 06
#
# Modelling notes:
#   - XGBoost with reg:tweedie objective directly models pure premium
#     (TotalClaimAmount / Exposure) without requiring freq-sev separation.
#   - The response is capped at the 99th percentile of per-claim severity
#     (matching the severity model cap in script 03, ~€16,327) so that the
#     comparison with the freq-sev GLM is on equal terms — both models predict
#     the attritional loss layer. The large loss loading from script 04 is
#     applied to both models in script 06.
#   - Pure premium is NOT capped directly (dividing by small exposure values
#     produces very large per-policy-year figures that do not reflect true
#     large loss events). Instead, TotalClaimAmount is capped at the severity
#     threshold before dividing by exposure.
#   - Exposure enters as a weight (policy-years), not an offset.
#     XGBoost does not support offsets natively; weighting by exposure
#     is the standard approximation.
#   - CV scores were numerically unstable (Inf) due to extreme zero-inflation
#     in the pure premium response (96.3% zeros). Early stopping rounds are
#     used as the primary selection criterion instead of CV score.
#   - SHAP values via {shapviz} provide consistent, additive feature
#     attribution — preferred over gain-based importance for correlated
#     features (e.g. DrivAge and BonusMalus).
#
# Output files:
#   outputs/models/xgb_model.rds           — final XGBoost model
#   outputs/data/xgb_predictions.rds       — XGBoost predictions
#   outputs/figures/xgb_*.png              — SHAP and diagnostic figures
#   outputs/tables/xgb_cv_results.csv      — CV tuning results
# =============================================================================

library(tidyverse)
library(xgboost)
library(shapviz)
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
# 1. LOAD AND PREPARE DATA
# =============================================================================

dat    <- readRDS("outputs/data/model_data.rds")
pp_dat <- readRDS("outputs/data/pure_premium.rds")

dat <- dat |>
  left_join(pp_dat |> select(IDpol, pp_freq_sev, pp_tweedie),
            by = "IDpol")

cat("Policies:", nrow(dat), "\n")

# Categorical encoding for XGBoost numeric matrix
brand_counts <- dat |> count(VehBrand) |>
  mutate(keep = n / sum(n) >= 0.01)

dat <- dat |>
  mutate(
    VehBrand_collapsed = if_else(
      VehBrand %in% brand_counts$VehBrand[brand_counts$keep],
      as.character(VehBrand), "Other"
    ),
    VehGas_num     = as.integer(VehGas == "Regular"),
    Area_num       = as.integer(Area),
    VehBrand_num   = as.integer(factor(VehBrand_collapsed)),
    log_BonusMalus = log(BonusMalus),
    log_Density    = log(Density),
    PurePremium    = TotalClaimAmount / Exposure
  )


# =============================================================================
# 2. CAP RESPONSE — SEVERITY SCALE (MATCHING SCRIPT 03)
# =============================================================================
# Cap is applied to TotalClaimAmount (on the claim cost scale), NOT to
# PurePremium directly. Dividing TotalClaimAmount by small Exposure values
# produces very large per-policy-year figures that overstate true large loss
# events and distort the 99th percentile threshold.
#
# Threshold is recomputed from per-claim severity to match script 03 exactly.
# PurePremium_capped is then derived from the capped claim amount.

CAP_THRESHOLD_SEV <- quantile(
  dat$TotalClaimAmount[dat$HasClaim] / dat$ClaimNb[dat$HasClaim],
  0.99
)

cat(sprintf("\n--- Response capping ---\n"))
cat(sprintf("Severity cap threshold (matches script 03): €%.0f\n", CAP_THRESHOLD_SEV))

dat <- dat |>
  mutate(
    TotalClaim_capped  = pmin(TotalClaimAmount, CAP_THRESHOLD_SEV * ClaimNb),
    PurePremium_capped = TotalClaim_capped / Exposure
  )

cat(sprintf("Policies above cap: %d (%.2f%%)\n",
            sum(dat$TotalClaimAmount > CAP_THRESHOLD_SEV * pmax(dat$ClaimNb, 1)),
            100 * mean(dat$TotalClaimAmount > CAP_THRESHOLD_SEV * pmax(dat$ClaimNb, 1))))
cat(sprintf("Mean PP uncapped: €%.2f | Mean PP capped: €%.2f\n",
            weighted.mean(dat$PurePremium, dat$Exposure),
            weighted.mean(dat$PurePremium_capped, dat$Exposure)))


# =============================================================================
# 3. TRAIN / TEST SPLIT AND DMATRIX
# =============================================================================

features <- c(
  "log_BonusMalus",
  "DrivAge",
  "VehAge",
  "VehPower",
  "VehGas_num",
  "Area_num",
  "VehBrand_num",
  "log_Density",
  "Density"
)

cat("\nFeatures:", paste(features, collapse = ", "), "\n")

set.seed(42)
train_idx <- sample(nrow(dat), size = floor(0.8 * nrow(dat)))
test_idx  <- setdiff(seq_len(nrow(dat)), train_idx)
dat_train <- dat[train_idx, ]
dat_test  <- dat[test_idx, ]

saveRDS(test_idx, "outputs/data/test_idx.rds")

cat(sprintf("Train: %d policies (%.1f%% claim rate)\n",
            nrow(dat_train), 100 * mean(dat_train$HasClaim)))
cat(sprintf("Test:  %d policies (%.1f%% claim rate)\n",
            nrow(dat_test), 100 * mean(dat_test$HasClaim)))

dtrain <- xgb.DMatrix(
  data   = as.matrix(dat_train[, features]),
  label  = dat_train$PurePremium_capped,
  weight = dat_train$Exposure
)

dtest <- xgb.DMatrix(
  data   = as.matrix(dat_test[, features]),
  label  = dat_test$PurePremium_capped,
  weight = dat_test$Exposure
)

dall <- xgb.DMatrix(
  data   = as.matrix(dat[, features]),
  label  = dat$PurePremium_capped,
  weight = dat$Exposure
)


# =============================================================================
# 4. CROSS-VALIDATED TUNING
# =============================================================================
# Grid search over max_depth, eta, and tweedie_variance_power.
# CV scores may still be Inf due to zero-inflation in the response —
# early stopping rounds are used as the primary selection criterion if so.

cat("\n--- Hyperparameter tuning via CV ---\n")

param_grid <- expand.grid(
  max_depth              = c(3, 5),
  eta                    = c(0.05, 0.1),
  tweedie_variance_power = c(1.3, 1.5, 1.7),
  subsample              = 0.8,
  colsample_bytree       = 0.8,
  min_child_weight       = 10
)

cat(sprintf("Grid size: %d combinations\n", nrow(param_grid)))

cv_results <- vector("list", nrow(param_grid))

for (i in seq_len(nrow(param_grid))) {
  params <- list(
    objective              = "reg:tweedie",
    tweedie_variance_power = param_grid$tweedie_variance_power[i],
    max_depth              = param_grid$max_depth[i],
    eta                    = param_grid$eta[i],
    subsample              = param_grid$subsample[i],
    colsample_bytree       = param_grid$colsample_bytree[i],
    min_child_weight       = param_grid$min_child_weight[i],
    eval_metric            = "tweedie-nloglik@1.5"
  )
  
  cv <- xgb.cv(
    params                = params,
    data                  = dtrain,
    nrounds               = 500,
    nfold                 = 5,
    early_stopping_rounds = 30,
    verbose               = 0,
    showsd                = FALSE
  )
  
  best_round <- cv$best_iteration
  best_score <- min(cv$evaluation_log$test_tweedie_nloglik_1.5_mean)
  
  cv_results[[i]] <- tibble(
    max_depth              = param_grid$max_depth[i],
    eta                    = param_grid$eta[i],
    tweedie_variance_power = param_grid$tweedie_variance_power[i],
    best_round             = best_round,
    cv_score               = best_score
  )
  
  cat(sprintf("  [%d/%d] depth=%d eta=%.2f p=%.1f | rounds=%d score=%.6f\n",
              i, nrow(param_grid),
              param_grid$max_depth[i], param_grid$eta[i],
              param_grid$tweedie_variance_power[i],
              best_round, best_score))
}

cv_tab <- bind_rows(cv_results) |>
  arrange(cv_score)

cat("\n--- CV results (top 5) ---\n")
print(head(cv_tab, 5))
write_csv(cv_tab, "outputs/tables/xgb_cv_results.csv")

# Fallback: if all CV scores are Inf, select by longest early stopping run.
# Prefer depth=5 (more expressive) and lower eta (more robust) as tiebreakers.
if (all(is.infinite(cv_tab$cv_score))) {
  cat("\nNote: all CV scores Inf due to zero-inflation.\n")
  cat("Selecting best configuration by early stopping rounds.\n")
  best_params <- cv_tab |>
    arrange(desc(best_round), desc(max_depth), eta) |>
    dplyr::slice(1)
} else {
  best_params <- cv_tab |> dplyr::slice(1)
}

cat(sprintf("\nSelected: depth=%d eta=%.2f p=%.1f rounds=%d\n",
            best_params$max_depth, best_params$eta,
            best_params$tweedie_variance_power,
            best_params$best_round))


# =============================================================================
# 5. FINAL MODEL
# =============================================================================

cat("\n--- Fitting final XGBoost model ---\n")

final_params <- list(
  objective              = "reg:tweedie",
  tweedie_variance_power = best_params$tweedie_variance_power,
  max_depth              = best_params$max_depth,
  eta                    = best_params$eta,
  subsample              = 0.8,
  colsample_bytree       = 0.8,
  min_child_weight       = 10,
  eval_metric            = "tweedie-nloglik@1.5"
)

m_xgb <- xgb.train(
  params        = final_params,
  data          = dtrain,
  nrounds       = best_params$best_round,
  watchlist     = list(train = dtrain, test = dtest),
  verbose       = 1,
  print_every_n = 50
)

saveRDS(m_xgb, "outputs/models/xgb_model.rds")
cat("XGBoost model saved.\n")


# =============================================================================
# 6. PREDICTIONS AND CALIBRATION
# =============================================================================

dat <- dat |>
  mutate(pp_xgb = predict(m_xgb, dall))

cat(sprintf("\n--- XGBoost prediction summary ---\n"))
cat(sprintf("Mean predicted PP (XGBoost):           €%.2f\n",
            weighted.mean(dat$pp_xgb, dat$Exposure)))
cat(sprintf("Mean observed PP (capped):             €%.2f\n",
            weighted.mean(dat$PurePremium_capped, dat$Exposure)))
cat(sprintf("Mean observed PP (uncapped):           €%.2f\n",
            weighted.mean(dat$PurePremium, dat$Exposure)))
cat(sprintf("Ratio XGBoost/observed (capped):       %.4f\n",
            weighted.mean(dat$pp_xgb, dat$Exposure) /
              weighted.mean(dat$PurePremium_capped, dat$Exposure)))

# Test set calibration by decile
test_preds <- dat[-train_idx, ] |>
  mutate(
    pp_xgb      = predict(m_xgb, dtest),
    pred_decile = ntile(pp_xgb, 10)
  )

calib_xgb <- test_preds |>
  group_by(pred_decile) |>
  summarise(
    n_policies = n(),
    exposure   = sum(Exposure),
    obs_pp_cap = sum(TotalClaim_capped) / sum(Exposure),
    pred_pp    = weighted.mean(pp_xgb, Exposure),
    ratio      = obs_pp_cap / pred_pp,
    .groups    = "drop"
  )

cat("\n--- XGBoost calibration by decile (test set, capped) ---\n")
print(calib_xgb)


# =============================================================================
# 7. SHAP VALUES
# =============================================================================

cat("\n--- Computing SHAP values (sample of 5,000) ---\n")

set.seed(42)
shap_idx  <- sample(nrow(dat), 5000)
X_shap    <- as.matrix(dat[shap_idx, features])
shap_dmat <- xgb.DMatrix(data = X_shap)
shap_obj  <- shapviz(m_xgb, X_pred = X_shap, X = X_shap)

# --- 7a. Beeswarm importance ---
p_shap_bee <- sv_importance(shap_obj, kind = "beeswarm", max_display = 9) +
  labs(
    title    = "SHAP Feature Importance — XGBoost Pure Premium Model",
    subtitle = "Each point is a policy. Colour = feature value (red = high, blue = low).",
    x        = "Mean |SHAP value|"
  ) +
  plot_theme()

ggsave("outputs/figures/xgb_shap_beeswarm.png", p_shap_bee,
       width = 9, height = 6, dpi = 150)

# --- 7b. Bar importance ---
p_shap_bar <- sv_importance(shap_obj, kind = "bar", max_display = 9) +
  labs(title = "SHAP Mean Absolute Feature Importance",
       x     = "Mean |SHAP value|") +
  plot_theme()

ggsave("outputs/figures/xgb_shap_bar.png", p_shap_bar,
       width = 8, height = 5, dpi = 150)

# --- 7c. Waterfall plots: high-risk vs low-risk ---
shap_preds <- predict(m_xgb, shap_dmat)
idx_high   <- which.max(shap_preds)
idx_low    <- which.min(shap_preds)

p_waterfall_high <- sv_waterfall(shap_obj, row_id = idx_high) +
  labs(
    title    = "SHAP Waterfall — Highest Risk Policy",
    subtitle = sprintf("Predicted PP: €%.0f/year (attritional)", shap_preds[idx_high])
  ) +
  plot_theme()

p_waterfall_low <- sv_waterfall(shap_obj, row_id = idx_low) +
  labs(
    title    = "SHAP Waterfall — Lowest Risk Policy",
    subtitle = sprintf("Predicted PP: €%.0f/year", shap_preds[idx_low])
  ) +
  plot_theme()

ggsave("outputs/figures/xgb_waterfall_high.png", p_waterfall_high,
       width = 8, height = 6, dpi = 150)
ggsave("outputs/figures/xgb_waterfall_low.png", p_waterfall_low,
       width = 8, height = 6, dpi = 150)

cat("\n--- High risk policy profile ---\n")
print(dat[shap_idx[idx_high],
          c("DrivAge", "BonusMalus", "VehPower", "VehAge",
            "Area", "Density", "pp_xgb")])

cat("\n--- Low risk policy profile ---\n")
print(dat[shap_idx[idx_low],
          c("DrivAge", "BonusMalus", "VehPower", "VehAge",
            "Area", "Density", "pp_xgb")])

# --- 7d. Dependence plots ---
p_dep_bm <- sv_dependence(shap_obj, v = "log_BonusMalus",
                          color_var = "DrivAge") +
  labs(title    = "SHAP Dependence — log(Bonus-Malus)",
       subtitle = "Colour = driver age",
       x = "log(Bonus-Malus)", y = "SHAP value") +
  plot_theme()

p_dep_age <- sv_dependence(shap_obj, v = "DrivAge",
                           color_var = "log_BonusMalus") +
  labs(title    = "SHAP Dependence — Driver Age",
       subtitle = "Colour = log(Bonus-Malus)",
       x = "Driver age", y = "SHAP value") +
  plot_theme()

ggsave("outputs/figures/xgb_shap_dependence.png",
       p_dep_bm + p_dep_age,
       width = 12, height = 5, dpi = 150)


# =============================================================================
# 8. ONE-WAY LIFT — XGBoost vs GLM
# =============================================================================
# Both models compared on the attritional scale:
#   - XGBoost: direct capped predictions
#   - Freq-Sev GLM: unloaded by dividing by LARGE_LOSS_LOADING (1.4245)
#   - Observed: capped claim amount / exposure

LARGE_LOSS_LOADING <- 1.4245

xgb_lift <- function(data, var, var_label, nbins = 10) {
  data |>
    mutate(bin = ntile(.data[[var]], nbins)) |>
    group_by(bin) |>
    summarise(
      midpoint  = mean(.data[[var]]),
      obs_pp    = sum(TotalClaim_capped) / sum(Exposure),
      pred_glm  = weighted.mean(pp_freq_sev / LARGE_LOSS_LOADING, Exposure),
      pred_xgb  = weighted.mean(pp_xgb, Exposure),
      .groups   = "drop"
    ) |>
    pivot_longer(c(obs_pp, pred_glm, pred_xgb),
                 names_to = "type", values_to = "pp") |>
    mutate(type = recode(type,
                         "obs_pp"   = "Observed (capped)",
                         "pred_glm" = "Freq-Sev GLM (attritional)",
                         "pred_xgb" = "XGBoost (attritional)"
    )) |>
    ggplot(aes(x = midpoint, y = pp, colour = type, linetype = type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.8) +
    scale_y_continuous(labels = comma) +
    scale_colour_manual(values = c(
      "Observed (capped)"          = "grey40",
      "Freq-Sev GLM (attritional)" = "#2166ac",
      "XGBoost (attritional)"      = "#d6604d"
    )) +
    scale_linetype_manual(values = c(
      "Observed (capped)"          = "dotted",
      "Freq-Sev GLM (attritional)" = "solid",
      "XGBoost (attritional)"      = "dashed"
    )) +
    labs(x = var_label, y = "Pure premium — attritional (€/year)",
         colour = NULL, linetype = NULL) +
    plot_theme() +
    theme(legend.position = "bottom")
}

p_xgb_drivage <- xgb_lift(dat, "DrivAge",    "Driver age") +
  labs(title = "Pure Premium by Driver Age — GLM vs XGBoost")
p_xgb_bm      <- xgb_lift(dat, "BonusMalus", "Bonus-Malus score") +
  labs(title = "Pure Premium by Bonus-Malus — GLM vs XGBoost")
p_xgb_vehage  <- xgb_lift(dat, "VehAge",     "Vehicle age") +
  labs(title = "Pure Premium by Vehicle Age — GLM vs XGBoost")
p_xgb_density <- xgb_lift(dat, "Density",    "Population density") +
  labs(title = "Pure Premium by Density — GLM vs XGBoost")

p_xgb_lift <- (p_xgb_drivage + p_xgb_bm) / (p_xgb_vehage + p_xgb_density)

ggsave("outputs/figures/xgb_lift.png", p_xgb_lift,
       width = 12, height = 10, dpi = 150)


# =============================================================================
# 9. SAVE PREDICTIONS
# =============================================================================

xgb_preds <- dat |>
  select(IDpol, Exposure, ClaimNb, TotalClaimAmount,
         TotalClaim_capped, HasClaim, pp_xgb)

saveRDS(xgb_preds, "outputs/data/xgb_predictions.rds")
cat("\nXGBoost predictions saved to outputs/data/xgb_predictions.rds\n")
cat("Rows:", nrow(xgb_preds), "\n")
cat("\nEnsemble modelling complete.\n")