# Insurance Pure Premium Modelling
### Frequency-Severity GLM Pipeline with Ensemble Comparison

![R](https://img.shields.io/badge/R-4.x-276DC3?logo=r&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

A reproducible R pipeline for motor insurance **pure premium modelling**, built on the French Motor Third-Party Liability dataset (`freMTPL2`). The project implements the industry-standard **frequency-severity separation** approach using Poisson and Gamma GLMs, compares performance against an XGBoost Tweedie ensemble, and evaluates models using actuarially appropriate metrics including the normalised Gini coefficient and double lift chart.

---

## Business Context

In non-life (property & casualty) insurance, pricing is the process of estimating the **expected cost of a risk** before agreeing to insure it. Underpricing a risk destroys margin; overpricing it loses business to competitors.

The core quantity of interest is the **pure premium** — the expected claims cost per unit of exposure (typically per policy-year):

$$\text{Pure Premium} = \frac{\text{Expected Claims Cost}}{\text{Exposure}}$$

A well-calibrated pure premium model allows an insurer to:
- Segment risk more accurately than competitors
- Price complex or unusual risks with less reliance on heuristics
- Identify loss-making segments within an existing book

This project models pure premium for a French motor liability portfolio of ~678,000 policies, where the modelling task is to estimate expected annual claims cost per policy given policyholder and vehicle characteristics.

---

## Dataset

**Source:** `CASdatasets` R package — `freMTPL2freq` (claim frequency) and `freMTPL2sev` (claim severity).

This is a real-world dataset widely used in actuarial science education and research. It is not synthetic.

| Table | Rows | Key fields |
|---|---|---|
| `freMTPL2freq` | 678,013 | `PolicyID`, `Exposure`, `ClaimNb`, driver/vehicle covariates |
| `freMTPL2sev` | 26,639 | `PolicyID`, `ClaimAmount` (one row per claim) |

**Key variables:**

- `Exposure` — fraction of year the policy was active (0–1). **Critical for modelling.**
- `ClaimNb` — number of claims in the exposure period
- `ClaimAmount` — individual claim payment
- `BonusMalus` — experience-rated score; a proxy for unobserved driver risk
- `VehPower`, `VehAge`, `VehBrand`, `VehGas` — vehicle characteristics
- `DrivAge` — policyholder age
- `Region`, `Area` — spatial risk factors

**To obtain the data:**
```r
install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/")
library(CASdatasets)
data(freMTPL2freq)
data(freMTPL2sev)
```

---

## Modelling Approach

### Why separate frequency and severity?

A natural instinct is to model pure premium directly as a single response variable. Two alternatives exist:

1. **Tweedie GLM / XGBoost Tweedie**: model pure premium directly using a compound Poisson-Gamma distribution
2. **Frequency-severity separation**: model claim frequency (Poisson) and claim severity (Gamma) independently, then multiply predictions

The frequency-severity approach is the **industry standard** in P&C pricing for good reasons:

- Frequency and severity are driven by **different risk factors**. A young driver in a city has high frequency risk (more incidents); a high-powered vehicle has high severity risk (more expensive repairs). Separating them allows each sub-model to learn the right signal.
- Severity is modelled only on **policies with at least one claim**, avoiding the zero-inflation problem that distorts a single-model approach.
- The two-model structure gives underwriters and pricing actuaries **interpretable, decomposable outputs** — they can understand whether a rate change is driven by frequency or severity.
- It aligns with regulatory and reserving workflows, where frequency and severity are tracked separately.

The Tweedie single-model approach is included as a **benchmark**, but the frequency-severity pipeline is the primary output.

---

## Pipeline

```
freMTPL2freq + freMTPL2sev
        │
        ▼
   01_eda.R              Exploratory analysis, exposure distribution,
                         zero-inflation, loss ratio by covariate
        │
        ▼
   02_frequency_model.R  Poisson GLM with log(Exposure) offset
                         Overdispersion check → quasi-Poisson if needed
        │
        ▼
   03_severity_model.R   Gamma GLM on claims-only subset
                         Claim count as weights
        │
        ▼
   04_pure_premium.R     Multiply frequency × severity predictions
                         Compare against Tweedie GLM baseline
        │
        ▼
   05_ensemble.R         XGBoost (reg:tweedie objective)
                         Tuned via cross-validation
                         SHAP values via {shapviz}
        │
        ▼
   06_evaluation.R       Normalised Gini coefficient
                         Double lift chart
                         Calibration by decile
                         GLM vs XGBoost comparison
```

---

## Key Results

Pure premium figures are in €/policy-year. The freq-sev GLM and Tweedie GLM incorporate a large loss loading of ×1.4245 for claims above the 99th percentile severity threshold (€16,327). Evaluation bases differ by model — see notes below.

| Metric | Freq-Sev GLM | XGBoost | Tweedie GLM |
|---|---|---|---|
| Evaluation set | Full dataset (in-sample) | 20% holdout (out-of-sample) | Full dataset (in-sample) |
| Observed loss basis | Uncapped | Capped (attritional layer) | Uncapped |
| Normalised Gini | **0.370** | 0.350 | 0.355 |
| Top decile ratio | 6.19 | 8.97 | 8.24 |
| Mean predicted PP | €166.36 | €113.35 | €210.76 |
| Mean observed PP | €167.12 | €118.07 | €167.12 |
| Portfolio ratio (pred/obs) | **0.995** | 0.960 | 1.263 |
| Exposure-weighted MAE | €313 | €215 | €353 |

**Notes on comparability:** Gini figures are not directly comparable across models. XGBoost is evaluated out-of-sample against capped (attritional) losses — a harder and fairer test. The GLMs are evaluated in-sample against uncapped losses including the large loss loading. The XGBoost Gini of 0.350 on a held-out test set is broadly equivalent to the GLMs' in-sample Gini of ~0.37, suggesting comparable discrimination once evaluation basis is aligned.

**Key findings:**

- **The freq-sev GLM is well-calibrated** at portfolio level (ratio 0.995) and the most interpretable model. Its transparency and decomposability — separate frequency and severity outputs — make it the preferred model in a regulatory or actuarial review context.
- **XGBoost has superior segment-level calibration.** Its double lift chart is flat across all deciles (ratios 0.89–1.20 with no systematic pattern). The freq-sev GLM shows a ratio of 1.73 in decile 9 — a 73% underprice in a specific mid-high risk segment concentrated in urban Area B policies. In a live book, this segment would be loss-making and would drive adverse selection.
- **The Tweedie GLM overpredicts** by 26.3% (ratio 1.263) and is outperformed on all metrics. The variance power parameter was fixed at *p* = 1.5 rather than estimated via profile likelihood; this is the primary driver of miscalibration and is a clear next step.
- **In production**, the freq-sev GLM would be the filed rating model. XGBoost would serve as a challenge model to identify segments where the GLM misprices — as demonstrated by the decile 9 finding.

**Concentration curves and double lift chart:**

`outputs/figures/eval_lorenz.png` — Lorenz curves for all three models  
`outputs/figures/eval_double_lift.png` — actual/predicted ratio by decile

**SHAP summary (XGBoost):**

`outputs/figures/xgb_shap_beeswarm.png` — global feature importance  
`outputs/figures/xgb_waterfall_high.png` — highest risk policy (age 18, BonusMalus 100, predicted €1,444/year attritional)  
`outputs/figures/xgb_waterfall_low.png` — lowest risk policy (age 31, BonusMalus 54, rural, predicted €8/year)

---

## Evaluation Metrics

Standard classification metrics (AUC, accuracy) are not appropriate for insurance pricing models, where the goal is **risk rank-ordering**, not binary prediction.

### Normalised Gini Coefficient
Measures how well the model discriminates between high and low risk policies. Derived from the Lorenz curve of predicted vs actual loss. A value of 0 indicates no discrimination; 1 indicates perfect rank-ordering. In practice, well-performing motor pricing models achieve normalised Gini of 0.25–0.45.

### Double Lift Chart
Policies are sorted by predicted pure premium and binned into deciles. For each decile, the ratio of actual to expected loss is plotted. A well-calibrated model produces a flat line at 1.0 across deciles — systematic departures indicate model bias in specific risk segments.

### Calibration by Decile
Compares the mean predicted pure premium against the mean actual pure premium within each predicted decile. Assesses absolute accuracy (not just rank-ordering).

---

## Modelling Decisions

A log of non-trivial modelling choices and their justifications:

| Decision | Rationale |
|---|---|
| `log(Exposure)` as offset, not predictor | Exposure is not a risk factor; it scales the observation window. Treating it as an offset constrains its coefficient to 1 on the log scale, which is actuarially correct. |
| Gamma for severity, not lognormal | Gamma GLM keeps predictions on the original scale and handles heteroscedasticity naturally. Lognormal requires retransformation with a smearing correction, introducing bias. |
| Severity cap on claim cost scale, not pure premium scale | Dividing total claim amount by small exposure values produces very large per-policy-year figures that distort the 99th percentile threshold. Capping at the per-claim severity level (€16,327) is consistent with the severity model and avoids exposure-driven artefacts. |
| Tweedie XGBoost for ensemble | Allows direct pure premium prediction without requiring separate freq/sev pipelines in the ML model. The `p` (power) parameter is tuned alongside other hyperparameters. |
| XGBoost CV model selection by early stopping rounds | Tweedie log-likelihood CV scores were numerically unstable (all Inf) due to 96% zero-inflation in the pure premium response. Early stopping convergence is used as a proxy for model quality; the combination with the most rounds before early stopping is preferred. |
| SHAP over variable importance | Permutation importance and gain-based importance are unstable with correlated features. SHAP values provide consistent, additive attribution. |
| Normalised Gini over AUC | AUC treats the problem as binary. Normalised Gini respects the continuous, skewed nature of insurance loss distributions. |

---

## Limitations & Next Steps

- **Evaluation methodology**: The GLMs are evaluated in-sample (fitted and assessed on the full dataset). A three-way train/validation/test split would give unbiased out-of-sample GLM metrics. XGBoost uses a 20% holdout but early stopping was performed on the same test set, introducing a minor optimism bias in rounds selection; a dedicated validation set would eliminate this.
- **Spatial structure**: Region is treated as a fixed effect. A spatial random effect (e.g. via `mgcv::gam` with a Markov random field smoother) would better capture geographic risk variation.
- **Unobserved heterogeneity**: BonusMalus is a useful proxy but a mixed-effects model (e.g. random intercept by Region) would more formally account for clustering.
- **Temporal effects**: The dataset has no policy year field, so trend modelling is not possible here. In production, loss development and trend adjustments are standard.
- **Telematics / third-party data**: Real pricing models increasingly incorporate telematics (speed, braking) and external data (weather, crime indices). These are absent here.
- **Catastrophe / large loss handling**: Claims above the 99th percentile severity threshold (€16,327) are excluded from the attritional models and loaded back via a fixed factor (1.4245). In production, a dedicated large loss model (e.g. a Pareto tail fit) would replace the flat loading.

---

## How to Run

This project uses [`renv`](https://rstudio.github.io/renv/) for reproducibility.

```r
# 1. Clone the repo
# 2. Open the .Rproj file in RStudio (sets working directory to project root)

# 3. Install CASdatasets from its own repository BEFORE restoring the renv environment
#    (renv cannot resolve this package automatically as it is not on CRAN)
install.packages("CASdatasets", repos = "http://cas.uqam.ca/pub/R/")

# 4. Restore the package environment
renv::restore()

# 5. Run scripts in order
source("R/01_eda.R")
source("R/02_frequency_model.R")
source("R/03_severity_model.R")
source("R/04_pure_premium.R")
source("R/05_ensemble.R")   # saves outputs/data/test_idx.rds — required by script 06
source("R/06_evaluation.R")
```

Figures are written to `outputs/figures/`. Model objects are saved to `outputs/models/` (gitignored due to size). `outputs/data/test_idx.rds` is saved by script 05 and must exist before running script 06.

**R version:** 4.5.1  
**Key packages:** `tidyverse`, `CASdatasets`, `xgboost`, `shapviz`, `broom`, `patchwork`, `scales`

---

## Repository Structure

```
insurance-pricing-model/
├── README.md
├── insurance-pricing-model.Rproj
├── renv.lock
├── .gitignore
├── data/                        # not tracked
├── R/
│   ├── 01_eda.R
│   ├── 02_frequency_model.R
│   ├── 03_severity_model.R
│   ├── 04_pure_premium.R
│   ├── 05_ensemble.R
│   └── 06_evaluation.R
├── outputs/
│   ├── figures/
│   ├── tables/
│   ├── data/
│   │   └── test_idx.rds         # saved by script 05, required by script 06
│   └── models/                  # not tracked
└── report/
    └── summary.md
```

---
*Data source: Charpentier, A. (ed.) Computational Actuarial Science with R. CRC Press, 2014. Dataset available via the `CASdatasets` package.*
