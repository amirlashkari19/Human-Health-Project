# P2 — DLNM Analyst Report Section

**Project:** Milan Air Pollution & Respiratory Morbidity (HHLab)
**Analyst role:** Person 2 — DLNM Analyst
**Coverage:** Trello Day 2 → Day 11–12 (setup, primary + 9 screening DLNMs, RR surfaces, model reduction, attributable risk, diagnostics, sensitivity, writing)
**Date:** 15 May 2026

---

## 1. Data and analytic sample

We used the analysis-ready weekly time series produced by P1: 155 ISO-weeks covering W18/2023 through W16/2026. Per the P1 protocol, the single labelled outlier week (W8/2025, `outlier_flag == 1`) was removed before model fitting, leaving **n = 154** complete-case observations. W52/2025 is flagged as a partial week (`w52_2025_flag == 1`); its impact is tested in sensitivity (Section 6.A and 7).

All three pollutants — NO₂, PM₂.₅, PM₁₀ — and the meteorological covariates (temperature, relative humidity) are weekly means in their native units (µg/m³, °C, %). The three respiratory outcomes are weekly aggregated case counts: respiratory disease (primary outcome), influenza-like illness (ILI), and pneumonia.

Per P1 guidance, **single-pollutant models only** are fit because PM₂.₅ and PM₁₀ are collinear (ρ = 0.956).

## 2. Modelling framework

### 2.1 Cross-basis specification (S1.1A/B)

For each pollutant–outcome pair we fit a quasi-Poisson generalised linear model with a cross-basis on the pollutant:

\[
\log E(Y_t) = \alpha + s(x_t, \ell; \boldsymbol{\eta}) + ns(\text{week\_index}_t, df=8) + ns(\text{temp}_t, df=3) + ns(\text{hum}_t, df=3)
\]

The cross-basis \(s(x_t,\ell;\boldsymbol{\eta})\) (built with `dlnm::crossbasis`) uses:

- **Maximum lag:** 4 weeks (weekly cadence; covers acute respiratory response windows for ambient pollution).
- **Variable (exposure) dimension:** natural cubic spline, `df = 3`.
- **Lag dimension:** natural cubic spline, `df = 3`.

Quasi-Poisson was chosen a priori because weekly counts typically show overdispersion; \(\hat\phi\) is reported per model and confirms this.

### 2.2 Confounder strategy (per P1 spec; S1.3B/C)

- **Seasonality / long-term trend:** `ns(week_index, df = 8)` — ~2.6 effective df/year over ~3 years.
- **Temperature:** `ns(temp_mean, df = 3)` — within-season confounder.
- **Humidity:** `ns(humidity_mean, df = 3)` — within-season confounder.

### 2.3 Reference values (WHO 2021)

| Pollutant | Reference (µg/m³) |
|-----------|------------------:|
| NO₂       | 10                |
| PM₂.₅     | 5                 |
| PM₁₀      | 15                |

Cumulative RR is reported at the 75th percentile of each pollutant's observed distribution versus reference.

## 3. Primary model — NO₂ → respiratory disease

- **Cumulative RR over lags 0–4, p75 (42.03 µg/m³) vs ref (10 µg/m³):** **RR = 1.778 (95% CI 1.052, 3.006)** ★
- Model converged with no aliasing; dispersion \(\hat\phi\) = 56.21 (quasi-Poisson family fully warranted).
- The 3-D RR surface (`dlnm/rr_surfaces/primary_NO2_respiratory_disease_surface.png`) and contour show the strongest excess risk at high NO₂ and short lags (0–2 weeks), consistent with an acute irritant mechanism.
- Diagnostics show no major residual structure after seasonal and meteorological adjustment.

## 4. Screening — all 9 single-pollutant models

Table 1. Cumulative RR over lags 0–4, p75 vs reference. ★ = 95% CI excludes 1.

| Model | Pollutant | Outcome     | Ref | p75   | Cum RR | 95% CI low | 95% CI high | Note |
|------:|:---------:|:------------|----:|------:|-------:|-----------:|------------:|:-----|
| M1    | NO₂       | Respiratory |  10 | 42.03 | **1.778** | 1.052 | 3.006 | ★ Primary |
| M2    | PM₂.₅     | Respiratory |   5 | 24.07 | 0.928 | 0.677 | 1.272 |   |
| M3    | PM₁₀      | Respiratory |  15 | 34.25 | 0.931 | 0.777 | 1.116 |   |
| M4    | NO₂       | ILI         |  10 | 42.03 | 0.963 | 0.427 | 2.170 |   |
| M5    | PM₂.₅     | ILI         |   5 | 24.07 | 0.944 | 0.580 | 1.536 |   |
| M6    | PM₁₀      | ILI         |  15 | 34.25 | 0.894 | 0.679 | 1.177 |   |
| M7    | NO₂       | Pneumonia   |  10 | 42.03 | 0.768 | 0.378 | 1.560 |   |
| M8    | PM₂.₅     | Pneumonia   |   5 | 24.07 | 0.848 | 0.559 | 1.286 |   |
| M9    | PM₁₀      | Pneumonia   |  15 | 34.25 | 0.927 | 0.729 | 1.179 |   |

Convergence: all 9 fits converged; no aliasing. Dispersion 12.25 ≤ \(\hat\phi\) ≤ 56.21 (warnings log empty).

Full tables: `dlnm/model_reduction/cumulative_RR_table.csv`, `lag_specific_RR_table.csv`, `predictor_specific_RR_table.csv`.

## 5. Attributable risk (S1.4)

Method: Gasparrini & Leone (2014) closed-form forward attributable fraction (AF) and attributable number (AN) using the fitted cross-basis, counterfactual reference exposure, and 2 000-iteration parametric bootstrap on the model coefficients for empirical 95% intervals.

Table 2. Attributable fraction and attributable number (lags 0–4 window).

| Model | Pollutant | Outcome     | Ref | AF % | AF % low | AF % high | AN     | Note |
|------:|:---------:|:------------|----:|-----:|---------:|----------:|-------:|:-----|
| M1    | NO₂       | Respiratory |  10 | **34.88** | 1.55 | 51.05 | 139 312 | ★ AF interval excludes 0 |
| M2    | PM₂.₅     | Respiratory |   5 | −5.79 | −34.66 | 16.36 | −23 132 | null |
| M3    | PM₁₀      | Respiratory |  15 | −3.69 | −25.61 | 23.34 | −14 744 | null |
| M4    | NO₂       | ILI         |  10 | −11.59 | −86.70 | 41.29 | −16 337 | null |
| M5    | PM₂.₅     | ILI         |   5 | −2.99 | −52.63 | 26.83 | −4 208 | null |
| M6    | PM₁₀      | ILI         |  15 | −4.51 | −28.36 | 36.58 | −6 363 | null |
| M7    | NO₂       | Pneumonia   |  10 | −38.94 | −112.84 | 23.60 | −22 056 | null |
| M8    | PM₂.₅     | Pneumonia   |   5 | −17.84 | −61.47 | 14.41 | −10 103 | null |
| M9    | PM₁₀      | Pneumonia   |  15 | −6.69 | −40.30 | 26.70 | −3 789 | null |

**Caveat (`dlnm/attributable_risk/AR_caveats.txt`):** these are *benchmark counterfactual* fractions — what share of weekly cases would not have occurred if ambient exposure had been at the reference level, assuming the fitted DLNM is the true causal model. They are not proof of causality; only M1 has a 95% interval excluding zero. Intervals reflect coefficient uncertainty only (not exposure misclassification or model-form uncertainty — see Section 6).

## 6. Sensitivity analysis (Day 9–10)

### 6.A Main grid

A 144-cell grid varies: max lag (2/4/6), exposure df (2/3), lag df (2/3), seasonality df (6/8/10), meteorology functional form (`ns(df=3)` vs linear), and partial-week handling (include / exclude W52/2025).

| Classification     | Count | Definition |
|--------------------|------:|:-----------|
| robust             | 82    | RR within ±25% of primary AND CI excludes 1 |
| partially_robust   | 62    | RR within ±50% of primary OR (CI doesn't exclude 1 but estimate close) |
| unstable           |  0    | RR > ±50% from primary or convergence failure |

All 144 specifications converged. The headline NO₂–respiratory association is **robust or partially robust under every grid cell** (`dlnm/sensitivity/dlnm_sensitivity_table.csv`, summary plot `dlnm_sensitivity_summary.png`).

### 6.B Fourier vs natural-spline seasonality

| Seasonality model         | Cum RR | 95% CI low | 95% CI high | AIC proxy |
|---------------------------|-------:|-----------:|------------:|----------:|
| ns(week_index, df = 6)    | 2.045  | 1.305 | 3.204 | 7176.3 |
| ns(week_index, df = 8)    | 1.778  | 1.052 | 3.006 | 6881.6 |
| ns(week_index, df = 10)   | 1.606  | 1.005 | 2.566 | 5599.5 |
| Fourier (sin52+cos52+sin26+cos26) | 0.756 | 0.492 | 1.160 | 6436.7 |

**Important model-form sensitivity to flag:** the natural-spline time control consistently produces RR > 1.6 with CI excluding 1, but a low-order Fourier representation pushes the estimate below 1. This indicates that the headline finding is partially dependent on letting seasonality flex more than a sin/cos pair allows. The natural-spline specification is preferred a priori (it captures inter-annual drift the Fourier terms cannot) and is the one used in the primary model. This sensitivity is reported transparently in the limitations.

### 6.C Winter interaction (secondary)

A secondary winter-stratified analysis (interaction with `winter_flag`) gives a non-winter cumulative RR of 2.024 (95% CI 0.375, 10.928) and a winter-only refit RR of 0.083 (0.027, 0.251). The winter-only sample is small (n = 73), the confidence intervals are extreme in both directions, and the result is **unstable**. Labelled as exploratory only (`dlnm/sensitivity/dlnm_sensitivity_winter.csv`); it should not be cited as a substantive finding.

## 7. Diagnostics (Day 7–8)

- **Observed vs fitted, residuals time series, residual histogram, residual ACF** for the primary model (`dlnm/diagnostics/primary_*.png`) — no obvious residual structure after seasonal/meteorological adjustment.
- **Cook's distance:** flagged weeks with `cook > 4/n` are W52/2024 (cook = 0.34), W04/2024 (0.19), W51/2023 (0.12), W22/2023 (0.12), W02/2024 (0.10), W49/2023 (0.07). All concentrated in mid-winter weeks of high cases — consistent with high-leverage flu peaks rather than data-quality artefacts.
- **W52/2025 in-vs-out comparison:** including the partial week gives RR = 1.778 (1.052, 3.006); excluding it gives RR = 1.761 (1.047, 2.964). The primary estimate is essentially invariant to partial-week handling.
- **Max |DFBETA| on cross-basis coefficients** plotted over time (`primary_max_dfbeta_cb.png`); occasional excursions above 2/√n in winter peaks but no week single-handedly drives the result.

## 8. Limitations (model-side)

- Lag horizon is short (max 4 weeks at weekly cadence); longer chronic effects cannot be detected with this design.
- PM signals are essentially null in this sample — likely a power / contrast issue at the weekly scale, not necessarily true absence of effect.
- ILI and pneumonia outcomes show no pollutant signal; ILI is heavily seasonal, pneumonia counts are small.
- The headline NO₂ result is **stable across natural-spline seasonality df = 6/8/10 but reverses under a low-order Fourier seasonality**; readers should be told.
- Attributable risk intervals are bootstrap on coefficients only — they understate true uncertainty.

## 9. What goes to P3/P4

- Spreadsheets in `dlnm/model_reduction/` and `dlnm/attributable_risk/` are ready to be appendix tables.
- The primary model RR surface, cumRR curve, and the sensitivity summary plot are the three figures that should appear in the main report body.
- The 8 non-primary screening models go into the appendix.

## 10. Deliverables index

```
p2_deliverables/
├── dlnm/
│   ├── scripts/
│   │   ├── 01_fit_dlnm_models.R          — primary + 9 screening DLNMs
│   │   ├── 02_sensitivity_analysis.R     — full grid + Fourier + winter interaction
│   │   ├── 03_attributable_risk.R        — AF/AN with bootstrap CIs
│   │   └── 04_diagnostics_extended.R     — Cook's distance, DFBETAs, W52 in/out
│   ├── model_objects/                    — 10 .rds + summaries + metadata + sessionInfo
│   ├── rr_surfaces/                      — 54 PNGs (6 views × 9 models)
│   ├── model_reduction/                  — cumulative / lag-specific / predictor-specific RR tables
│   ├── attributable_risk/                — attributable_risk_table.csv + AR_caveats.txt
│   ├── diagnostics/                      — 6 PNGs + Cook's top-10 CSV + W52 in/out CSV
│   └── sensitivity/                      — sensitivity_table.csv + fourier.csv + winter.csv + summary PNG
├── data/processed/analysis_ready.csv     — frozen P1 output
└── report/draft/dlnm_section_P2.md       — this document
```

*Reproducibility:* every artefact in this folder is regenerated by running the four scripts in order (`01` → `02` → `03` → `04`) against `data/processed/analysis_ready.csv` under R 4.5.0 with `dlnm`, `splines`, `MASS`, `dplyr`, `ggplot2`, `readr`, `broom`.
