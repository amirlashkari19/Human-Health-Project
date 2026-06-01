# P3 — Rc / WMARM-like (City-Level Adaptation) Report Section

**Project:** Milan Air Pollution & Respiratory Morbidity (HHLab)
**Analyst role:** Person 3 — Rc / WMARM-like Adaptation Analyst
**Tasks covered:** P3 Trello cards Day 2 → Day 11–12 (full P3 stack)
**Date:** 15 May 2026

---

## 1. Why APHREH-ADSMap is relevant — and why this is a *city-level adaptation*

The APHREH-ADSMap framework is a vulnerability/relevance toolkit built around two concepts:

- **Rc (city-level relevance)** — a per-cell summary of how strongly an exposure perturbs an outcome relative to a non-exposure baseline window, weighted by exposure intensity and statistical confidence.
- **WMARM (Weighted Multi-Area Relevance Map)** — a population-weighted spatial aggregation of Rc across multiple Basic Spatial Areas (BSAs).

The Milan dataset contains a **single city-level BSA** (the Comune di Milano, population 1,407,044), so the spatial aggregation step of WMARM is mathematically trivial. We therefore implement:

- **Rc** in full per `(pollutant, outcome, year, lag)` cell, and
- a **WMARM-like score** that summarises the per-cell standardised effect across years using inverse-CI-width weights.

Throughout the section the original APHREH-ADSMap terminology is kept where the underlying computation is intact, and the explicit qualifier "WMARM-like" is used wherever the spatial aggregation step is replaced by a temporal/cross-year aggregation.

## 2. Inputs and parameters

| Parameter | Value | Source |
|---|---|---|
| Sample | n = 154 weeks (W18/2023 → W16/2026, W8/2025 outlier excluded) | P1 `analysis_ready.csv` |
| Population denominator | 1,407,044 (Comune di Milano) | analyst input |
| Pollutants | NO₂, PM₂.₅, PM₁₀ (weekly means, µg/m³) | P1 |
| Outcomes | respiratory disease (primary), ILI, pneumonia | P1 |
| Default exposure threshold (`pc`) | 0.75 (yearly per-pollutant percentile) | P3 spec |
| Lags evaluated | 0, 1, 2 weeks (lagged outcomes) | P3 spec |
| Baseline window | ±3 weeks, expand to ±4, then ±5 if fewer than 2 non-exposed neighbours | P3 spec |
| Bootstrap iterations | 1,000 (300 in sensitivity grid for runtime) | P3 spec |
| Multiplicative noise (`nz`) | 0.05 | P3 spec |
| RNG seed | 42 | reproducibility |

## 3. Methods

### 3.1 Exposure classification (Day 2-3)

For each pollutant and each calendar year, we computed the within-year 75th percentile of the weekly mean and labelled each week as **exposed** (≥ threshold) or **non-exposed** (< threshold). Yearly thresholds were used so that exposure is identified relative to that year's local distribution. Counts: 35 exposed / 119 non-exposed weeks per pollutant (24 / 76% — close to the nominal 25 / 75% split, with small departures due to ties at the threshold and the truncated 2023 (W18+) and 2026 (≤W16) calendars).

### 3.2 Weekly incidence and lagging (Day 3-4)

Weekly incidence was computed as `outcome_count / 1,407,044`. Lagged outcomes (lag 0, 1, 2) were built by shifting the incidence series forward by *L* weeks; the last *L* weeks of the series were dropped from the lag-*L* analysis because the post-exposure outcome is unobserved.

### 3.3 Baseline incidence and differential incidence (Day 4-5)

For each target week *t*, pollutant *p* and lag *L*, the baseline window comprises the non-exposed weeks within ±3 weeks of *t* (excluding *t* itself). The window was expanded to ±4, then ±5 weeks if fewer than two non-exposed neighbours were available. Baseline incidence was the **median** of lag-*L* incidence over those neighbours, and **Δinc** was defined as

> Δinc(*t*) = lagged-incidence(*t*) − median-baseline.

Across the 4,059 cells produced, **3,753 (92.5%) used the default ±3-week window**; 216 (5.3%) expanded to ±4; 90 (2.2%) expanded to ±5; none required exclusion. The expansion log is preserved at `tables/baseline_window_expansions.csv`.

### 3.4 Exposure weights (Day 5-6)

For each week and pollutant, the distance from that year's threshold was log1p-transformed and normalised to [0, 1] across that pollutant's exposed weeks (highest exposure → 1, mild exposure → small positive). Non-exposed weeks (and any with missing pollutant) received a low default weight of 0.01 so that they retain non-zero contribution to the weighted statistics. Weights are saved in `tables/exposure_weights.csv` and visualised at `figures/exposure_weights_timeline.png` — the timeline shows the expected winter pollution peaks dominating across all three pollutants.

### 3.5 Bootstrap weighted Mann-Whitney (Day 6-7)

For each `(pollutant, outcome, year, lag)` cell we drew 1,000 bootstrap replicates. Within each replicate:

1. Sample **with replacement**, equal *n* (*n* = min of exposed and non-exposed in the cell) from the exposed and non-exposed Δinc distributions.
2. Apply 5% multiplicative Gaussian noise: `Δinc_noisy = Δinc × (1 + N(0, 0.05))`.
3. Apply the per-week exposure weights computed in §3.4.
4. Compute a **weighted Mann-Whitney U statistic**:

> *U_w* = Σ_i Σ_j *w_i* · *w_j* · I[ x_i > y_j ] (+ 0.5 for ties)

with asymptotic normal-approximation *p*-value and rank-biserial effect size

> *r* = *U_w* / (Σ*w_i* · Σ*w_j*) ∈ [0, 1], 0.5 = null.

We retain *U*, *p*-value, *r*, and the direction (sign of *r* − 0.5) for every iteration. Output: `tables/bootstrap_results.csv` (108 cells × 1,000 = 108,000 rows).

### 3.6 Rc computation (Day 7-8) — handoff table for P4

For each cell:

> Vul = weighted-mean( *r*<sub>boot</sub> , weights = (1 − *p*<sub>boot</sub>)<sub>+</sub> )

with **Rc ≡ Vul**, and 95% confidence bounds taken as the inverse-(1 − *p*)-weighted 2.5th and 97.5th percentiles of the bootstrap *r* distribution. Each cell receives a 9-class APHREH-ADSMap-style label based on the deviation |Vul − 0.5|:

> Negligible / Very low / Low / Moderate-low / Moderate / Moderate-high / High / Very high / Critical, evenly partitioned over [0, 0.5].

This table is the **critical hand-off to P4**: `tables/Rc_weighted_CI_table.csv` (108 rows). The full version with bootstrap diagnostics is `tables/weekly_citylevel_Rc_by_pollutant_outcome_year.csv`.

### 3.7 WMARM-like score (Day 8-9)

For each `(pollutant, outcome, lag)` triple we summarise across the four calendar years:

- `sd_r` = standard deviation of bootstrap *r* (per cell)
- signed standardised effect: *se* = (Vul − 0.5) / sd_r (per cell)
- absolute standardised effect: |*se*|
- inverse-CI weight: 1 / (CI_high − CI_low)
- **city-level WMARM-like se** = inverse-CI-weighted mean of signed *se* across years.

Output: `tables/citylevel_WMARM_like_summary.csv` and an explicit caveat at `tables/WMARM_like_caveat.txt`.

### 3.8 Sensitivity surfaces (Day 9-10)

For each pollutant against the **primary outcome** (respiratory disease) we re-ran the full pipeline at every cell of the `pc` × `lag` grid (`pc` ∈ {0.50, 0.60, 0.70, 0.75, 0.80, 0.90} × `lag` ∈ {0, 1, 2}), producing a city-level WMARM-like se per cell (300 bootstrap iterations per cell for runtime). For each pollutant we render a 3D surface and a 2D heatmap; the city-level se is floored on `sd_r ≥ 10⁻³` and clipped to ±10 to suppress numerical instabilities at low-count edge cells. Outputs: `sensitivity/sensitivity_grid.csv` and `figures/WMARM_like_surface_{NO2|PM25|PM10}.png`.

## 4. Results

### 4.1 Cell-level Rc highlights

Mean Vul averaged across years and lags is consistently > 0.5 (i.e. **exposed weeks tend to have higher Δinc than non-exposed neighbours** — the harmful direction):

| Pollutant | Outcome | mean Vul | sd | Notable cells |
|---|---|---:|---:|---|
| NO₂ | respiratory_disease | 0.579 | 0.346 | 2023 lag-1: Vul = 0.97, **Critical**; 2026 lag-2: Vul = 0.03 (small-n) |
| PM₂.₅ | respiratory_disease | 0.530 | 0.215 | 2023 lag-1 high; 2026 cells unstable |
| PM₁₀ | respiratory_disease | 0.518 | 0.206 | weakest year-averaged Vul of the three |
| NO₂ | pneumonia | 0.600 | 0.299 | 2023 lag-1: Vul = 0.98, **Critical** |
| (all pollutants × all outcomes) | | 0.51 – 0.63 | | |

The most robust **Critical** classifications are concentrated in 2023 (which begins at W18 and is therefore largely a winter-pollution sample) — this is consistent with a real signal but the inflated cell-level Vul also reflects fewer non-exposed neighbours within ±3-week windows in the truncated year.

### 4.2 City-level WMARM-like summary (primary outcome only)

For respiratory disease, summarised over 2023 – 2026 with inverse-CI weighting:

| Pollutant | Lag | mean Vul | citylevel se (weighted) |
|---|---:|---:|---:|
| **NO₂** | **1** | **0.663** | **4.48** |
| NO₂ | 2 | 0.580 | 1.50 |
| NO₂ | 0 | 0.496 | 0.28 |
| PM₂.₅ | 0 | 0.608 | 1.82 |
| PM₂.₅ | 1 | 0.548 | 0.48 |
| PM₂.₅ | 2 | 0.435 | -0.61 |
| PM₁₀ | 0 | 0.585 | 1.80 |
| PM₁₀ | 1 | 0.545 | 0.44 |
| PM₁₀ | 2 | 0.424 | -0.80 |

**The strongest city-level signal is NO₂ → respiratory at lag 1 (se = 4.48).** This is concordant with the P2 DLNM headline finding that NO₂ → respiratory disease is the only pollutant–outcome combination whose cumulative RR (1.78, 95% CI 1.05–3.01) excludes 1, with the strongest contribution at short lags (1–2 weeks).

### 4.3 Sensitivity surface

Across the `pc` × `lag` sensitivity grid for respiratory disease:

| Pollutant | Best (pc, lag) | Peak se | Mean across grid |
|---|---|---:|---:|
| **NO₂** | **(0.75, 1)** | **2.77** | 0.86 |
| PM₂.₅ | (0.90, 0) | 3.56* | 0.59 |
| PM₁₀ | (0.90, 0) | 1.73 | 0.30 |

\* PM₂.₅ peak at `pc = 0.90` should be interpreted with caution — only ~10% of weeks within each year clear that threshold, so the corresponding bootstrap is built on a very small effective sample; the next-largest PM₂.₅ cells are around 1.4 and the surface is generally flat.

NO₂'s sensitivity surface peaks inside the default specification region (`pc = 0.75`, lag 1) and is the only surface whose maximum is *not* on a corner of the grid — that is the desirable property: the headline configuration is also the locally optimal one.

## 5. Cross-method comparison with P2 DLNM

| Question | P2 DLNM | P3 Rc / WMARM-like |
|---|---|---|
| Which pollutant has the strongest signal on respiratory disease? | NO₂ (cum RR 1.78, CI 1.05–3.01) | **NO₂** (city-level se 4.48 at lag 1) |
| At which lag(s) is the effect strongest? | Lags 0–2 (surface peak at high NO₂, short lags) | **Lag 1** (peak); lag 2 secondary |
| Is the result robust to specification? | 82/144 robust, 62/144 partially robust; Fourier seasonality flips sign | Sensitivity peak inside the default cell; concordant ranking across the grid |
| What is the magnitude expressed in case terms? | AF ≈ 35%, ~139,000 attributable cases (CI 1.5%–51%) | (Rc is a vulnerability metric, not a counterfactual — directly compared on ranking, not magnitude) |

The two methods come from very different statistical traditions (parametric DLNM regression vs. non-parametric weighted bootstrap) and yet **agree on the same headline**: NO₂ is the pollutant with material relevance for respiratory morbidity, the strongest signal sits at short lags, and PM₁₀ / PM₂.₅ effects on the same outcome are weaker.

## 6. Limitations

- **Single BSA.** Spatial WMARM is degenerate with one city; the WMARM-like score is an aggregation across years rather than across spatial units. This is documented in `tables/WMARM_like_caveat.txt`.
- **Truncated 2023 / 2026.** 2023 starts at W18 and 2026 ends at W16; both years are dominated by partial seasonal cycles, which inflates within-year variance and explains the more extreme cell-level Vul values in those years.
- **Asymptotic-normal Mann-Whitney p-value.** With small *n_each* per bootstrap (especially at extreme `pc`), the asymptotic p-value can be optimistic; CIs are accordingly wide for those cells.
- **Yearly thresholds.** The default `pc = 0.75` is *within-year*, so thresholds drift with the underlying pollution distribution. Sensitivity over `pc` quantifies the impact.
- **Population denominator.** The 1,407,044 Milan municipality figure assumes the ED-visit catchment matches the city. If visits include the metro catchment (~3.25M), incidences scale uniformly and cancel out of *r*, so Rc/WMARM-like results are denominator-invariant; only the absolute incidence reported in `lagged_incidence.csv` would change.
- **Causality.** Rc is associational. The 9-class label communicates *relevance*, not preventable fraction.

## 7. Deliverables index

```
p3_deliverables/
├── scripts/
│   └── 01_run_p3_pipeline.py               # full pipeline
├── data/processed/
│   └── p3_parameters.json                  # pinned parameters
├── rc_wmarm/
│   ├── tables/
│   │   ├── exposure_classification.csv
│   │   ├── lagged_incidence.csv
│   │   ├── differential_incidence.csv
│   │   ├── baseline_window_expansions.csv
│   │   ├── exposure_weights.csv
│   │   ├── bootstrap_results.csv           # 108,000 rows
│   │   ├── weekly_citylevel_Rc_by_pollutant_outcome_year.csv
│   │   ├── Rc_weighted_CI_table.csv        # >>> handoff to P4
│   │   ├── citylevel_WMARM_like_summary.csv
│   │   └── WMARM_like_caveat.txt
│   ├── sensitivity/
│   │   ├── sensitivity_grid.csv            # pc x lag x pollutant grid
│   │   └── sensitivity_summary.txt
│   └── figures/
│       ├── exposure_weights_timeline.png
│       └── WMARM_like_surface_{NO2,PM25,PM10}.png
└── report/draft/
    └── rc_wmarm_section_P3.md              # this document
```

## 8. Reproducibility

```bash
cd p3_deliverables/scripts
python3 01_run_p3_pipeline.py
```

Requires Python 3.10 with `numpy`, `pandas`, `matplotlib`. No external data: input is `../analysis_ready.csv` (frozen P1 output, used by every fit). RNG seed 42 fixes the bootstrap. Total runtime ≈ 10–20 s on a 2024-class laptop.

---

*Contact: Amir Lashkari — amir.lashkari1996@gmail.com — PoliMi*
