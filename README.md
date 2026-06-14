# Air Pollution and Respiratory Health in Milan (2023-2026)

A weekly time-series study of how ambient air pollution relates to respiratory
emergency department (ED) visits in Milan. The project is organised into four
phases:

- **Phase 1 - Data preparation and exploratory analysis:** build one clean
  weekly dataset and the exploratory charts.
- **Phase 2 - Distributed Lag Non-Linear Models (DLNM):** the main analysis of
  how weekly pollutant exposure relates to respiratory ED visits.
- **Phase 3 - Rc / WMARM-like analysis:** check the same question with a second,
  non-parametric method that shares no assumptions with the DLNM.
- **Phase 4 - Variable screening (3S-GeoXAI Stage I) and spatial context:**
  confirm the predictor choices, and place the results on a map.

Each phase reads from a single shared dataset, so the cleaning and the main
choices are made once and reused, which keeps the whole workflow easy to follow
and to reproduce on another machine.

## Repository structure

```
Phase 1 - Preprocessing/     data preparation and EDA (Python)
Phase 2 - DLNM/              DLNM models, sensitivity, attributable risk (R)
Phase 3 - Rc-WMARM/          Rc / WMARM-like relevance analysis (Python)
Phase 4 - 3S-GeoXAI/         variable screening (Python) + QGIS spatial maps
analysis_ready.csv          the clean dataset from Phase 1, used by Phases 2, 3 and 4
requirements.txt            Python dependencies for Phases 1, 3 and 4
```

## Setup

Phases 1, 3 and 4 use Python; Phase 2 uses R, so you need both.

Python (Phases 1, 3 and 4):

```
pip install -r requirements.txt
```

R (Phase 2) - install the packages once from an R console:

```
install.packages(c("dlnm", "MASS", "dplyr", "ggplot2", "readr", "broom"))
```

(`splines` ships with base R.)

---

# Phase 1 - Data Preparation and Exploratory Analysis

This phase takes the raw data we collected, cleans it, checks it, and produces
one tidy dataset that the later phases use directly. It also draws the
exploratory charts we use to understand the data before any modelling.
Everything runs from a single command.

## What this phase produces

The main output is `analysis_ready.csv`. Every later script starts from it, so
the cleaning rules and the outlier decision are made once, in one place, and are
easy to check. The script also draws nine figures (the ones used in the report)
in the `p1_figures` folder.

## Files

| File | What it is |
|------|------------|
| `p1_preprocessing.py` | The main script. Builds `analysis_ready.csv` and the charts. |
| `dataset_settimanale_finale 2.csv` | The raw weekly input (see "Where the data comes from"). |
| `analysis_ready.csv` | The clean output dataset used by the next phases. |
| `p1_figures/` | The nine exploratory figures created by the script. |

## How to run it

You need Python 3.10 or newer and the packages in `requirements.txt`. Then:

```
python p1_preprocessing.py
```

That one command reads `dataset_settimanale_finale 2.csv`, writes
`analysis_ready.csv`, and saves the nine figures in `p1_figures`. The file names
are set at the bottom of the script if you ever need to change them.

## Where the data comes from

The weekly input file was built from three sources for the City of Milan and the
study period (week 18 of 2023 to week 16 of 2026):

- **Air quality:** daily NO2, PM2.5 and PM10 values from the ARPA Lombardia
  open-data portal. We used the five monitoring stations inside the Milan
  municipal boundary (one central station measuring NO2 and PM10, three full
  stations measuring NO2, PM2.5 and PM10, and one peripheral station measuring
  NO2 only). The daily values were averaged across the stations and then grouped
  into ISO weeks.
- **Health data:** weekly ED visit counts for respiratory disease,
  influenza-like illness (ILI) and pneumonia, provided at city level.
- **Weather:** weekly mean temperature and relative humidity from the same ARPA
  network.

These sources were combined into the weekly file
(`dataset_settimanale_finale 2.csv`) before the script runs. That merging was
done once by hand, so it is described here rather than scripted. The repository
includes this weekly file as the starting point; the large daily raw exports are
kept locally and are not part of the repository.

We start the study at week 18 of 2023 on purpose. The COVID-19 period strongly
changed both pollution levels and hospital use until 2022, so beginning after it
gives a more normal picture of the exposure and outcome relationship.

## What the script does, step by step

1. Reads the raw file and detects whether it uses commas or semicolons.
2. Turns the seven daily values of each pollutant into weekly summaries (mean,
   maximum and 75th percentile). We keep all three because the later methods use
   different measures: the DLNM uses the weekly mean, and the Rc analysis uses
   the 75th percentile to mark high-exposure weeks.
3. Sorts the weeks and adds a week counter and a readable week label.
4. Adds season labels (winter, spring, summer) and Fourier terms used later to
   control for seasonality.
5. Adds quality flags: how many daily values are missing in each week, and a
   flag for week 52 of 2025, which only had five of seven days because a station
   was under maintenance. This week is kept but flagged.
6. Flags the week 8 of 2025 outlier (see below) instead of deleting it.
7. Checks for missing values and prints a short summary.
8. Saves the selected columns to `analysis_ready.csv`.

## The outlier (week 8 of 2025)

In late February 2025 the respiratory ED count is about 64% below the winter
average, which is far too low to be a real change in demand during a cold,
high-pollution week. We treat it as a short reporting problem in the health
data. The script does not delete it; it adds a column called `outlier_flag` (set
to 1 for this week). The later phases simply keep the rows where
`outlier_flag == 0`. This keeps the decision visible and easy to reverse.

## A note on the number of weeks

The dataset has **155** weekly records in total. After removing the week 8 of
2025 outlier, **154** weeks remain for the main analysis. Both numbers are
correct: 155 is the full series, and 154 is the analytic sample.

## The output dataset (`analysis_ready.csv`)

One row per week, 42 columns. The main groups are:

- **Identifiers:** `year`, `week`, `week_index`, `year_week`, `week_start`.
- **Exposure:** weekly `mean`, `max` and `p75` for NO2, PM2.5 and PM10, plus the
  number of valid days per pollutant.
- **Weather:** `temp_mean`, `humidity_mean`.
- **Outcomes:** `respiratory_disease`, `ILI`, `pneumonia` (ED visits per week).
- **Seasonality:** season label, season flags, and the Fourier terms
  (`sin52`/`cos52`, `sin26`/`cos26`).
- **Quality flags:** `partial_pollution_week`, `complete_pollution_week`, the
  per-pollutant missing-day counts, `w52_2025_flag` and `outlier_flag`.

## What the exploratory charts show

The nine figures summarise the patterns we found before modelling:

- All pollutants and all outcomes are much higher in winter.
- PM2.5 and PM10 move almost identically (Spearman correlation about 0.96), so we
  cannot put both in the same model and instead fit one pollutant at a time.
- Temperature is strongly and negatively linked to respiratory visits
  (correlation about -0.83), so season and temperature must be controlled for
  before reading any pollution effect.

These findings are the reason the later phases use single-pollutant models with
careful seasonal and temperature adjustment.

---

# Phase 2 - Distributed Lag Non-Linear Models (DLNM)

This phase fits the models that link weekly pollutant exposure to weekly
respiratory ED visits. It starts from `analysis_ready.csv` and produces the
fitted models, the relative-risk (RR) figures, the summary tables, a sensitivity
analysis, an attributable-risk calculation, and extended diagnostics.

## Scripts

| File | What it does |
|------|--------------|
| `01_fit_dlnm_models.R` | Fits the primary model and the 9 single-pollutant models; saves the model objects, RR figures, the model-reduction tables and the basic diagnostics. Run this first. |
| `02_sensitivity_analysis.R` | Re-runs the primary model across 144 specifications (lag, spline degrees of freedom, seasonality, partial week) and checks how stable the result is. |
| `03_attributable_risk.R` | Computes the attributable fraction and number for each model against the WHO reference exposures, with bootstrap confidence intervals. |
| `04_diagnostics_extended.R` | Cook's distance, DFBETA influence, and an include-vs-exclude check for the partial week W52/2025. |

## How to run

With R (4.x) and the packages from the Setup section:

```
Rscript 01_fit_dlnm_models.R
Rscript 02_sensitivity_analysis.R
Rscript 03_attributable_risk.R
Rscript 04_diagnostics_extended.R
```

Run `01` first, because it creates the model objects and tables the report
relies on. Each script reads `analysis_ready.csv` from the same folder and writes
its results into the subfolders below, which are created automatically.

## Outputs

| Folder | Contents |
|--------|----------|
| `model_objects/` | The fitted models (`.rds`), one text summary per model, `model_metadata.csv` (convergence and over-dispersion for all 9 models), `warnings_log.txt`, and `sessionInfo.txt`. |
| `rr_surfaces/` | Six figures per model (3D RR surface, contour, cumulative RR, lag-response at the 75th percentile, and exposure-response at lag 0 and lag 2). |
| `model_reduction/` | `cumulative_RR_table.csv`, `lag_specific_RR_table.csv`, `predictor_specific_RR_table.csv`. |
| `sensitivity/` | The 144-specification table, the Fourier-vs-spline comparison, the winter-interaction check, and a summary figure. |
| `attributable_risk/` | `attributable_risk_table.csv` and a caveat note. |
| `diagnostics/` | Observed-vs-fitted, residuals over time, residual histogram and ACF, Cook's distance, DFBETA, and the W52/2025 in-vs-out comparison. |

## Main modelling choices

- Quasi-Poisson regression, because the weekly counts are over-dispersed.
- A cross-basis with natural cubic splines in both the exposure and the lag
  dimension, with a maximum lag of 4 weeks.
- Seasonality is controlled with a natural spline on the week index, and
  temperature and humidity are added as within-season confounders.
- One pollutant per model, because the three pollutants are strongly correlated.
- The W8/2025 outlier is excluded; the partial week W52/2025 is kept in the main
  analysis and tested separately in the diagnostics.
- Reference (counterfactual) exposures follow the WHO 2021 guidelines:
  NO2 = 10, PM2.5 = 5, PM10 = 15 ug/m3.

## Notes and limitations

- Only the primary association (NO2 to respiratory disease) reaches statistical
  significance; the other models are reported for completeness.
- The sensitivity analysis is included on purpose to show how stable the main
  result is. It holds across most specifications but weakens when seasonality is
  modelled with Fourier terms instead of a spline. We keep this visible rather
  than hide it, as an honest assessment of the method's limits.
- The attributable-risk numbers are computed under a stated counterfactual and
  are associational, not proof of cause and effect (see `AR_caveats.txt`).

---

# Phase 3 - Rc / WMARM-like Analysis

This phase looks at the same question as Phase 2 but with a different,
non-parametric method: a weekly, city-level adaptation of the APHREH-ADSMap
relevance approach. Using a second method that shares no modelling assumptions
with the DLNM is a deliberate check: if both methods point to the same pollutant,
the finding is more trustworthy. It starts from `analysis_ready.csv` and runs the
whole pipeline with one command.

## Files

| File | What it is |
|------|------------|
| `rc_wmarm_analysis.py` | The single script that runs the whole analysis. |
| `analysis_ready.csv` | The Phase 1 output, used as input. |

## How to run

With Python 3.10 or newer and the packages in `requirements.txt`:

```
python rc_wmarm_analysis.py
```

The script reads `analysis_ready.csv` from the same folder and writes all results
into subfolders that it creates automatically.

## What the method does, in plain terms

1. **Exposure classification.** For each pollutant and each year, a week is
   labelled "exposed" if its value is above that year's 75th percentile. Using a
   yearly threshold keeps the comparison fair when overall pollution drifts
   between years.
2. **Incidence and lagging.** Weekly ED visits are turned into an incidence per
   resident, and shifted by 0, 1 and 2 weeks to test for delayed effects.
3. **Baseline and differential incidence.** Each week is compared with the median
   of its non-exposed neighbours within plus or minus three weeks (widened to
   four or five if too few neighbours are available). This local comparison
   removes the seasonal pattern without a regression.
4. **Exposure weights.** Weeks that are far above the threshold get more weight,
   so the strongest pollution weeks count for more.
5. **Bootstrap weighted Mann-Whitney test.** A resampling test (1,000 repeats)
   asks whether exposed weeks have a higher differential incidence than
   non-exposed weeks, giving a relevance score (Vul) between 0 and 1.
6. **Rc and the 9-class label.** The score is summarised with a confidence
   interval and placed on a nine-level scale from Negligible to Critical.
7. **WMARM-like score.** The yearly scores are combined into one city-level
   number per pollutant, outcome and lag.
8. **Sensitivity grid.** The whole thing is repeated over a grid of thresholds
   and lags, to check that the result does not depend on one chosen threshold.

## Outputs

| Folder | Contents |
|--------|----------|
| `rc_wmarm/tables/` | Exposure classification, lagged and differential incidence, exposure weights, the bootstrap results, the Rc tables, and the city-level WMARM-like summary. |
| `rc_wmarm/figures/` | The exposure-weight timeline and the threshold-by-lag surfaces for each pollutant. |
| `rc_wmarm/sensitivity/` | The sensitivity grid and a short summary. |
| `data/processed/p3_parameters.json` | A record of every parameter used, for reproducibility. |

## Main choices

- We use a single, city-level area, because the health data exist only at city
  level. The original spatial step of the method is therefore replaced by a
  summary across years; this is stated openly and the score is called
  "WMARM-like" rather than a full spatial WMARM (see `WMARM_like_caveat.txt`).
- The baseline window is local in time (plus or minus three weeks), which plays
  the same role as the seasonal adjustment in the DLNM but without a model.
- All random steps use a fixed seed, so the results are the same on every run.

## Notes and limitations

- The score is a measure of relevance and association, not proof of cause.
- The 2023 and 2026 results are less reliable because those years are only partly
  covered (the data start in week 18 of 2023 and end in week 16 of 2026), which
  leaves fewer weeks to compare. This is visible in the tables and is taken into
  account when reading the results.

---

# Phase 4 - Variable Screening (3S-GeoXAI Stage I) and Spatial Context

This phase has two parts. The first is a variable-screening script that confirms,
with a simple and transparent method, the predictor choices used in Phase 2. The
second is a set of QGIS maps that place the study and its results in their
geographic frame.

## Part A - Variable screening (`3S-GeoXAI/`)

This is Stage I of the 3S-GeoXAI framework: a univariate Spearman screening of
the candidate predictors against each outcome, followed by collinearity removal.
Stages II (MGWR) and III (Random Forest + SHAP) are not run, by design: the
health data exist only at city level (no spatial grid for MGWR) and 154 weekly
observations are too few for a reliable Random Forest with SHAP. Stage I
therefore acts as a check that supports the DLNM design, not as a separate
discovery method.

| File | What it is |
|------|------------|
| `stage1_screening.py` | The screening script. Reads `analysis_ready.csv` from the same folder. |
| `analysis_ready.csv` | The Phase 1 output, used as input. |

Run it with Python and the packages in `requirements.txt`:

```
python stage1_screening.py
```

It writes its results to `stage1_output/`: ranked-predictor tables and figures
per outcome, a combined three-panel ranking figure, the list of collinear pairs
and the keep/drop decisions, and a lagged-correlation heatmap (lags 0-3) that
supports the choice of lag window in the DLNM.

As expected on a seasonal time series, the screening flags the pollutants as
correlated with the seasonal driver (temperature and the Fourier terms). This is
the expected result and is exactly why a DLNM is used in Phase 2; it is not
evidence that the pollutants are irrelevant.

## Part B - QGIS spatial context (`QGIS/`)

These maps were produced in QGIS to give the study a geographic frame. They are
contextual only: the health outcomes are city-level, so no district-level
health-risk claim is made.

| File | What it is |
|------|------------|
| `milan_stations_clean.csv` | The ARPA monitoring stations in and around Milan (coordinates, station type, and which pollutants each one measures). |
| `map3_Rc_respiratory_full.csv` | The city-level Rc results from Phase 3 (pollutant, year, lag, Rc, confidence interval, class, direction), formatted for the map inset. |
| `map3_Rc_respiratory_compact.csv` | A shorter version of the same table. |
| `maps/map1_monitoring_stations_pollutants.png` | The ARPA stations inside Milan and the pollutants they record. |
| `maps/Map2_NIL_Respiratory_2024.png` | Respiratory ED visits for 2024 shown over Milan's NIL districts, for spatial context only (the values are a population-based allocation, not measured per district). |
| `maps/map3_citylevel_Rc_inset.png` | The Phase 3 Rc results placed as an inset over the city. |
| `maps/map4_yearly_pollution_panels.png` | Year-by-year panels pairing pollution levels with respiratory demand. |

The maps reuse the Phase 3 Rc output, so they stay consistent with the rest of
the analysis. They were prepared interactively in QGIS rather than by a script,
so the CSVs above are the inputs and the PNGs are the exported results.
