# Air Pollution and Respiratory Health in Milan (2023-2026)

A weekly time-series study of how ambient air pollution relates to respiratory
emergency department (ED) visits in Milan. The project is organised into phases.
This document covers the first two:

- **Phase 1 - Data preparation and exploratory analysis:** build one clean
  weekly dataset and the exploratory charts.
- **Phase 2 - Distributed Lag Non-Linear Models (DLNM):** model how weekly
  pollutant exposure relates to respiratory ED visits.

Each phase reads from a single shared dataset, so the cleaning and the main
choices are made once and reused, which keeps the whole workflow easy to follow
and to reproduce on another machine.

## Repository structure

```
Phase 1 - Preprocessing/     data preparation and EDA (Python)
Phase 2 - DLNM/              DLNM models, sensitivity, attributable risk (R)
analysis_ready.csv          the clean dataset produced by Phase 1 and used by Phase 2
requirements.txt            dependencies for both phases
```

## Setup

Phase 1 uses Python and Phase 2 uses R, so you need both.

Python (Phase 1):

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
