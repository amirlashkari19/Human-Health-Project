# Phase 1 — Data Preparation and Exploratory Analysis

Milan air pollution and respiratory emergency department (ED) study, 2023–2026.

This is the first phase of the project. Its job is to take the raw data we
collected, clean it, check it, and produce one tidy dataset that the later
phases (the DLNM models and the Rc/WMARM-like analysis) can use directly. It
also draws the exploratory charts that we use to understand the data before
any modelling.

Everything here works from a single command, so the whole phase can be
reproduced on another computer without changing the code.

## What this phase produces

The main output is `analysis_ready.csv`. This is the file the next phases
read. Every later script starts from it, so the cleaning rules and the outlier
decision are made once, in one place, and are easy to check.

The script also draws nine figures.

## Files in this folder

| File | What it is |
|------|------------|
| `p1_preprocessing.py` | The main script. Builds `analysis_ready.csv` and the charts. |
| `dataset_settimanale_finale 2.csv` | The raw weekly input (see "Where the data comes from"). |
| `analysis_ready.csv` | The clean output dataset used by the next phases. |
| `p1_figures/` | The nine exploratory figures created by the script. |
| `requirements.txt` | The Python packages needed to run the script. |

## How to run it

You need Python 3.10 or newer. Install the packages first:

```
pip install -r requirements.txt
```

Then run the script:

```
python p1_preprocessing.py
```

That one command does everything: it reads `dataset_settimanale_finale 2.csv`,
writes `analysis_ready.csv`, and saves the nine figures in the `p1_figures`
folder. The file names are set at the bottom of the script if you ever need to
change them.

## Where the data comes from

The weekly input file was built from three public or institutional sources for
the City of Milan and the study period (week 18 of 2023 to week 16 of 2026):

- **Air quality:** daily NO2, PM2.5 and PM10 values from the ARPA Lombardia
  open-data portal. We used the five monitoring stations inside the Milan
  municipal boundary (one central station measuring NO2 and PM10, three full
  stations measuring NO2, PM2.5 and PM10, and one peripheral station measuring
  NO2 only). The daily values were averaged across the stations and then
  grouped into ISO weeks.
- **Health data:** weekly ED visit counts for respiratory disease,
  influenza-like illness (ILI) and pneumonia, provided at city level for the
  study period.
- **Weather:** weekly mean temperature and relative humidity from the same
  ARPA network.

These sources were combined into the weekly file
(`dataset_settimanale_finale 2.csv`) before this script runs. That merging was
done once by hand, so it is described here rather than scripted. The repository
includes this weekly file as the starting point; the large daily raw exports
are kept locally and are not part of the repository.

We start the study at week 18 of 2023 on purpose. The COVID-19 period strongly
changed both pollution levels and hospital use until 2022, so beginning after
it gives a more normal picture of the exposure and outcome relationship.

## What the script does, step by step

1. Reads the raw file and detects whether it uses commas or semicolons.
2. Turns the seven daily values of each pollutant into weekly summaries (mean,
   maximum and 75th percentile). We keep all three because the later methods
   use different measures: the DLNM uses the weekly mean, and the Rc analysis
   uses the 75th percentile to mark high-exposure weeks.
3. Sorts the weeks and adds a week counter and a readable week label.
4. Adds season labels (winter, spring, summer) and Fourier terms that the
   models use to control for seasonality.
5. Adds quality flags: how many daily values are missing in each week, and a
   flag for week 52 of 2025, which only had five of seven days because a
   station was under maintenance. This week is kept but flagged.
6. Flags the week 8 of 2025 outlier (see below) instead of deleting it.
7. Checks for missing values and prints a short summary.
8. Saves the selected columns to `analysis_ready.csv`.

## The outlier (week 8 of 2025)

In late February 2025 the respiratory ED count is about 64% below the winter
average, which is far too low to be a real change in demand during a cold,
high-pollution week. We treat it as a short reporting problem in the health
data. The script does not delete it; it adds a column called `outlier_flag`
(set to 1 for this week). The later phases simply keep the rows where
`outlier_flag == 0`. Doing it this way keeps the decision visible and easy to
reverse if needed.

## A note on the number of weeks

The dataset has **155** weekly records in total. After removing the week 8 of
2025 outlier, **154** weeks remain for the main analysis. Both numbers are
correct: 155 is the full series, and 154 is the analytic sample.

## The output dataset (`analysis_ready.csv`)

One row per week, 42 columns. The main groups are:

- **Identifiers:** `year`, `week`, `week_index`, `year_week`, `week_start`.
- **Exposure:** weekly `mean`, `max` and `p75` for NO2, PM2.5 and PM10, plus
  the number of valid days per pollutant.
- **Weather:** `temp_mean`, `humidity_mean`.
- **Outcomes:** `respiratory_disease`, `ILI`, `pneumonia` (ED visits per week).
- **Seasonality:** season label, season flags, and the Fourier terms
  (`sin52`/`cos52`, `sin26`/`cos26`).
- **Quality flags:** `partial_pollution_week`, `complete_pollution_week`,
  the per-pollutant missing-day counts, `w52_2025_flag` and `outlier_flag`.

## What the exploratory charts show

The nine figures summarise the patterns we found before modelling. The most
important ones are:

- All pollutants and all outcomes are much higher in winter.
- PM2.5 and PM10 move almost identically (Spearman correlation about 0.96), so
  we cannot put both in the same model and instead fit one pollutant at a time.
- Temperature is strongly and negatively linked to respiratory visits
  (correlation about -0.83), so season and temperature must be controlled for
  before reading any pollution effect.

These findings are the reason the later phases use single-pollutant models with
careful seasonal and temperature adjustment.

## How this prepares the next phase

`analysis_ready.csv` is the single starting point for Phase 2 (the DLNM models)
and Phase 3 (the Rc/WMARM-like analysis). Because the weekly summaries, the
season and Fourier terms, the quality flags and the outlier decision are all
fixed in this file, the later phases do not repeat any cleaning. They just load
the file, keep the rows where `outlier_flag == 0`, and run their models.
