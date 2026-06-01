"""
HHLab — Preprocessing & Data Cleaning  (Operator 1)
Ecological weekly time-series study — Milan, 2023-2026
n = 155 weeks (W18/2023 to W16/2026)

This script:
  1. Loads the raw dataset (dataset_settimanale_finale 2.csv)
  2. Computes weekly means from 7 daily pollutant values
  3. Assigns identifiers, season labels, and Fourier harmonics
  4. Generates quality flags and marks the W8/2025 outlier
  5. Exports analysis_ready.csv  <- INPUT for P2 (DLNM) and P3 (Rc/WMARM)

RULES FOR P2 AND P3:
  - Always use analysis_ready.csv (never the raw dataset)
  - Always use single-pollutant models: never mix NO2, PM2.5 and PM10
    (PM2.5-PM10 collinearity: rho = 0.956)
  - Always exclude the W8/2025 outlier (outlier_flag == 1)
  - W52/2025 has only 5/7 measurement days -> use only in sensitivity analysis

Input:  dataset_settimanale_finale 2.csv
Output: analysis_ready.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0. FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════
INPUT_FILE  = 'dataset_settimanale_finale 2.csv'
OUTPUT_FILE = 'analysis_ready.csv'

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATASET
# ══════════════════════════════════════════════════════════════════════════════
# Auto-detect separator (comma or semicolon — Italian Excel CSVs use semicolons)
with open(INPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
    first_line = f.readline()
sep = ';' if first_line.count(';') > first_line.count(',') else ','

raw = pd.read_csv(INPUT_FILE, sep=sep)
df  = raw.copy()

print(f"Separator detected: '{sep}'")
print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# Rename columns if needed (adjust to match actual column names in your file)
rename_map = {
    'resp':      'respiratory_disease',
    'ili':       'ILI',
    'polm':      'pneumonia',
    'anno':      'year',
    'settimana': 'week',
}
df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

# ══════════════════════════════════════════════════════════════════════════════
# 2. WEEKLY POLLUTANT MEANS  (from 7 daily values)
# ══════════════════════════════════════════════════════════════════════════════
pollutants = ['NO2', 'PM25', 'PM10']

for p in pollutants:
    day_cols = [f'{p}_day{i}' for i in range(1, 8)]
    existing = [c for c in day_cols if c in df.columns]

    if existing:
        # Force numeric conversion (handles comma decimals and non-numeric values)
        num = df[existing].apply(
            lambda col: pd.to_numeric(
                col.astype(str).str.replace(',', '.'), errors='coerce'
            )
        )
        df[f'{p}_mean']   = num.mean(axis=1, skipna=True)
        df[f'{p}_max']    = num.max(axis=1,  skipna=True)
        df[f'{p}_p75']    = num.quantile(0.75, axis=1)
        df[f'{p}_n_days'] = num.notna().sum(axis=1)
    else:
        # Daily columns not present -> weekly mean assumed already computed
        df[f'{p}_n_days'] = 7

print("Weekly means computed: NO2_mean, PM25_mean, PM10_mean")

# Also force numeric conversion on outcome and meteo columns
for col in ['respiratory_disease', 'ILI', 'pneumonia', 'temp_mean',
            'humidity_mean', 'precip_total_mm', 'wind_speed_mean', 'pressure_mean']:
    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(',', '.'), errors='coerce'
        )

# ══════════════════════════════════════════════════════════════════════════════
# 3. SORT AND IDENTIFIERS
# ══════════════════════════════════════════════════════════════════════════════
df['year'] = pd.to_numeric(df['year'], errors='coerce')
df['week'] = pd.to_numeric(df['week'], errors='coerce')
df = df.sort_values(['year', 'week']).reset_index(drop=True)

# Sequential index 1-155 (used for trend spline and Fourier harmonics)
df['week_index'] = np.arange(1, len(df) + 1)

# Composite label e.g. "2023-W18"
df['year_week'] = (df['year'].astype(int).astype(str) + '-W' +
                   df['week'].astype(int).astype(str).str.zfill(2))

# Parse week_start date
if 'week_start' in df.columns:
    df['week_start'] = pd.to_datetime(df['week_start'], format='mixed', dayfirst=True)
else:
    df['week_start'] = pd.to_datetime(
        df['year'].astype(int).astype(str) +
        df['week'].astype(int).astype(str).str.zfill(2) + '1',
        format='%G%V%u'
    )

print(f"Period: {df['year_week'].iloc[0]} to {df['year_week'].iloc[-1]}")
print(f"Observations: {len(df)} weeks")

# ══════════════════════════════════════════════════════════════════════════════
# 4. SEASON
# ══════════════════════════════════════════════════════════════════════════════
# ISO week-based meteorological seasons:
#   Winter:  weeks 1-13 and 40-53  (October-March)
#   Spring:  weeks 14-26           (April-June)
#   Summer:  weeks 27-39           (July-September)

def assign_season(w):
    if w <= 13 or w >= 40: return 'winter'
    if w <= 26:            return 'spring'
    return 'summer'

df['season']      = df['week'].apply(assign_season)
df['winter_flag'] = (df['season'] == 'winter').astype(int)
df['spring_flag'] = (df['season'] == 'spring').astype(int)
df['summer_flag'] = (df['season'] == 'summer').astype(int)

print(f"Seasons: winter n={df['winter_flag'].sum()}, "
      f"summer n={df['summer_flag'].sum()}, "
      f"spring n={df['spring_flag'].sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 5. FOURIER HARMONICS  (seasonality control for P2's DLNM)
# ══════════════════════════════════════════════════════════════════════════════
# Capture annual and semi-annual periodicity as continuous variables.
# Used by P2 as the S1.3B component of the DLNM model.
#
# sin52/cos52 (= sin1/cos1): annual cycle   (period = 52 weeks)
# sin26/cos26 (= sin2/cos2): semi-annual    (period = 26 weeks)

t = df['week_index']

df['sin52'] = np.sin(2 * np.pi * t / 52)
df['cos52'] = np.cos(2 * np.pi * t / 52)
df['sin26'] = np.sin(4 * np.pi * t / 52)
df['cos26'] = np.cos(4 * np.pi * t / 52)

# Aliases used directly in p2_dlnm_handoff.py
df['sin1'] = df['sin52']
df['cos1'] = df['cos52']
df['sin2'] = df['sin26']
df['cos2'] = df['cos26']

# Numeric time index for the trend spline ns(t, df=3) in P2
df['time_numeric'] = df['week_index']

# ══════════════════════════════════════════════════════════════════════════════
# 6. QUALITY FLAGS
# ══════════════════════════════════════════════════════════════════════════════
for p in pollutants:
    df[f'missing_{p}_days'] = 7 - df[f'{p}_n_days'].fillna(0)

# Weeks with at least one missing measurement day
df['partial_pollution_week'] = (
    df[[f'missing_{p}_days' for p in pollutants]].max(axis=1) > 0
).astype(int)

# Weeks with all 7 days available
df['complete_pollution_week'] = (
    df[[f'missing_{p}_days' for p in pollutants]].max(axis=1) == 0
).astype(int)

# W52/2025: only 5/7 measurement days (partial year-end week)
# -> included in main analysis, excluded in sensitivity analysis
df['w52_2025_flag'] = ((df['year'] == 2025) & (df['week'] == 52)).astype(int)

print(f"Weeks with partial data: {df['partial_pollution_week'].sum()} "
      f"(of which W52/2025: {df['w52_2025_flag'].sum()})")

# ══════════════════════════════════════════════════════════════════════════════
# 7. OUTLIER FLAG — W8/2025
# ══════════════════════════════════════════════════════════════════════════════
# Week 8/2025 (February 2025): anomalously low counts (-64% vs winter mean).
# Likely cause: temporary under-reporting at the ED.
# -> Exclude from all main analyses using: df[df['outlier_flag'] == 0]

df['outlier_flag'] = ((df['year'] == 2025) & (df['week'] == 8)).astype(int)

outlier_row = df[df['outlier_flag'] == 1]
if len(outlier_row) > 0:
    row = outlier_row.iloc[0]
    print(f"Outlier W8/2025: respiratory={row['respiratory_disease']}, "
          f"ILI={row['ILI']}, pneumonia={row['pneumonia']}")

# ══════════════════════════════════════════════════════════════════════════════
# 8. MISSING DATA CHECK
# ══════════════════════════════════════════════════════════════════════════════
outcome_cols  = ['respiratory_disease', 'ILI', 'pneumonia']
exposure_cols = ['NO2_mean', 'PM25_mean', 'PM10_mean']
meteo_cols    = ['temp_mean', 'humidity_mean']

print("\nMissing data check:")
for col in outcome_cols + exposure_cols + meteo_cols:
    if col in df.columns:
        n_miss = df[col].isna().sum()
        print(f"  {col}: {n_miss} missing ({100*n_miss/len(df):.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 9. SELECT FINAL COLUMNS AND EXPORT
# ══════════════════════════════════════════════════════════════════════════════
final_cols = [
    # Identifiers
    'year', 'week', 'week_index', 'year_week', 'week_start',
    # Season
    'season', 'winter_flag', 'spring_flag', 'summer_flag',
    # Exposure — weekly summaries (main variables for P2 and P3)
    'NO2_mean',  'NO2_max',  'NO2_p75',  'NO2_n_days',
    'PM25_mean', 'PM25_max', 'PM25_p75', 'PM25_n_days',
    'PM10_mean', 'PM10_max', 'PM10_p75', 'PM10_n_days',
    # Meteorological confounders
    'temp_mean', 'humidity_mean',
    # Outcomes (weekly ED visits)
    'respiratory_disease', 'ILI', 'pneumonia',
    # Fourier harmonics for seasonality (P2: DLNM)
    'sin52', 'cos52', 'sin26', 'cos26',
    'sin1',  'cos1',  'sin2',  'cos2',
    # Temporal index for trend spline (P2)
    'time_numeric',
    # Quality flags
    'partial_pollution_week', 'complete_pollution_week',
    'missing_NO2_days', 'missing_PM25_days', 'missing_PM10_days',
    'w52_2025_flag',
    'outlier_flag',
]

final_cols = [c for c in final_cols if c in df.columns]
analysis_ready = df[final_cols].copy()
analysis_ready.to_csv(OUTPUT_FILE, index=False)

print(f"\n✅ {OUTPUT_FILE} saved")
print(f"   Rows:    {len(analysis_ready)} weeks")
print(f"   Columns: {len(analysis_ready.columns)}")
print(f"   Outlier weeks (outlier_flag=1): {analysis_ready['outlier_flag'].sum()}")
print(f"   Complete weeks (all 7 days):    {analysis_ready['complete_pollution_week'].sum()}")

# ══════════════════════════════════════════════════════════════════════════════
# 10. VARIABLE DICTIONARY
# ══════════════════════════════════════════════════════════════════════════════
print("""
VARIABLE DICTIONARY — analysis_ready.csv
─────────────────────────────────────────────────────────────────────────────
 Variable                   Unit            Description
─────────────────────────────────────────────────────────────────────────────
 year / week                ISO             Year and ISO week number
 week_index                 1-155           Sequential week index
 year_week                  e.g. 2023-W18   Composite label
 week_start                 date            Monday of the ISO week
─────────────────────────────────────────────────────────────────────────────
 season                     categorical     winter / spring / summer
 winter_flag / spring_flag  0/1             Binary season flags
   / summer_flag
─────────────────────────────────────────────────────────────────────────────
 NO2_mean / PM25_mean       ug/m3           Weekly mean (from 7 daily values)
   / PM10_mean
 NO2_max  / PM25_max        ug/m3           Weekly maximum
   / PM10_max
 NO2_p75  / PM25_p75        ug/m3           Weekly 75th percentile
   / PM10_p75
 NO2_n_days / ...           days (1-7)      Valid measurement days
─────────────────────────────────────────────────────────────────────────────
 temp_mean                  C               Weekly mean temperature
 humidity_mean              %               Weekly mean relative humidity
─────────────────────────────────────────────────────────────────────────────
 respiratory_disease        visits/week     Main outcome
 ILI                        visits/week     Influenza-like illness
 pneumonia                  visits/week     Pneumonia
─────────────────────────────────────────────────────────────────────────────
 sin52 / cos52 (= sin1/cos1)  —            Fourier: annual cycle (52 weeks)
 sin26 / cos26 (= sin2/cos2)  —            Fourier: semi-annual (26 weeks)
 time_numeric               1-155           Continuous index for trend spline
─────────────────────────────────────────────────────────────────────────────
 outlier_flag               0/1             W8/2025 -> EXCLUDE from analyses
 w52_2025_flag              0/1             W52/2025 -> sensitivity only
 partial_pollution_week     0/1             >= 1 missing measurement day
─────────────────────────────────────────────────────────────────────────────
""")

# ══════════════════════════════════════════════════════════════════════════════
# 11. LOADING SNIPPET FOR P2 AND P3
# ══════════════════════════════════════════════════════════════════════════════
print("""
# -- Standard loading block (copy to the top of the P2 / P3 scripts) ----------

import pandas as pd
import numpy as np

df = pd.read_csv('analysis_ready.csv')
df['week_start'] = pd.to_datetime(df['week_start'])
df = df.sort_values('week_index').reset_index(drop=True)

# Exclude W8/2025 outlier for all main analyses
df_clean = df[df['outlier_flag'] == 0].copy()

# Winter subset (for sensitivity analysis)
df_winter = df_clean[df_clean['winter_flag'] == 1].copy()

print(f"n total (outlier excluded): {len(df_clean)}")
print(f"n winter: {len(df_winter)}")

# NOTE: always single-pollutant models
# NO2_mean  -> one model
# PM25_mean -> one model
# PM10_mean -> one model
# Never combine them in the same regression.
""")
OUTPUT_FILE = '/Users/gaiabarberis/Desktop/Project-human-health/analysis_ready.csv'