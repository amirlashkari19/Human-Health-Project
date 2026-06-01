# Data

## Raw data

| File | Description | Size |
|---|---|---|
| `raw/dataset_settimanale_finale 2.csv` | Weekly Milan air pollution + respiratory morbidity dataset (2023–2026) | ~32 KB |

The raw CSV is committed directly — it is non-sensitive aggregate ecological data
(weekly city-level counts, no individual patient records).

## Processed data

| File | Description | Source |
|---|---|---|
| `processed/analysis_ready.csv` | Cleaned, feature-engineered dataset ready for modelling | Output of `p1_data_prep/P1_script.py` |

### To regenerate `analysis_ready.csv`

```bash
cd /path/to/repo
python p1_data_prep/P1_script.py \
  --input "data/raw/dataset_settimanale_finale 2.csv" \
  --output data/processed/analysis_ready.csv
```

## Key variables

| Variable | Type | Description |
|---|---|---|
| `year`, `week` | int | ISO year and week number |
| `NO2_mean`, `PM25_mean`, `PM10_mean` | float | Weekly mean pollutant concentrations (µg/m³) |
| `temp_mean`, `humidity_mean` | float | Weekly mean meteorological covariates |
| `respiratory_disease`, `ILI`, `pneumonia` | int | Weekly ED/outpatient visit counts |
| `outlier_flag` | 0/1 | W8/2025 outlier (−64% vs winter mean) — excluded from main analyses |
| `w52_2025_flag` | 0/1 | W52/2025 partial week (5/7 measurement days) — sensitivity only |
