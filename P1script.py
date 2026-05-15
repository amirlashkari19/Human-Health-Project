# Preprocessing and data cleaning — HHLab project, Milan 2023-2026
# Loads the raw weekly dataset, computes pollutant summaries, assigns
# season labels and Fourier harmonics, flags the W8/2025 outlier, and
# exports analysis_ready.csv for use by P2 and P3.
#
# Usage:  python p1_preprocessing_handoff.py [--input file.csv] [--output out.csv]
# Requirements: pandas, numpy

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def detect_separator(filepath: str) -> str:
    """Auto-detect CSV separator (comma or semicolon)."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert a series to numeric, replacing comma decimals with dots."""
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def assign_season(week: int) -> str:
    """
    ISO week-based meteorological seasons:
      Winter: weeks  1-13 and 40-53  (October-March)
      Spring: weeks 14-26            (April-June)
      Summer: weeks 27-39            (July-September)
    """
    if week <= 13 or week >= 40:
        return "winter"
    if week <= 26:
        return "spring"
    return "summer"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(input_file: str, output_file: str) -> None:

    # ── 1. LOAD RAW DATASET ────────────────────────────────────────────────────
    sep = detect_separator(input_file)
    raw = pd.read_csv(input_file, sep=sep)
    df = raw.copy()

    print(f"Separator detected: '{sep}'")
    print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    rename_map = {
        "resp":      "respiratory_disease",
        "ili":       "ILI",
        "polm":      "pneumonia",
        "anno":      "year",
        "settimana": "week",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # ── 2. WEEKLY POLLUTANT MEANS (from 7 daily values) ───────────────────────
    pollutants = ["NO2", "PM25", "PM10"]

    for p in pollutants:
        day_cols = [f"{p}_day{i}" for i in range(1, 8)]
        existing = [c for c in day_cols if c in df.columns]

        if existing:
            num = df[existing].apply(to_numeric_safe)
            df[f"{p}_mean"]   = num.mean(axis=1, skipna=True)
            df[f"{p}_max"]    = num.max(axis=1, skipna=True)
            df[f"{p}_p75"]    = num.quantile(0.75, axis=1)
            df[f"{p}_n_days"] = num.notna().sum(axis=1)
        else:
            df[f"{p}_n_days"] = 7

    print("Weekly means computed: NO2_mean, PM25_mean, PM10_mean")

    numeric_cols = [
        "respiratory_disease", "ILI", "pneumonia",
        "temp_mean", "humidity_mean", "precip_total_mm",
        "wind_speed_mean", "pressure_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # ── 3. SORT AND IDENTIFIERS ────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.sort_values(["year", "week"]).reset_index(drop=True)

    df["week_index"] = np.arange(1, len(df) + 1)

    df["year_week"] = (
        df["year"].astype(int).astype(str)
        + "-W"
        + df["week"].astype(int).astype(str).str.zfill(2)
    )

    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(
            df["week_start"], format="mixed", dayfirst=True
        )
    else:
        df["week_start"] = pd.to_datetime(
            df["year"].astype(int).astype(str)
            + df["week"].astype(int).astype(str).str.zfill(2)
            + "1",
            format="%G%V%u",
        )

    print(f"Period: {df['year_week'].iloc[0]} to {df['year_week'].iloc[-1]}")
    print(f"Observations: {len(df)} weeks")

    # ── 4. SEASON ──────────────────────────────────────────────────────────────
    df["season"]      = df["week"].apply(assign_season)
    df["winter_flag"] = (df["season"] == "winter").astype(int)
    df["spring_flag"] = (df["season"] == "spring").astype(int)
    df["summer_flag"] = (df["season"] == "summer").astype(int)

    print(
        f"Seasons: winter n={df['winter_flag'].sum()}, "
        f"summer n={df['summer_flag'].sum()}, "
        f"spring n={df['spring_flag'].sum()}"
    )

    # ── 5. FOURIER HARMONICS ───────────────────────────────────────────────────
    # sin52/cos52: annual cycle (period = 52 weeks)
    # sin26/cos26: semi-annual  (period = 26 weeks)
    t = df["week_index"]

    df["sin52"] = np.sin(2 * np.pi * t / 52)
    df["cos52"] = np.cos(2 * np.pi * t / 52)
    df["sin26"] = np.sin(4 * np.pi * t / 52)
    df["cos26"] = np.cos(4 * np.pi * t / 52)

    df["sin1"] = df["sin52"]
    df["cos1"] = df["cos52"]
    df["sin2"] = df["sin26"]
    df["cos2"] = df["cos26"]

    df["time_numeric"] = df["week_index"]

    # ── 6. QUALITY FLAGS ───────────────────────────────────────────────────────
    for p in pollutants:
        df[f"missing_{p}_days"] = 7 - df[f"{p}_n_days"].fillna(0)

    df["partial_pollution_week"] = (
        df[[f"missing_{p}_days" for p in pollutants]].max(axis=1) > 0
    ).astype(int)

    df["complete_pollution_week"] = (
        df[[f"missing_{p}_days" for p in pollutants]].max(axis=1) == 0
    ).astype(int)

    # W52/2025: only 5/7 measurement days -> sensitivity analysis only
    df["w52_2025_flag"] = ((df["year"] == 2025) & (df["week"] == 52)).astype(int)

    print(
        f"Weeks with partial data: {df['partial_pollution_week'].sum()} "
        f"(of which W52/2025: {df['w52_2025_flag'].sum()})"
    )

    # ── 7. OUTLIER FLAG — W8/2025 ─────────────────────────────────────────────
    # February 2025: -64% vs winter mean, likely ED under-reporting.
    # Exclude from all main analyses: df[df['outlier_flag'] == 0]
    df["outlier_flag"] = ((df["year"] == 2025) & (df["week"] == 8)).astype(int)

    outlier_row = df[df["outlier_flag"] == 1]
    if len(outlier_row) > 0:
        row = outlier_row.iloc[0]
        print(
            f"Outlier W8/2025: respiratory={row['respiratory_disease']}, "
            f"ILI={row['ILI']}, pneumonia={row['pneumonia']}"
        )

    # ── 8. MISSING DATA CHECK ─────────────────────────────────────────────────
    outcome_cols  = ["respiratory_disease", "ILI", "pneumonia"]
    exposure_cols = ["NO2_mean", "PM25_mean", "PM10_mean"]
    meteo_cols    = ["temp_mean", "humidity_mean"]

    print("\nMissing data check:")
    for col in outcome_cols + exposure_cols + meteo_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            print(f"  {col}: {n_miss} missing ({100 * n_miss / len(df):.1f}%)")

    # ── 9. SELECT FINAL COLUMNS AND EXPORT ────────────────────────────────────
    final_cols = [
        "year", "week", "week_index", "year_week", "week_start",
        "season", "winter_flag", "spring_flag", "summer_flag",
        "NO2_mean",  "NO2_max",  "NO2_p75",  "NO2_n_days",
        "PM25_mean", "PM25_max", "PM25_p75", "PM25_n_days",
        "PM10_mean", "PM10_max", "PM10_p75", "PM10_n_days",
        "temp_mean", "humidity_mean",
        "respiratory_disease", "ILI", "pneumonia",
        "sin52", "cos52", "sin26", "cos26",
        "sin1",  "cos1",  "sin2",  "cos2",
        "time_numeric",
        "partial_pollution_week", "complete_pollution_week",
        "missing_NO2_days", "missing_PM25_days", "missing_PM10_days",
        "w52_2025_flag",
        "outlier_flag",
    ]

    final_cols = [c for c in final_cols if c in df.columns]
    analysis_ready = df[final_cols].copy()
    analysis_ready.to_csv(output_file, index=False)

    print(f"\n✅  {output_file} saved")
    print(f"    Rows:    {len(analysis_ready)} weeks")
    print(f"    Columns: {len(analysis_ready.columns)}")
    print(f"    Outlier weeks (outlier_flag=1): {analysis_ready['outlier_flag'].sum()}")
    print(f"    Complete weeks (all 7 days):    {analysis_ready['complete_pollution_week'].sum()}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HHLab P1 — Preprocessing & Data Cleaning"
    )
    parser.add_argument(
        "--input",
        default="dataset_settimanale_finale 2.csv",
        help="Path to the raw CSV dataset",
    )
    parser.add_argument(
        "--output",
        default="analysis_ready.csv",
        help="Path for the cleaned output CSV",
    )
    args = parser.parse_args()
    main(input_file=args.input, output_file=args.output)