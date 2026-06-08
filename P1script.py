# Preprocessing and data cleaning — HHLab project, Milan 2023-2026
#
#
# What it does in plain terms:
#   - loads the raw weekly dataset exported from ARPA Lombardia / AREU
#   - collapses 7 daily pollutant readings into per-week summaries
#   - attaches season labels and Fourier harmonics (needed by P2's DLNM)
#   - flags and documents the W8/2025 outlier so downstream scripts know
#     to exclude it rather than silently absorbing it
#   - writes a single clean file (analysis_ready.csv) that P2 and P3
#     both read as their starting point — one source of truth
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
    """
    Auto-detect whether the CSV uses commas or semicolons as separator.

    Italian exports from ARPA and regional health systems often default to
    semicolons (because commas are used as decimal separators in Italian
    locale). Rather than hardcoding one convention, we just check the first
    line and let the file tell us what it uses.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """
    Convert a column to numeric, handling the Italian decimal comma.

    ARPA Lombardia files sometimes write 12,5 instead of 12.5. pandas
    doesn't handle this automatically, so we replace commas before parsing.
    Anything that still can't be converted becomes NaN rather than crashing.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def assign_season(week: int) -> str:
    """
    Map an ISO week number to a meteorological season label.

    We use a three-season scheme (winter / spring / summer) rather than
    four, because autumn and winter show very similar pollution and health
    patterns in Milan and splitting them would create small groups.

      Winter: weeks  1-13 and 40-53  (October-March)
      Spring: weeks 14-26            (April-June)
      Summer: weeks 27-39            (July-September)

    These boundaries were chosen to align with the pollution inversion
    season in the Po Valley, not the astronomical calendar.
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
    # We print the detected separator and column names early so that anyone
    # running this script can immediately spot if the wrong file was loaded
    # or if column names have changed between data exports.
    sep = detect_separator(input_file)
    raw = pd.read_csv(input_file, sep=sep)
    df = raw.copy()

    print(f"Separator detected: '{sep}'")
    print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # The raw file uses Italian abbreviations; rename to English for clarity
    # across the international team.
    rename_map = {
        "resp":      "respiratory_disease",
        "ili":       "ILI",
        "polm":      "pneumonia",
        "anno":      "year",
        "settimana": "week",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


    # ── 2. WEEKLY POLLUTANT MEANS (from 7 daily values) ───────────────────────
    # ARPA provides one row per day; we already received the data pre-aggregated
    # to weekly rows with seven day-columns per pollutant. We compute the mean,
    # max and 75th percentile for each week because the two analysis methods use
    # different exposure metrics: DLNM (P2) uses the weekly mean; Rc (P3) uses
    # the 75th percentile to identify "high-exposure" weeks.
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
            # No day-level columns found — assume the weekly mean is already
            # present in a column named e.g. "NO2_mean" and set n_days = 7.
            df[f"{p}_n_days"] = 7

    print("Weekly means computed: NO2_mean, PM25_mean, PM10_mean")

    # Parse all continuous outcome and meteorological columns to numeric
    # in one pass, using the same comma-safe conversion.
    numeric_cols = [
        "respiratory_disease", "ILI", "pneumonia",
        "temp_mean", "humidity_mean", "precip_total_mm",
        "wind_speed_mean", "pressure_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])


    # ── 3. SORT AND IDENTIFIERS ────────────────────────────────────────────────
    # week_index is a plain 1, 2, 3 … counter used by the DLNM spline term.
    # year_week is a human-readable label (e.g. "2023-W18") used in plots.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.sort_values(["year", "week"]).reset_index(drop=True)

    df["week_index"] = np.arange(1, len(df) + 1)
    df["year_week"] = (
        df["year"].astype(int).astype(str)
        + "-W"
        + df["week"].astype(int).astype(str).str.zfill(2)
    )

    # Parse or reconstruct the Monday date for each ISO week.
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
    # Season is used both as a descriptive grouping variable in the EDA and
    # as a set of binary flags that P2 can optionally include as covariates.
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
    # The DLNM sensitivity grid includes specifications that replace the natural
    # spline seasonal term with a Fourier series. We pre-compute the harmonic
    # terms here so P2 doesn't need to re-derive them from scratch.
    #
    # sin52/cos52 capture the annual cycle (52-week period).
    # sin26/cos26 capture the semi-annual cycle (26-week period).
    #
    # sin1/cos1 and sin2/cos2 are aliases kept for backward compatibility
    # with earlier P2 model scripts written before the naming was standardised.
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
    # We track how many days within each week are missing pollutant readings.
    # A week with all 7 days present is flagged as complete; anything less
    # is flagged as partial. This lets P2 and P3 run sensitivity checks
    # restricted to complete weeks without re-doing the missingness logic.
    #
    # W52/2025 had only 5 out of 7 measurement days (station under maintenance).
    # It is retained in the main dataset but flagged so it can be excluded from
    # sensitivity analyses if desired.
    for p in pollutants:
        df[f"missing_{p}_days"] = 7 - df[f"{p}_n_days"].fillna(0)

    df["partial_pollution_week"] = (
        df[[f"missing_{p}_days" for p in pollutants]].max(axis=1) > 0
    ).astype(int)
    df["complete_pollution_week"] = (
        df[[f"missing_{p}_days" for p in pollutants]].max(axis=1) == 0
    ).astype(int)

    df["w52_2025_flag"] = ((df["year"] == 2025) & (df["week"] == 52)).astype(int)

    print(
        f"Weeks with partial data: {df['partial_pollution_week'].sum()} "
        f"(of which W52/2025: {df['w52_2025_flag'].sum()})"
    )


    # ── 7. OUTLIER FLAG — W8/2025 ─────────────────────────────────────────────
    # Week 8 of 2025 (late February) shows respiratory ED visits roughly 64%
    # below the winter mean — far outside any plausible biological range.
    # After looking into it, we attributed this to a temporary under-reporting
    # issue in the AREU coding system rather than a real drop in demand.
    #
    # We flag it here rather than dropping it silently, so that P2 and P3
    # can filter with df[df['outlier_flag'] == 0] and the exclusion is
    # documented and reproducible, not buried inside model code.
    df["outlier_flag"] = ((df["year"] == 2025) & (df["week"] == 8)).astype(int)

    outlier_row = df[df["outlier_flag"] == 1]
    if len(outlier_row) > 0:
        row = outlier_row.iloc[0]
        print(
            f"Outlier W8/2025: respiratory={row['respiratory_disease']}, "
            f"ILI={row['ILI']}, pneumonia={row['pneumonia']}"
        )


    # ── 8. MISSING DATA CHECK ─────────────────────────────────────────────────
    # A quick sanity check printed to the console so we can confirm the
    # dataset is complete before handing it off to P2 and P3. The target
    # is zero missing values in outcomes and exposures for main-analysis weeks.
    outcome_cols  = ["respiratory_disease", "ILI", "pneumonia"]
    exposure_cols = ["NO2_mean", "PM25_mean", "PM10_mean"]
    meteo_cols    = ["temp_mean", "humidity_mean"]

    print("\nMissing data check:")
    for col in outcome_cols + exposure_cols + meteo_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            print(f"  {col}: {n_miss} missing ({100 * n_miss / len(df):.1f}%)")


    # ── 9. SELECT FINAL COLUMNS AND EXPORT ────────────────────────────────────
    # We keep only the columns that P2 or P3 actually need, to avoid passing
    # hundreds of intermediate columns downstream and making the handoff file
    # hard to navigate. The list is explicit so anyone reading the P2/P3 scripts
    # can trace each variable back to this step without guessing.
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
