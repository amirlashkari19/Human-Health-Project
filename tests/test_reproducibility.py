"""
Reproducibility tests for the Human Health Project (Milan, 2023-2026).

Test 1 — P1: analysis_ready.csv has 154 outlier-excluded rows.
Test 2 — P2: two-pollutant DLNM reproduces NO2 cum RR within ±5% of 2.80.
Test 3 — P3: Rc handoff table has 108 rows and NO2 has the most CI_low > 0.5 cells.
"""

import pathlib
import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]

# ── helpers ──────────────────────────────────────────────────────────────────

def load(relative_path: str) -> pd.DataFrame:
    path = REPO / relative_path
    assert path.exists(), f"File not found: {path}"
    return pd.read_csv(path)


# ── Test 1: P1 output row count ───────────────────────────────────────────────

def test_analysis_ready_outlier_excluded_rows():
    """analysis_ready.csv must have exactly 154 rows with outlier_flag == 0."""
    df = load("data/processed/analysis_ready.csv")

    assert "outlier_flag" in df.columns, "outlier_flag column missing"

    total_rows = len(df)
    assert total_rows == 155, (
        f"Expected 155 total rows (155 weeks), got {total_rows}"
    )

    clean_rows = len(df[df["outlier_flag"] == 0])
    assert clean_rows == 154, (
        f"Expected 154 outlier-excluded rows, got {clean_rows}"
    )


# ── Test 2: P2 two-pollutant DLNM NO2 cumulative RR ──────────────────────────

def test_two_pollutant_no2_cum_rr():
    """
    Two-pollutant DLNM (NO2 + PM10) for respiratory disease must reproduce
    NO2 cumulative RR within ±5% of the reference value 2.80.
    """
    df = load("p2_dlnm/two_pollutant_RR_table.csv")

    assert "cum_RR_NO2" in df.columns, "cum_RR_NO2 column missing"
    assert "model_id" in df.columns, "model_id column missing"

    row = df[df["model_id"] == "M2P_respiratory_disease"]
    assert len(row) == 1, (
        f"Expected 1 row for M2P_respiratory_disease, got {len(row)}"
    )

    reference_rr = 2.80
    tolerance = 0.05  # ±5%
    actual_rr = float(row["cum_RR_NO2"].iloc[0])

    assert abs(actual_rr - reference_rr) / reference_rr <= tolerance, (
        f"NO2 cum RR {actual_rr:.3f} deviates more than 5% from reference {reference_rr}"
    )


# ── Test 3: P3 Rc table shape and NO2 dominance ───────────────────────────────

def test_p3_rc_table_shape_and_no2_dominance():
    """
    Rc_weighted_CI_table.csv must have 108 rows, and NO2 must have
    the most rows where Rc_CI_low > 0.5 (CI excludes 0.5) compared to
    PM10 and PM2.5.
    """
    df = load("p3_rc_wmarm/tables/Rc_weighted_CI_table.csv")

    # Shape check
    assert len(df) == 108, (
        f"Expected 108 rows in Rc table, got {len(df)}"
    )

    required_cols = {"pollutant", "Rc_CI_low", "Rc_CI_high", "Rc_class"}
    missing_cols = required_cols - set(df.columns)
    assert not missing_cols, f"Missing columns: {missing_cols}"

    # NO2 dominance: most CI_low > 0.5 cells
    ci_counts = {}
    for pollutant in df["pollutant"].unique():
        sub = df[df["pollutant"] == pollutant]
        ci_counts[pollutant] = int((sub["Rc_CI_low"] > 0.5).sum())

    assert "NO2" in ci_counts, "NO2 not found in pollutant column"

    no2_count = ci_counts["NO2"]
    others_max = max(v for k, v in ci_counts.items() if k != "NO2")

    assert no2_count > others_max, (
        f"Expected NO2 to have the most CI_low > 0.5 cells "
        f"(NO2={no2_count}, others max={others_max})"
    )
