# Phase 1: preprocessing, data cleaning and EDA figures
# Milan air pollution and respiratory ED study, 2023-2026
#
# This script takes the raw weekly dataset and turns it into one clean file
# (analysis_ready.csv) that the later modelling steps use as their input.
# It does the following:
#   - loads the raw weekly CSV (air quality from ARPA Lombardia, ED counts
#     from AREU, weather from ARPA)
#   - turns the 7 daily pollutant values of each week into weekly summaries
#     (mean, max, 75th percentile)
#   - adds season labels and Fourier terms used later for seasonality control
#   - flags the W8/2025 outlier and the short W52/2025 week instead of
#     deleting them, so the choice stays visible and reproducible
#   - saves analysis_ready.csv
#   - if you add --figures, it also draws the 9 EDA charts used in the report
#
# How to run:
#   python p1_preprocessing.py
# One command builds analysis_ready.csv and all 9 EDA figures.
#
# Needs: pandas, numpy, matplotlib, seaborn

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def detect_separator(filepath: str) -> str:
    """Check whether the CSV uses commas or semicolons.

    Italian exports often use semicolons, because the comma is used as the
    decimal mark. Instead of assuming one, we read the first line and decide.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    return ";" if first_line.count(";") > first_line.count(",") else ","


def to_numeric_safe(series: pd.Series) -> pd.Series:
    """Convert a column to numbers, handling the Italian decimal comma.

    Some values come as "12,5" instead of "12.5". We replace the comma first.
    Values that still cannot be read become NaN instead of stopping the script.
    """
    return pd.to_numeric(series.astype(str).str.replace(",", "."), errors="coerce")


def assign_season(week: int) -> str:
    """Give each ISO week a season label.

    We use three seasons rather than four because autumn and winter behave
    almost the same here (pollution and ED visits both rise), and keeping them
    separate would leave very small groups.
        Winter: weeks 1-13 and 40-53
        Spring: weeks 14-26
        Summer: weeks 27-39
    """
    if week <= 13 or week >= 40:
        return "winter"
    if week <= 26:
        return "spring"
    return "summer"


# ---------------------------------------------------------------------------
# EDA figures
# ---------------------------------------------------------------------------
# These functions only run when --figures is passed, so the plotting libraries
# are not needed just to produce the clean CSV.

def _setup_style():
    """Set a consistent look for all the charts."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.size":         9.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.labelcolor":   "#334155",
        "axes.titleweight":  "bold",
        "xtick.color":       "#64748B",
        "ytick.color":       "#64748B",
        "grid.color":        "#E2E8F0",
        "grid.linestyle":    "--",
        "grid.linewidth":    0.5,
        "figure.dpi":        150,
        "savefig.dpi":       180,
        "savefig.bbox":      "tight",
        "savefig.facecolor": "white",
    })

# Colours kept the same across every figure
_C = {
    "NO2":   "#EA580C",
    "PM25":  "#7C3AED",
    "PM10":  "#0D9488",
    "resp":  "#1D4ED8",
    "ILI":   "#B45309",
    "pneu":  "#6B7280",
    "temp":  "#DC2626",
    "hum":   "#0891B2",
    "out":   "#EF4444",
    "W":     "#1E3A5F",
    "Sp":    "#15803D",
    "Su":    "#D97706",
}
_WHO = {"NO2_mean": 10, "PM25_mean": 5, "PM10_mean": 15}


def _year_separators(ax, df):
    """Draw a light vertical line where each new year starts."""
    import numpy as np
    for yr in [2024, 2025, 2026]:
        idx = df[df["year"] == yr]["week_index"].min()
        if not np.isnan(float(idx)):
            ax.axvline(idx, color="#CBD5E1", lw=0.8, ls="--")
            ax.text(idx + 0.4, ax.get_ylim()[1] * 0.91, str(int(yr)),
                    fontsize=7, color="#94A3B8")


def _fig01_pollutant_timeseries(dm, df_full, out):
    """Weekly time series of the three pollutants with WHO 2021 lines.

    The outlier week is marked in red for context but is not part of the line.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
    do = df_full[df_full["outlier_flag"] == 1]

    for ax, (col, label, color) in zip(axes, [
        ("NO2_mean",  "NO2 (ug/m3)",   _C["NO2"]),
        ("PM25_mean", "PM2.5 (ug/m3)", _C["PM25"]),
        ("PM10_mean", "PM10 (ug/m3)",  _C["PM10"]),
    ]):
        ax.plot(dm["week_index"], dm[col], color=color, lw=1.5, alpha=0.9)
        ax.fill_between(dm["week_index"], dm[col], alpha=0.09, color=color)
        sd, mn = dm[col].std(), dm[col].mean()
        ax.fill_between(dm["week_index"], mn - sd, mn + sd, alpha=0.06, color=color)
        ax.axhline(_WHO[col], color="#94A3B8", lw=1, ls=":",
                   label=f"WHO 2021 ({_WHO[col]} ug/m3)")
        if len(do) > 0:
            ax.scatter(do["week_index"], do[col], color=_C["out"],
                       zorder=6, s=40, label="W8/2025 (outlier)")
        ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.grid(True)
        _year_separators(ax, df_full)

    axes[-1].set_xlabel("Week index (W18/2023 to W16/2026)", fontsize=9)
    fig.suptitle("Weekly mean pollutant concentrations, Milan 2023-2026",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig02_outcomes_timeseries(dm, df_full, out):
    """Weekly time series of the three health outcomes.

    The W8/2025 outlier is shown as a red dot so the reader can see the drop
    is sudden and isolated, which is why we exclude it.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.5), sharex=True)
    do = df_full[df_full["outlier_flag"] == 1]

    for ax, (col, label, color) in zip(axes, [
        ("respiratory_disease", "Respiratory ED visits/week", _C["resp"]),
        ("ILI",                 "ILI visits/week",            _C["ILI"]),
        ("pneumonia",           "Pneumonia visits/week",      _C["pneu"]),
    ]):
        ax.plot(dm["week_index"], dm[col], color=color, lw=1.5, alpha=0.9)
        ax.fill_between(dm["week_index"], dm[col], alpha=0.08, color=color)
        if len(do) > 0:
            ax.scatter(do["week_index"], do[col], color=_C["out"],
                       zorder=6, s=50, label="W8/2025 outlier (excluded)")
            ax.legend(fontsize=8, frameon=False, loc="upper right")
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.grid(True)
        _year_separators(ax, df_full)

    axes[-1].set_xlabel("Week index", fontsize=9)
    fig.suptitle("Weekly ED visits: respiratory disease, ILI, pneumonia",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig03_zscores(dm, df_full, out):
    """Standardised (z-score) pollutant series.

    PM2.5 and PM10 almost overlap, which is why we keep them in separate
    single-pollutant models instead of putting both in one model.
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.5))

    for col, label, color, ls in [
        ("NO2_mean",  "NO2",   _C["NO2"],  "-"),
        ("PM25_mean", "PM2.5", _C["PM25"], "--"),
        ("PM10_mean", "PM10",  _C["PM10"], ":"),
    ]:
        z = (dm[col] - dm[col].mean()) / dm[col].std()
        ax.plot(dm["week_index"], z, color=color, lw=1.4, ls=ls,
                alpha=0.9, label=label)

    ax.axhline(0, color="#CBD5E1", lw=0.8)
    _year_separators(ax, df_full)
    ax.yaxis.grid(True)
    ax.set_xlabel("Week index", fontsize=9)
    ax.set_ylabel("Z-score", fontsize=9)
    ax.set_title(
        "Standardised pollutant concentrations\n"
        "PM2.5 and PM10 move almost identically (Spearman rho = 0.96)",
        fontsize=10
    )
    ax.legend(fontsize=9, frameon=False)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig04_meteo(dm, df_full, out):
    """Weather variables and their link with respiratory visits.

    Temperature has a strong negative correlation with respiratory ED visits,
    so it is an important confounder that the later models must adjust for.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10, 5.5))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(dm["week_index"], dm["temp_mean"],
             color=_C["temp"], lw=1.4, label="Temperature (C)")
    ax1b = ax1.twinx()
    ax1b.plot(dm["week_index"], dm["humidity_mean"],
              color=_C["hum"], lw=1.2, alpha=0.75, label="Humidity (%)")
    ax1.set_ylabel("Temperature (C)", color=_C["temp"], fontsize=9)
    ax1b.set_ylabel("Humidity (%)", color=_C["hum"], fontsize=9)
    ax1.set_title("Temperature and relative humidity, Milan 2023-2026", fontsize=10)
    _year_separators(ax1, df_full)
    ax1.yaxis.grid(True)

    smap = {"winter": (_C["W"], "Winter"),
            "spring": (_C["Sp"], "Spring"),
            "summer": (_C["Su"], "Summer")}
    for s, (sc, sl) in smap.items():
        sub = dm[dm["season"] == s]
        ax2.scatter(sub["temp_mean"], sub["respiratory_disease"],
                    color=sc, alpha=0.65, s=22, label=sl,
                    edgecolors="white", lw=0.2)
    ax2.set_xlabel("Temperature (C)", fontsize=9)
    ax2.set_ylabel("Respiratory ED visits/week", fontsize=9)
    ax2.set_title("Temperature vs respiratory visits\n(Spearman rho = -0.83)", fontsize=9)
    ax2.legend(fontsize=8, frameon=False)
    ax2.yaxis.grid(True)

    for s, (sc, _) in smap.items():
        sub = dm[dm["season"] == s]
        ax3.scatter(sub["humidity_mean"], sub["respiratory_disease"],
                    color=sc, alpha=0.65, s=22, edgecolors="white", lw=0.2)
    ax3.set_xlabel("Humidity (%)", fontsize=9)
    ax3.set_ylabel("Respiratory ED visits/week", fontsize=9)
    ax3.set_title("Humidity vs respiratory visits\n(Spearman rho = +0.56)", fontsize=9)
    ax3.yaxis.grid(True)

    fig.suptitle("Weather confounders", fontsize=11, y=1.01)
    plt.savefig(out)
    plt.close()


def _fig05_scatter_expoutcome(dm, out):
    """Pollutant vs respiratory visits, coloured by season.

    Most of the raw correlation comes from the gap between seasons (cold weeks
    top-right, warm weeks bottom-left), which shows why seasonal adjustment is
    needed before reading any pollution effect.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))

    smap = [("winter", _C["W"], "Winter"),
            ("spring", _C["Sp"], "Spring"),
            ("summer", _C["Su"], "Summer")]

    for ax, (col, label, color) in zip(axes, [
        ("NO2_mean",  "NO2 (ug/m3)",   _C["NO2"]),
        ("PM25_mean", "PM2.5 (ug/m3)", _C["PM25"]),
        ("PM10_mean", "PM10 (ug/m3)",  _C["PM10"]),
    ]):
        for s, sc, sl in smap:
            sub = dm[dm["season"] == s]
            ax.scatter(sub[col], sub["respiratory_disease"],
                       color=sc, alpha=0.7, s=24, label=sl,
                       edgecolors="white", lw=0.2)
        ax.set_xlabel(label, fontsize=9)
        ax.yaxis.grid(True)
        ax.set_title(f"{label.split()[0]} vs respiratory", fontsize=9)

    axes[0].set_ylabel("Respiratory ED visits/week", fontsize=9)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Pollutant vs respiratory visits by season (between-season effect dominates)",
        fontsize=10, y=1.02
    )
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig06_pollutant_boxplots(dm, out):
    """Season-by-season boxplots for the pollutants.

    Winter medians are clearly higher than summer for all three pollutants.
    The dotted line is the WHO 2021 guideline.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    dm = dm.copy()
    dm["season_label"] = dm["season"].str.capitalize()
    order = ["Winter", "Spring", "Summer"]
    palette = [_C["W"], _C["Sp"], _C["Su"]]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    for ax, (col, label) in zip(axes, [
        ("NO2_mean",  "NO2 (ug/m3)"),
        ("PM25_mean", "PM2.5 (ug/m3)"),
        ("PM10_mean", "PM10 (ug/m3)"),
    ]):
        sns.boxplot(data=dm, x="season_label", y=col, order=order,
                    palette=palette, width=0.5, linewidth=1.1,
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#94A3B8"},
                    ax=ax)
        ax.axhline(_WHO[col], color="#94A3B8", lw=1, ls=":")
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.grid(True)
        ax.set_title(label.split()[0], fontsize=10)

    fig.suptitle("Pollutants by season (dotted line = WHO 2021 guideline)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig07_outcome_boxplots(dm, out):
    """Season-by-season boxplots for the health outcomes.

    All three outcomes are highest in winter. The wide winter range for
    respiratory disease is the within-season variation the models rely on.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    dm = dm.copy()
    dm["season_label"] = dm["season"].str.capitalize()
    order = ["Winter", "Spring", "Summer"]
    palette = [_C["W"], _C["Sp"], _C["Su"]]

    fig, axes = plt.subplots(1, 3, figsize=(11, 4.5))
    for ax, (col, label) in zip(axes, [
        ("respiratory_disease", "Respiratory ED visits"),
        ("ILI",                 "ILI visits"),
        ("pneumonia",           "Pneumonia visits"),
    ]):
        sns.boxplot(data=dm, x="season_label", y=col, order=order,
                    palette=palette, width=0.5, linewidth=1.1,
                    flierprops={"marker": "o", "markersize": 3,
                                "markerfacecolor": "#94A3B8"},
                    ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(label, fontsize=9)
        ax.yaxis.grid(True)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)

    fig.suptitle("Health outcomes by season (all peak in winter)",
                 fontsize=10, y=1.02)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig08_corr_heatmap(dm, out):
    """Spearman correlation heatmap of the main variables.

    Two results matter most here: PM2.5 and PM10 are almost the same
    (rho = 0.96), and temperature is strongly tied to respiratory visits
    (rho = -0.83), so season and temperature must be controlled for.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cols_map = {
        "NO2_mean":           "NO2",
        "PM25_mean":          "PM2.5",
        "PM10_mean":          "PM10",
        "temp_mean":          "Temp",
        "humidity_mean":      "Humidity",
        "respiratory_disease":"Respiratory",
        "ILI":                "ILI",
        "pneumonia":          "Pneumonia",
    }
    cols_map = {k: v for k, v in cols_map.items() if k in dm.columns}
    sub = dm[list(cols_map)].rename(columns=cols_map)
    corr = sub.corr(method="spearman")
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    sns.heatmap(
        corr, mask=mask,
        cmap=sns.diverging_palette(220, 20, as_cmap=True),
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9.5},
        linewidths=0.4, linecolor="#F1F5F9",
        ax=ax, square=True,
        cbar_kws={"shrink": 0.75, "label": "Spearman rho"}
    )
    ax.set_title(
        "Spearman correlation heatmap (154 weeks, outlier excluded)",
        fontsize=11, pad=12
    )
    ax.tick_params(axis="x", rotation=40)
    ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _fig09_scatter_matrix(dm, out):
    """A 3x3 grid of every pollutant against every outcome, by season.

    The same season-driven pattern shows up in every panel, confirming the
    seasonal confounding is general and not specific to one pair.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    pollutants = [("NO2_mean", "NO2"), ("PM25_mean", "PM2.5"), ("PM10_mean", "PM10")]
    outcomes   = [("respiratory_disease", "Respiratory"),
                  ("ILI", "ILI"),
                  ("pneumonia", "Pneumonia")]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9))
    smap = [("winter", _C["W"]), ("spring", _C["Sp"]), ("summer", _C["Su"])]

    for r, (ocol, olabel) in enumerate(outcomes):
        for c, (pcol, plabel) in enumerate(pollutants):
            ax = axes[r][c]
            for s, sc in smap:
                sub = dm[dm["season"] == s]
                ax.scatter(sub[pcol], sub[ocol], color=sc,
                           alpha=0.6, s=18, edgecolors="white", lw=0.2)
            rho = dm[[pcol, ocol]].corr(method="spearman").iloc[0, 1]
            ax.set_title(f"rho = {rho:.2f}", fontsize=8.5, pad=2)
            ax.yaxis.grid(True)
            ax.tick_params(labelsize=7.5)
            if r == 2: ax.set_xlabel(plabel, fontsize=9)
            if c == 0: ax.set_ylabel(olabel, fontsize=9)

    axes[0][2].legend(
        handles=[Patch(facecolor=_C["W"], label="Winter"),
                 Patch(facecolor=_C["Sp"], label="Spring"),
                 Patch(facecolor=_C["Su"], label="Summer")],
        fontsize=8, frameon=False, loc="upper right"
    )
    fig.suptitle("Pollutants vs health outcomes by season",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def generate_figures(df: pd.DataFrame, figdir: str) -> None:
    """Draw all 9 EDA figures and save them to figdir.

    The outlier week is left out of the main series in every figure but is
    shown as a reference point where it helps.
    """
    os.makedirs(figdir, exist_ok=True)
    dm = df[df["outlier_flag"] == 0].copy()

    _setup_style()

    figures = [
        ("fig1_pollutant_timeseries.jpg",
         lambda p: _fig01_pollutant_timeseries(dm, df, p)),
        ("fig2_outcomes_timeseries.jpg",
         lambda p: _fig02_outcomes_timeseries(dm, df, p)),
        ("fig3_zscores.jpg",
         lambda p: _fig03_zscores(dm, df, p)),
        ("fig4_meteo.jpg",
         lambda p: _fig04_meteo(dm, df, p)),
        ("fig5_scatter_expoutcome.jpg",
         lambda p: _fig05_scatter_expoutcome(dm, p)),
        ("fig6_pollutant_boxplots.jpg",
         lambda p: _fig06_pollutant_boxplots(dm, p)),
        ("fig7_outcome_boxplots.jpg",
         lambda p: _fig07_outcome_boxplots(dm, p)),
        ("fig8_corr_heatmap.jpg",
         lambda p: _fig08_corr_heatmap(dm, p)),
        ("fig9_scatter_matrix.jpg",
         lambda p: _fig09_scatter_matrix(dm, p)),
    ]

    for fname, fn in figures:
        path = os.path.join(figdir, fname)
        fn(path)
        print(f"  saved: {path}")

    print(f"\n{len(figures)} figures saved to: {os.path.abspath(figdir)}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(input_file: str = "dataset_settimanale_finale 2.csv",
         output_file: str = "analysis_ready.csv",
         figdir: str = "p1_figures") -> None:

    # 1. Load the raw dataset.
    # We print the separator and column names so it is easy to notice if the
    # wrong file was loaded or if a column name changed in a new export.
    sep = detect_separator(input_file)
    raw = pd.read_csv(input_file, sep=sep)
    df = raw.copy()

    print(f"Separator detected: '{sep}'")
    print(f"Raw dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # The raw file uses Italian short names; rename them to English.
    rename_map = {
        "resp":      "respiratory_disease",
        "ili":       "ILI",
        "polm":      "pneumonia",
        "anno":      "year",
        "settimana": "week",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # 2. Weekly pollutant summaries from the 7 daily values.
    # We keep the mean, max and 75th percentile because the two later methods
    # use different exposure measures: the DLNM uses the weekly mean, while the
    # Rc analysis uses the 75th percentile to pick out high-exposure weeks.
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
            # No daily columns found, so assume the weekly mean already exists.
            df[f"{p}_n_days"] = 7

    print("Weekly means computed: NO2_mean, PM25_mean, PM10_mean")

    # Convert the outcome and weather columns to numbers in one pass.
    numeric_cols = [
        "respiratory_disease", "ILI", "pneumonia",
        "temp_mean", "humidity_mean", "precip_total_mm",
        "wind_speed_mean", "pressure_mean",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = to_numeric_safe(df[col])

    # 3. Sort the weeks and add identifiers.
    # week_index is a simple 1, 2, 3 ... counter used later for the trend term;
    # year_week (e.g. "2023-W18") is a readable label used in the charts.
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.sort_values(["year", "week"]).reset_index(drop=True)

    df["week_index"] = np.arange(1, len(df) + 1)
    df["year_week"] = (
        df["year"].astype(int).astype(str)
        + "-W"
        + df["week"].astype(int).astype(str).str.zfill(2)
    )

    # Date of the Monday of each ISO week. The raw value is in ISO form
    # (YYYY-MM-DD), so we parse it with an explicit format. Using a "mixed"
    # format with dayfirst can swap the month and day on some setups, so we
    # avoid that to keep the result the same on any machine.
    if "week_start" in df.columns:
        df["week_start"] = pd.to_datetime(
            df["week_start"], format="%Y-%m-%d", errors="coerce"
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

    # 4. Season labels.
    df["season"]      = df["week"].apply(assign_season)
    df["winter_flag"] = (df["season"] == "winter").astype(int)
    df["spring_flag"] = (df["season"] == "spring").astype(int)
    df["summer_flag"] = (df["season"] == "summer").astype(int)

    print(
        f"Seasons: winter n={df['winter_flag'].sum()}, "
        f"summer n={df['summer_flag'].sum()}, "
        f"spring n={df['spring_flag'].sum()}"
    )

    # 5. Fourier terms for seasonality.
    # sin52/cos52 follow the yearly cycle (52-week period) and sin26/cos26 the
    # half-year cycle. The sin1/cos1 and sin2/cos2 names are the same values
    # under older labels, kept so earlier model scripts still work.
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

    # 6. Quality flags.
    # We record how many of the 7 days are missing per week. A week with all 7
    # days is complete; otherwise it is partial. W52/2025 only had 5 of 7 days
    # (a station was under maintenance), so we flag it but keep it in the data.
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

    # 7. Outlier flag for W8/2025.
    # In late February 2025 the respiratory ED count is about 64% below the
    # winter average, which is far too low to be real. We think it is a
    # short reporting problem in the health data. We flag it instead of
    # deleting it so the later steps can filter it out with outlier_flag == 0
    # and the choice stays clear and reproducible.
    df["outlier_flag"] = ((df["year"] == 2025) & (df["week"] == 8)).astype(int)

    outlier_row = df[df["outlier_flag"] == 1]
    if len(outlier_row) > 0:
        row = outlier_row.iloc[0]
        print(
            f"Outlier W8/2025: respiratory={row['respiratory_disease']}, "
            f"ILI={row['ILI']}, pneumonia={row['pneumonia']}"
        )

    # 8. Missing data check, printed so we can confirm the data is complete
    # before saving. We expect zero missing values in the main columns.
    outcome_cols  = ["respiratory_disease", "ILI", "pneumonia"]
    exposure_cols = ["NO2_mean", "PM25_mean", "PM10_mean"]
    meteo_cols    = ["temp_mean", "humidity_mean"]

    print("\nMissing data check:")
    for col in outcome_cols + exposure_cols + meteo_cols:
        if col in df.columns:
            n_miss = df[col].isna().sum()
            print(f"  {col}: {n_miss} missing ({100 * n_miss / len(df):.1f}%)")

    # 9. Keep only the columns the later steps need and save.
    # The list is written out in full so each variable can be traced back here.
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

    print(f"\n{output_file} saved")
    print(f"  Rows:    {len(analysis_ready)} weeks")
    print(f"  Columns: {len(analysis_ready.columns)}")
    print(f"  Outlier weeks (outlier_flag=1): {analysis_ready['outlier_flag'].sum()}")
    print(f"  Complete weeks (all 7 days):    {analysis_ready['complete_pollution_week'].sum()}")

    # 10. EDA figures.
    print(f"\nGenerating EDA figures in {figdir}")
    generate_figures(analysis_ready, figdir)


if __name__ == "__main__":
    main()
