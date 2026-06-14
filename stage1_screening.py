"""
3S-GeoXAI Stage I - Variable Screening

Stage I of the 3S-GeoXAI framework: univariate Spearman screening of candidate
predictors against each respiratory outcome, followed by collinearity removal.
Stages II (MGWR) and III (Random Forest + SHAP) are not implemented here: the
health outcomes exist only at city level (no spatial grid for MGWR) and
n = 154 weekly observations is too small for a meaningful RF + SHAP analysis.

Input : analysis_ready.csv (from Phase 1, in the same folder)
Output: stage1_output/figures/*.png and *.pdf, stage1_output/tables/*.csv

Run:
    python stage1_screening.py                 # writes ./stage1_output
    python stage1_screening.py --out some/dir

Needs: pandas, numpy, scipy, matplotlib
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from scipy.stats import spearmanr


INPUT_CSV = Path(__file__).resolve().parent / "analysis_ready.csv"

# --- parameters ---
COLLINEARITY_THRESHOLD = 0.70
LAG_RANGE = [0, 1, 2, 3]
BOOTSTRAP_ITER = 1000
RNG_SEED = 20260517
EXCLUDE_OUTLIER = True   # drop the W8/2025 outlier (outlier_flag == 1)

OUTCOMES = ["respiratory_disease", "ILI", "pneumonia"]

PREDICTOR_GROUPS = {
    "Pollutants (mean)": ["NO2_mean", "PM25_mean", "PM10_mean"],
    "Pollutants (p75)":  ["NO2_p75", "PM25_p75", "PM10_p75"],
    "Pollutants (max)":  ["NO2_max", "PM25_max", "PM10_max"],
    "Meteorology":       ["temp_mean", "humidity_mean"],
    "Seasonality":       ["sin52", "cos52", "sin26", "cos26"],
}

PALETTE = {
    "Pollutants (mean)": "#8B2635",
    "Pollutants (p75)":  "#C44536",
    "Pollutants (max)":  "#E08E45",
    "Meteorology":       "#2E5A88",
    "Seasonality":       "#6B7F5C",
    "_dropped":          "#B8B8B8",
}
FONT_FAMILY = "DejaVu Serif"


# ------------------------------------------------------------------ #
#  Load
# ------------------------------------------------------------------ #

def load_data():
    """Read analysis_ready.csv and build the Spearman matrix used for the
    collinearity check. Stop with a clear message if the file is missing."""
    if not INPUT_CSV.exists():
        raise SystemExit(
            f"Could not find analysis_ready.csv at:\n  {INPUT_CSV}\n"
            "Put the Phase 1 output in the same folder as this script."
        )
    df = pd.read_csv(INPUT_CSV)

    if EXCLUDE_OUTLIER and "outlier_flag" in df.columns:
        n0 = len(df)
        df = df[df["outlier_flag"] == 0].reset_index(drop=True)
        print(f"[load] excluded {n0 - len(df)} outlier week(s) => n={len(df)}")

    # Spearman correlation matrix over the predictors present in the data,
    # used by the collinearity step below.
    preds = [p for p in all_predictors() if p in df.columns]
    spearman = df[preds].corr(method="spearman")
    print(f"[load] analysis_ready {df.shape}, spearman_matrix {spearman.shape}")
    return df, spearman


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #

def all_predictors():
    return [p for grp in PREDICTOR_GROUPS.values() for p in grp]


def predictor_group(name):
    for grp, members in PREDICTOR_GROUPS.items():
        if name in members:
            return grp
    return "Other"


def bootstrap_spearman_ci(x, y, n_iter=BOOTSTRAP_ITER, alpha=0.05):
    """Point rho plus percentile bootstrap 95% CI."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 5:
        rho = spearmanr(x, y)[0] if len(x) >= 2 else np.nan
        return rho, np.nan, np.nan
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(RNG_SEED)
    rhos = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, len(x), len(x))
        rhos[i] = spearmanr(x[idx], y[idx])[0]
    return rho, np.nanpercentile(rhos, 2.5), np.nanpercentile(rhos, 97.5)


def rank_predictors(df, predictors):
    """Contemporaneous |rho| ranking per outcome, with bootstrap CI."""
    ranked = {}
    for outcome in OUTCOMES:
        rows = []
        for p in predictors:
            rho, lo, hi = bootstrap_spearman_ci(df[p].values, df[outcome].values)
            rows.append({"predictor": p, "group": predictor_group(p),
                         "outcome": outcome, "rho": rho, "abs_rho": abs(rho),
                         "ci_low": lo, "ci_high": hi})
        ranked[outcome] = (pd.DataFrame(rows)
                           .sort_values("abs_rho", ascending=False)
                           .reset_index(drop=True))
    return ranked


def detect_collinear_pairs(spearman, predictors, threshold=COLLINEARITY_THRESHOLD):
    """Predictor-predictor pairs with |rho| > threshold."""
    rows = []
    for i, a in enumerate(predictors):
        for b in predictors[i + 1:]:
            if a in spearman.index and b in spearman.columns:
                r = spearman.loc[a, b]
                if abs(r) > threshold:
                    rows.append({"predictor_1": a, "predictor_2": b,
                                 "rho": r, "abs_rho": abs(r), "flagged": True})
    cols = ["predictor_1", "predictor_2", "rho", "abs_rho", "flagged"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_values("abs_rho", ascending=False)


def resolve_collinearity(ranked, pairs, threshold=COLLINEARITY_THRESHOLD):
    """Greedy retention: keep a predictor unless collinear with one already kept."""
    lookup = {}
    for _, r in pairs.iterrows():
        lookup[(r["predictor_1"], r["predictor_2"])] = r["abs_rho"]
        lookup[(r["predictor_2"], r["predictor_1"])] = r["abs_rho"]
    retained, decisions = [], []
    for _, row in ranked.iterrows():
        p = row["predictor"]
        conflict = next((k for k in retained
                         if lookup.get((p, k), 0.0) > threshold), None)
        if conflict is None:
            retained.append(p)
            decisions.append("retained")
        else:
            decisions.append(f"dropped (|rho|>{threshold:.2f} with {conflict})")
    out = ranked.copy()
    out["decision"] = decisions
    return out


def lagged_correlations(df, predictors, lags=LAG_RANGE):
    """Spearman rho between predictor(t) and outcome(t+lag), with bootstrap CI."""
    rows = []
    for outcome in OUTCOMES:
        for p in predictors:
            for lag in lags:
                x = df[p].values[:len(df) - lag]
                y = df[outcome].values[lag:]
                rho, lo, hi = bootstrap_spearman_ci(x, y)
                rows.append({"outcome": outcome, "predictor": p,
                             "group": predictor_group(p), "lag": lag,
                             "rho": rho, "abs_rho": abs(rho),
                             "ci_low": lo, "ci_high": hi, "n": len(x)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ #
#  Figures
# ------------------------------------------------------------------ #

def setup_style():
    plt.rcParams.update({
        "font.family": FONT_FAMILY, "font.size": 10,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#2A2A2A", "axes.linewidth": 0.8,
        "xtick.color": "#2A2A2A", "ytick.color": "#2A2A2A",
        "figure.facecolor": "white", "axes.facecolor": "#FAFAF7",
        "grid.color": "#E0E0DA", "grid.linewidth": 0.5,
    })


def _bar_colors(df):
    colors, hatches = [], []
    for _, row in df.iterrows():
        if row["decision"] == "retained":
            colors.append(PALETTE[row["group"]])
            hatches.append("")
        else:
            colors.append(PALETTE["_dropped"])
            hatches.append("//")
    return colors, hatches


def _legend_handles():
    h = [mpatches.Patch(facecolor=PALETTE[g], edgecolor="#2A2A2A", label=g)
         for g in PREDICTOR_GROUPS]
    h.append(mpatches.Patch(facecolor=PALETTE["_dropped"], edgecolor="#2A2A2A",
                            hatch="//", label="Dropped (collinear)"))
    return h


def plot_ranked_features(ranked, outcome, png, pdf):
    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=200)
    df = ranked.sort_values("abs_rho", ascending=True)
    colors, hatches = _bar_colors(df)
    bars = ax.barh(df["predictor"], df["abs_rho"], color=colors,
                   edgecolor="#2A2A2A", linewidth=0.7, hatch=hatches)

    lo, hi = df["ci_low"].abs().values, df["ci_high"].abs().values
    err = np.abs(np.vstack([df["abs_rho"] - np.minimum(lo, hi),
                            np.maximum(lo, hi) - df["abs_rho"]]))
    ax.errorbar(df["abs_rho"], df["predictor"], xerr=err, fmt="none",
                ecolor="#2A2A2A", elinewidth=0.6, capsize=2)

    for bar, rho in zip(bars, df["rho"]):
        sign = "+" if rho >= 0 else "−"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{sign}{abs(rho):.2f}", va="center", fontsize=9, color="#2A2A2A")

    ax.set_xlim(0, max(df["abs_rho"].max() * 1.25, 0.5))
    ax.set_xlabel("|Spearman rho| with outcome (with 95% bootstrap CI)")
    ax.set_title(f"Variable Screening — Stage I (3S-GeoXAI adaptation)\n"
                 f"Predictors ranked by |rho| with “{outcome}”",
                 loc="left", pad=14)
    ax.axvline(COLLINEARITY_THRESHOLD, ls="--", lw=0.8, color="#8B2635", alpha=0.4)
    ax.text(COLLINEARITY_THRESHOLD + 0.005, 0.5, f"|rho|={COLLINEARITY_THRESHOLD}",
            color="#8B2635", fontsize=8, alpha=0.7,
            transform=ax.get_xaxis_transform())
    ax.legend(handles=_legend_handles(), loc="lower right", fontsize=8,
              frameon=True, framealpha=0.95, edgecolor="#E0E0DA")
    fig.text(0.01, 0.005,
             "Stage I of 3S-GeoXAI: univariate Spearman screening + "
             "collinearity removal (|rho| > 0.70).\nFull 3S-GeoXAI not "
             "implemented: dataset lacks fine-scale spatial predictors & outcomes.",
             fontsize=7, color="#666666", style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def plot_combined_ranked(ranked_by_outcome, png, pdf):
    fig = plt.figure(figsize=(15, 7), dpi=200)
    gs = GridSpec(1, 3, figure=fig, wspace=0.35, left=0.06, right=0.98,
                  top=0.88, bottom=0.12)
    xmax = max(d["abs_rho"].max() for d in ranked_by_outcome.values()) * 1.25
    for col, outcome in enumerate(OUTCOMES):
        ax = fig.add_subplot(gs[0, col])
        df = ranked_by_outcome[outcome].sort_values("abs_rho", ascending=True)
        colors, hatches = _bar_colors(df)
        bars = ax.barh(df["predictor"], df["abs_rho"], color=colors,
                       edgecolor="#2A2A2A", linewidth=0.6, hatch=hatches)
        for bar, rho in zip(bars, df["rho"]):
            sign = "+" if rho >= 0 else "−"
            ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                    f"{sign}{abs(rho):.2f}", va="center", fontsize=7.5,
                    color="#2A2A2A")
        ax.set_xlim(0, xmax)
        ax.set_xlabel("|Spearman rho|", fontsize=9)
        ax.set_title(outcome.replace("_", " ").title(), loc="left",
                     fontsize=11, pad=8)
        ax.axvline(COLLINEARITY_THRESHOLD, ls="--", lw=0.6, color="#8B2635", alpha=0.3)
        ax.tick_params(axis="both", labelsize=8)
    fig.suptitle("Variable Screening — Stage I (3S-GeoXAI adaptation): "
                 "predictor ranking by outcome",
                 fontsize=13, fontweight="bold", x=0.06, ha="left", y=0.96)
    fig.legend(handles=_legend_handles(), loc="lower center", ncol=6,
               fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.01))
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


def plot_lagged_heatmap(lagged, png, pdf):
    fig, axes = plt.subplots(1, 3, figsize=(14, 8), dpi=200, sharey=True,
                             gridspec_kw={"wspace": 0.12})
    preds = all_predictors()
    vmax = max(lagged["rho"].abs().max(), 0.3)
    im = None
    for ax, outcome in zip(axes, OUTCOMES):
        sub = lagged[lagged["outcome"] == outcome]
        pivot = sub.pivot(index="predictor", columns="lag", values="rho").reindex(preds)
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(LAG_RANGE)))
        ax.set_xticklabels([f"lag {l}" for l in LAG_RANGE])
        if ax is axes[0]:
            ax.set_yticks(range(len(preds)))
            ax.set_yticklabels(preds, fontsize=9)
        ax.set_title(outcome.replace("_", " ").title(), loc="left", fontsize=11, pad=6)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7,
                            color="white" if abs(v) > vmax * 0.6 else "#2A2A2A")
    cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
    cbar.set_label("Spearman rho (signed)", fontsize=9)
    fig.suptitle("Lagged Spearman correlations — predictor (t) vs outcome (t + lag)",
                 fontsize=12, fontweight="bold", x=0.08, ha="left", y=0.97)
    fig.text(0.08, 0.005,
             "Appendix figure: lagged rho informs the DLNM cross-basis lag range.",
             fontsize=8, color="#666666", style="italic")
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------------ #
#  Driver
# ------------------------------------------------------------------ #

def main(out_dir):
    setup_style()
    fig_dir, tab_dir = out_dir / "figures", out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df, spearman = load_data()
    predictors = [p for p in all_predictors() if p in df.columns]
    missing = [p for p in all_predictors() if p not in df.columns]
    if missing:
        print(f"[warn] predictors absent from data, skipped: {missing}")
    print(f"[info] n_weeks={len(df)} | n_predictors={len(predictors)}")

    print("[1] contemporaneous ranking with bootstrap CI")
    ranked = rank_predictors(df, predictors)

    print(f"[2] collinearity detection (|rho| > {COLLINEARITY_THRESHOLD})")
    pairs = detect_collinear_pairs(spearman, predictors)
    pairs.to_csv(tab_dir / "collinearity_pairs.csv", index=False)
    print(f"    {len(pairs)} pair(s) flagged")

    print("[3] greedy retention rule")
    for outcome in OUTCOMES:
        ranked[outcome] = resolve_collinearity(ranked[outcome], pairs)
        ranked[outcome].to_csv(tab_dir / f"ranked_predictors_{outcome}.csv", index=False)
    pd.concat(ranked.values(), ignore_index=True).to_csv(
        tab_dir / "collinearity_decisions.csv", index=False)

    print(f"[4] lagged correlations (lags {LAG_RANGE})")
    lagged = lagged_correlations(df, predictors)
    lagged.to_csv(tab_dir / "lagged_correlations.csv", index=False)

    print("[5] figures")
    for outcome in OUTCOMES:
        plot_ranked_features(ranked[outcome], outcome,
                             fig_dir / f"ranked_features_{outcome}.png",
                             fig_dir / f"ranked_features_{outcome}.pdf")
    plot_combined_ranked(ranked, fig_dir / "ranked_features_combined.png",
                         fig_dir / "ranked_features_combined.pdf")
    plot_lagged_heatmap(lagged, fig_dir / "lagged_heatmap.png",
                        fig_dir / "lagged_heatmap.pdf")

    print("\n[done] Stage I screening complete.")
    print(f"       tables  -> {tab_dir}")
    print(f"       figures -> {fig_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent / "stage1_output",
                        help="output base directory (default: ./stage1_output)")
    args = parser.parse_args()
    main(args.out.resolve())
