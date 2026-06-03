"""
HHLab — P3 (Rc / WMARM-like Adaptation Analyst)
================================================
Implements the full weekly city-level adaptation of APHREH-ADSMap on the
Milan analysis_ready dataset (n = 154 weeks after W8/2025 outlier exclusion).

Pipeline (matches the P3 Trello cards Day 2 → Day 9-10):

  1. Setup & parameters            (Day 2)
  2. Exposure classification       (Day 2-3)  -> rc_wmarm/tables/exposure_classification.csv
  3. Weekly incidence + lagging    (Day 3-4)  -> rc_wmarm/tables/lagged_incidence.csv
  4. Baseline & differential inc.  (Day 4-5)  -> rc_wmarm/tables/differential_incidence.csv
                                                rc_wmarm/tables/baseline_window_expansions.csv
  5. Exposure weights              (Day 5-6)  -> rc_wmarm/tables/exposure_weights.csv
                                                rc_wmarm/figures/exposure_weights_timeline.png
  6. Bootstrap weighted Wilcoxon   (Day 6-7)  -> rc_wmarm/tables/bootstrap_results.csv
  7. Rc computation                (Day 7-8)  -> rc_wmarm/tables/weekly_citylevel_Rc_by_pollutant_outcome_year.csv
                                                rc_wmarm/tables/Rc_weighted_CI_table.csv
  8. WMARM-like score              (Day 8-9)  -> rc_wmarm/tables/citylevel_WMARM_like_summary.csv
  9. Sensitivity surfaces          (Day 9-10) -> rc_wmarm/sensitivity/sensitivity_grid.csv
                                                rc_wmarm/figures/WMARM_like_surface_*.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from math import erf, sqrt

# =========================================================================
# 1. SETUP & PARAMETERS  (Day 2 card)
# =========================================================================
HERE        = os.path.dirname(os.path.abspath(__file__))
ROOT        = os.path.dirname(HERE)
INPUT_CSV = os.path.join(HERE, 'analysis_ready.csv')
TABLES      = os.path.join(ROOT, 'rc_wmarm', 'tables')
FIGURES     = os.path.join(ROOT, 'rc_wmarm', 'figures')
SENSITIVITY = os.path.join(ROOT, 'rc_wmarm', 'sensitivity')
DATA_PROC   = os.path.join(ROOT, 'data', 'processed')
for d in (TABLES, FIGURES, SENSITIVITY, DATA_PROC):
    os.makedirs(d, exist_ok=True)

POLLUTANTS  = ['NO2_mean', 'PM25_mean', 'PM10_mean']
POL_LABELS  = {'NO2_mean': 'NO2', 'PM25_mean': 'PM2.5', 'PM10_mean': 'PM10'}
OUTCOMES    = ['respiratory_disease', 'ILI', 'pneumonia']
PRIMARY_OUT = 'respiratory_disease'

PC_DEFAULT  = 0.75
PC_GRID     = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90]
LAGS        = [0, 1, 2]
BASELINE_WINDOWS = [3, 4, 5]   # try ±3, expand to ±4, ±5
N_BOOT      = 1000
NOISE_SD    = 0.05
POPULATION  = 1_407_044         # Comune di Milano (user-supplied)
RNG_SEED    = 42

# 9-class Rc interpretation scale (defined locally — APHREH-ADSMap original
# is multi-cohort spatial; we adapt to a city-level scalar).  Bands are on
# |Vul| ∈ [0, 1] where Vul = weighted-mean rank-biserial (0.5 = null).
#   |Vul-0.5| -> deviation from null direction.
#   We map deviation [0, 0.5] to the 9 classes evenly.
RC_CLASS_BANDS = [
    (0.000, 0.050, 'Negligible'),
    (0.050, 0.100, 'Very low'),
    (0.100, 0.150, 'Low'),
    (0.150, 0.200, 'Moderate-low'),
    (0.200, 0.250, 'Moderate'),
    (0.250, 0.300, 'Moderate-high'),
    (0.300, 0.350, 'High'),
    (0.350, 0.400, 'Very high'),
    (0.400, 0.501, 'Critical'),
]

def classify_rc(vul):
    dev = abs(vul - 0.5)
    for lo, hi, label in RC_CLASS_BANDS:
        if lo <= dev < hi:
            return label
    return 'Unclassified'

# Load and clean
df = pd.read_csv(INPUT_CSV)
df = df[df['outlier_flag'] == 0].sort_values('week_index').reset_index(drop=True)
print(f"[Setup] Loaded analysis_ready.csv  n = {len(df)} weeks "
      f"({df['year_week'].iloc[0]} -> {df['year_week'].iloc[-1]})")
print(f"[Setup] Population denominator    = {POPULATION:,} (Milan municipality)")
print(f"[Setup] Years present              = {sorted(df['year'].unique().tolist())}")
print(f"[Setup] 2023 incomplete (starts W18): "
      f"{(df['year']==2023).sum()} weeks")
print(f"[Setup] W52/2025 partial flag     = {df['w52_2025_flag'].sum()}")

# Snapshot the parameters used (good practice for reproducibility)
PARAMS = dict(
    pollutants=POLLUTANTS, outcomes=OUTCOMES, primary_outcome=PRIMARY_OUT,
    pc_default=PC_DEFAULT, pc_grid=PC_GRID, lags=LAGS,
    baseline_windows=BASELINE_WINDOWS, n_boot=N_BOOT, noise_sd=NOISE_SD,
    population=POPULATION, seed=RNG_SEED,
    rc_class_bands=[[lo, hi, lab] for lo, hi, lab in RC_CLASS_BANDS],
    n_weeks=int(len(df)),
    period=f"{df['year_week'].iloc[0]} -> {df['year_week'].iloc[-1]}"
)
with open(os.path.join(DATA_PROC, 'p3_parameters.json'), 'w') as f:
    json.dump(PARAMS, f, indent=2)

# =========================================================================
# 2. EXPOSURE CLASSIFICATION  (Day 2-3 card)
# =========================================================================
print("\n[Exposure] Classifying weeks per pollutant per year (yearly p75)...")
exposure_rows = []
for pol in POLLUTANTS:
    for yr, sub in df.groupby('year'):
        thr = float(np.quantile(sub[pol].dropna(), PC_DEFAULT))
        for _, row in sub.iterrows():
            v = row[pol]
            if pd.isna(v):
                exposed = np.nan
            else:
                exposed = int(v >= thr)
            exposure_rows.append({
                'year_week': row['year_week'],
                'week_index': int(row['week_index']),
                'year': int(yr),
                'pollutant': POL_LABELS[pol],
                'value': float(v) if pd.notna(v) else np.nan,
                'threshold_pc75': round(thr, 3),
                'exposed': exposed,
                'incomplete_year_2023': int(yr == 2023),
                'partial_W52_2025': int(row['w52_2025_flag']),
            })

exposure_df = pd.DataFrame(exposure_rows)
exposure_df.to_csv(os.path.join(TABLES, 'exposure_classification.csv'), index=False)
print(exposure_df.groupby(['pollutant', 'year', 'exposed']).size()
      .unstack(fill_value=0).rename(columns={0.0: 'non_exposed', 1.0: 'exposed'}))

# =========================================================================
# 3. WEEKLY INCIDENCE + LAGGING  (Day 3-4 card)
# =========================================================================
print("\n[Incidence] Computing weekly incidence and lagged outcomes...")
inc = df[['year_week', 'week_index', 'year']].copy()
for o in OUTCOMES:
    inc[f'{o}_count'] = df[o].values
    inc[f'{o}_inc']   = df[o].values / POPULATION
    for L in LAGS:
        # lag-L outcome: incidence L weeks AFTER the target week
        if L == 0:
            inc[f'{o}_inc_lag{L}'] = inc[f'{o}_inc'].values
        else:
            shifted = np.concatenate([inc[f'{o}_inc'].values[L:], [np.nan]*L])
            inc[f'{o}_inc_lag{L}'] = shifted

inc.to_csv(os.path.join(TABLES, 'lagged_incidence.csv'), index=False)
print(f"  Saved lagged_incidence.csv  shape = {inc.shape}")
print(f"  Drop-due-to-lag preview: lag1 -> last week {inc.iloc[-1]['year_week']} "
      f"NaN = {pd.isna(inc.iloc[-1][f'{PRIMARY_OUT}_inc_lag1'])}")

# =========================================================================
# 4. BASELINE & DIFFERENTIAL INCIDENCE  (Day 4-5 card)
# =========================================================================
print("\n[Baseline] Building baseline windows (median of non-exposed neighbors)...")

# Map (week_index, pollutant) -> exposed flag for fast lookup
exp_map = {(int(r['week_index']), r['pollutant']): r['exposed']
           for _, r in exposure_df.iterrows()}

diff_rows = []
expansion_log = []

for pol_col in POLLUTANTS:
    pol_lab = POL_LABELS[pol_col]
    for o in OUTCOMES:
        for L in LAGS:
            inc_col = f'{o}_inc_lag{L}'
            for _, row in inc.iterrows():
                t   = int(row['week_index'])
                yr  = int(row['year'])
                inc_t = row[inc_col]
                if pd.isna(inc_t):
                    continue
                exposed_t = exp_map.get((t, pol_lab), np.nan)
                if pd.isna(exposed_t):
                    continue

                # Find baseline window (try ±3, expand if needed)
                used_window = None
                neighbor_inc = []
                for W in BASELINE_WINDOWS:
                    candidates = [tt for tt in range(t-W, t+W+1)
                                  if tt != t and 1 <= tt <= int(inc['week_index'].max())]
                    # restrict to non-exposed neighbors with valid lag-L incidence
                    vals = []
                    for tt in candidates:
                        ex = exp_map.get((tt, pol_lab), np.nan)
                        if ex == 0:
                            r2 = inc[inc['week_index'] == tt]
                            if len(r2) and pd.notna(r2[inc_col].iloc[0]):
                                vals.append(float(r2[inc_col].iloc[0]))
                    if len(vals) >= 2:
                        used_window = W
                        neighbor_inc = vals
                        break

                if used_window is None:
                    expansion_log.append({
                        'pollutant': pol_lab, 'outcome': o, 'lag': L,
                        'week_index': t, 'year_week': row['year_week'],
                        'note': 'no baseline available even at +/-5',
                    })
                    continue

                if used_window > 3:
                    expansion_log.append({
                        'pollutant': pol_lab, 'outcome': o, 'lag': L,
                        'week_index': t, 'year_week': row['year_week'],
                        'note': f'baseline expanded to +/-{used_window}',
                    })

                baseline = float(np.median(neighbor_inc))
                delta    = float(inc_t - baseline)
                diff_rows.append({
                    'pollutant': pol_lab, 'outcome': o, 'year': yr, 'lag': L,
                    'week_index': t, 'year_week': row['year_week'],
                    'exposed': int(exposed_t),
                    'inc_lag': inc_t, 'baseline_inc': baseline,
                    'delta_inc': delta, 'baseline_window': used_window,
                    'n_neighbors': len(neighbor_inc),
                })

diff_df = pd.DataFrame(diff_rows)
diff_df.to_csv(os.path.join(TABLES, 'differential_incidence.csv'), index=False)
exp_log_df = pd.DataFrame(expansion_log)
exp_log_df.to_csv(os.path.join(TABLES, 'baseline_window_expansions.csv'), index=False)
print(f"  Saved differential_incidence.csv  rows = {len(diff_df):,}")
print(f"  Baseline expansions logged       rows = {len(exp_log_df):,}")
print("  Window distribution:")
print(diff_df['baseline_window'].value_counts().sort_index().to_string())

# =========================================================================
# 5. EXPOSURE WEIGHTS  (Day 5-6 card)
# =========================================================================
print("\n[Weights] Computing log-distance exposure weights...")
weight_rows = []
for pol_col in POLLUTANTS:
    pol_lab = POL_LABELS[pol_col]
    # For each year, get the threshold and compute distances
    for yr, sub in df.groupby('year'):
        thr = float(np.quantile(sub[pol_col].dropna(), PC_DEFAULT))
        # log(distance + 1) for exposed weeks; default 0.01 otherwise
        log_d_exposed = []
        for _, r in sub.iterrows():
            v = r[pol_col]
            if pd.notna(v) and v >= thr:
                log_d_exposed.append(np.log1p(v - thr))
        max_log = max(log_d_exposed) if log_d_exposed else 1.0
        for _, r in sub.iterrows():
            v = r[pol_col]
            if pd.isna(v):
                w = np.nan
            elif v >= thr:
                w = np.log1p(v - thr) / max_log if max_log > 0 else 1.0
                w = max(w, 0.01)
            else:
                w = 0.01
            weight_rows.append({
                'year_week': r['year_week'],
                'week_index': int(r['week_index']),
                'year': int(yr),
                'pollutant': pol_lab,
                'value': float(v) if pd.notna(v) else np.nan,
                'threshold_pc75': round(thr, 3),
                'distance_from_thr': float(v - thr) if pd.notna(v) else np.nan,
                'exposed': int(v >= thr) if pd.notna(v) else np.nan,
                'weight': float(w) if pd.notna(w) else np.nan,
            })

weights_df = pd.DataFrame(weight_rows)
weights_df.to_csv(os.path.join(TABLES, 'exposure_weights.csv'), index=False)

# Plot weight timeline
fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)

for ax, pol_lab in zip(axes, ['NO2', 'PM2.5', 'PM10']):
    sub = weights_df[weights_df['pollutant'] == pol_lab].sort_values('week_index')

    x = sub['week_index'].to_numpy()
    y = sub['weight'].to_numpy()

    ax.plot(x, y, color='steelblue', lw=1.2)
    ax.fill_between(x, 0, y, alpha=0.3, color='steelblue')
    
    ax.set_ylabel(f'{pol_lab}\nweight'); ax.set_ylim(0, 1.05)
    ax.axhline(0.01, color='gray', lw=0.5, ls=':')
axes[-1].set_xlabel('Week index (1 = W18/2023)')
fig.suptitle('Exposure weights over time (yearly p75 threshold, log-distance, normalized)',
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES, 'exposure_weights_timeline.png'), dpi=130)
plt.close(fig)
print(f"  Saved exposure_weights.csv + exposure_weights_timeline.png")

# =========================================================================
# 6. BOOTSTRAP WEIGHTED WILCOXON (Mann-Whitney U)  (Day 6-7 card)
# =========================================================================
print("\n[Bootstrap] Running 1000-iter weighted Mann-Whitney bootstraps...")

# Helper: weighted Mann-Whitney U statistic + asymptotic p-value
def normal_cdf(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))

def weighted_mwu(x, wx, y, wy):
    """
    x = exposed values, wx = exposed weights (>0)
    y = non-exposed values, wy = non-exposed weights (>0)
    Returns (U_w, p_two_sided, r_effect_size, direction)
    Direction: +1 if exposed > non-exposed (higher Δinc with exposure), else -1.
    """
    x = np.asarray(x); y = np.asarray(y)
    wx = np.asarray(wx); wy = np.asarray(wy)
    nx, ny = len(x), len(y)
    if nx == 0 or ny == 0:
        return np.nan, np.nan, np.nan, 0
    # Pairwise contribution: w_i * w_j * I(x_i > y_j)  + 0.5 ties
    diff = x[:, None] - y[None, :]
    sign = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    w_pair = wx[:, None] * wy[None, :]
    U_w   = float(np.sum(sign * w_pair))
    n_eff = float(np.sum(wx))
    m_eff = float(np.sum(wy))
    if n_eff == 0 or m_eff == 0:
        return U_w, np.nan, np.nan, 0
    mu_U  = 0.5 * n_eff * m_eff
    var_U = n_eff * m_eff * (n_eff + m_eff + 1.0) / 12.0
    if var_U <= 0:
        return U_w, np.nan, np.nan, 0
    z = (U_w - mu_U) / np.sqrt(var_U)
    p = 2.0 * (1.0 - normal_cdf(abs(z)))
    r = U_w / (n_eff * m_eff)        # 0..1, 0.5 = null
    direction = 1 if r > 0.5 else (-1 if r < 0.5 else 0)
    return U_w, p, r, direction

rng = np.random.default_rng(RNG_SEED)
boot_rows = []

# Pre-index weights for fast lookup
w_map = {(int(r['week_index']), r['pollutant']): r['weight']
         for _, r in weights_df.iterrows()}

# Iterate over each (pollutant, outcome, year, lag) cell
group_keys = ['pollutant', 'outcome', 'year', 'lag']
groups = diff_df.groupby(group_keys)
n_groups = len(groups)
print(f"  Cells to bootstrap: {n_groups} (pollutant x outcome x year x lag)")

for (pol_lab, out, yr, L), gsub in groups:
    exp_sub  = gsub[gsub['exposed'] == 1]
    nexp_sub = gsub[gsub['exposed'] == 0]
    n_exp    = len(exp_sub)
    n_nexp   = len(nexp_sub)
    if n_exp == 0 or n_nexp == 0:
        # Cannot bootstrap; record empty
        boot_rows.append({
            'pollutant': pol_lab, 'outcome': out, 'year': int(yr), 'lag': int(L),
            'iter': 0, 'U': np.nan, 'p_value': np.nan, 'r': np.nan,
            'direction': 0, 'n_exp': n_exp, 'n_nexp': n_nexp,
        })
        continue
    n_each = min(n_exp, n_nexp)

    exp_vals  = exp_sub['delta_inc'].values
    nexp_vals = nexp_sub['delta_inc'].values
    exp_wts   = np.array([w_map.get((int(r['week_index']), pol_lab), 0.01)
                          for _, r in exp_sub.iterrows()])
    nexp_wts  = np.array([w_map.get((int(r['week_index']), pol_lab), 0.01)
                          for _, r in nexp_sub.iterrows()])

    for it in range(N_BOOT):
        idx_e = rng.integers(0, n_exp, size=n_each)
        idx_n = rng.integers(0, n_nexp, size=n_each)
        xe = exp_vals[idx_e]
        xn = nexp_vals[idx_n]
        we = exp_wts[idx_e]
        wn = nexp_wts[idx_n]
        # Add 5% multiplicative noise
        xe_n = xe * (1.0 + rng.normal(0, NOISE_SD, size=n_each))
        xn_n = xn * (1.0 + rng.normal(0, NOISE_SD, size=n_each))
        U, p, r, d = weighted_mwu(xe_n, we, xn_n, wn)
        boot_rows.append({
            'pollutant': pol_lab, 'outcome': out, 'year': int(yr), 'lag': int(L),
            'iter': it, 'U': U, 'p_value': p, 'r': r, 'direction': d,
            'n_exp': n_exp, 'n_nexp': n_nexp,
        })

boot_df = pd.DataFrame(boot_rows)
boot_df.to_csv(os.path.join(TABLES, 'bootstrap_results.csv'), index=False)
print(f"  Saved bootstrap_results.csv  rows = {len(boot_df):,}")

# =========================================================================
# 7. Rc COMPUTATION  (Day 7-8 card) -- CRITICAL HANDOFF
# =========================================================================
print("\n[Rc] Computing weighted Vul, Rc, weighted CI, 9-class label...")

def weighted_quantile(values, weights, q):
    sorter = np.argsort(values)
    v = values[sorter]; w = weights[sorter]
    cw = np.cumsum(w) - 0.5 * w
    cw /= cw[-1] if cw[-1] != 0 else 1.0
    return float(np.interp(q, cw, v))

rc_rows = []
for (pol_lab, out, yr, L), gsub in boot_df.groupby(['pollutant','outcome','year','lag']):
    g = gsub.dropna(subset=['r', 'p_value'])
    if len(g) == 0:
        continue
    w = (1.0 - g['p_value'].values).clip(min=0)
    r = g['r'].values
    if w.sum() == 0:
        vul = float(np.mean(r))
        ci_lo, ci_hi = (np.nanpercentile(r, 2.5), np.nanpercentile(r, 97.5))
    else:
        vul = float(np.average(r, weights=w))
        ci_lo = weighted_quantile(r, w, 0.025)
        ci_hi = weighted_quantile(r, w, 0.975)
    sd_r = float(np.std(r, ddof=1))
    rc_rows.append({
        'pollutant': pol_lab, 'outcome': out, 'year': int(yr), 'lag': int(L),
        'n_iter': int(len(g)),
        'mean_r': round(float(np.mean(r)), 4),
        'sd_r': round(sd_r, 4),
        'mean_p': round(float(np.mean(g['p_value'])), 4),
        'Vul': round(vul, 4),
        'Rc': round(vul, 4),
        'Rc_CI_low':  round(ci_lo, 4),
        'Rc_CI_high': round(ci_hi, 4),
        'Rc_class': classify_rc(vul),
    })

rc_df = pd.DataFrame(rc_rows)
rc_df.to_csv(os.path.join(TABLES, 'weekly_citylevel_Rc_by_pollutant_outcome_year.csv'),
             index=False)
# Slim handoff table for P4
rc_df[['pollutant','outcome','year','lag','Rc','Rc_CI_low','Rc_CI_high','Rc_class']].to_csv(
    os.path.join(TABLES, 'Rc_weighted_CI_table.csv'), index=False)
print(f"  Saved Rc tables (rows = {len(rc_df)})")
print(rc_df.head(12).to_string(index=False))

# =========================================================================
# 8. WMARM-LIKE SCORE  (Day 8-9 card)
# =========================================================================
print("\n[WMARM-like] Computing standardized effect & inverse-CI weight...")
wmarm_rows = []
for (pol_lab, out, L), gsub in rc_df.groupby(['pollutant','outcome','lag']):
    # Per-year se = Vul / sd_r (signed); summary across years
    se_signed = (gsub['Vul'] - 0.5) / gsub['sd_r'].replace(0, np.nan)
    abs_se    = se_signed.abs()
    weight_CI = 1.0 / (gsub['Rc_CI_high'] - gsub['Rc_CI_low']).replace(0, np.nan)
    # City-level (across years) summary -- weighted by inverse-CI width
    if weight_CI.sum() > 0:
        city_se = float(np.average(se_signed.dropna(),
                                    weights=weight_CI[se_signed.notna()]))
    else:
        city_se = float(se_signed.mean())
    wmarm_rows.append({
        'pollutant': pol_lab, 'outcome': out, 'lag': int(L),
        'n_years': int(len(gsub)),
        'mean_Vul': round(float(gsub['Vul'].mean()), 4),
        'mean_sd_r': round(float(gsub['sd_r'].mean()), 4),
        'mean_se_signed': round(float(se_signed.mean(skipna=True)), 4),
        'mean_abs_se': round(float(abs_se.mean(skipna=True)), 4),
        'mean_weightCI': round(float(weight_CI.mean(skipna=True)), 4),
        'citylevel_se_weighted': round(city_se, 4),
    })

wmarm_df = pd.DataFrame(wmarm_rows)
wmarm_df.to_csv(os.path.join(TABLES, 'citylevel_WMARM_like_summary.csv'), index=False)
print(f"  Saved citylevel_WMARM_like_summary.csv  (rows = {len(wmarm_df)})")
print(wmarm_df.to_string(index=False))

CAVEAT = (
"WMARM-like caveat\n"
"=================\n"
"The original APHREH-ADSMap WMARM is a population-weighted spatial aggregation\n"
"across multiple Basic Spatial Areas (BSAs). This dataset contains a single\n"
"city-level BSA (Milan as one unit), so population-weighted spatial aggregation\n"
"is mathematically trivial. The score reported here is therefore a WMARM-LIKE\n"
"relevance score, computed as the inverse-CI-weighted standardized effect\n"
"(se = Vul / sd_r) summarized across years for each (pollutant, outcome, lag),\n"
"and is not a full spatial WMARM.\n"
)
with open(os.path.join(TABLES, 'WMARM_like_caveat.txt'), 'w') as f:
    f.write(CAVEAT)

# =========================================================================
# 9. SENSITIVITY SURFACES  (Day 9-10 card)
# =========================================================================
print("\n[Sensitivity] pc x lag grid for primary outcome (respiratory_disease)...")
# Refactor: write a function that runs the entire pipeline (steps 2-8) for a
# given pc and lag, and returns the city-level WMARM-like score for the
# primary outcome and a chosen pollutant.
# To keep this fast, we evaluate per (pollutant, pc, lag) using a SINGLE-LAG
# pipeline (no need to recompute all lags for each cell).

def run_for_cell(pol_col, pc, lag, n_boot=300):
    pol_lab = POL_LABELS[pol_col]
    rng2 = np.random.default_rng(RNG_SEED + int(pc*100) + lag*7 + hash(pol_lab) % 1000)

    # 2'. Yearly thresholds + classify exposure
    local_exp = {}      # (week_index) -> exposed
    local_thr = {}      # year -> threshold
    for yr, sub in df.groupby('year'):
        thr = float(np.quantile(sub[pol_col].dropna(), pc))
        local_thr[int(yr)] = thr
        for _, r in sub.iterrows():
            local_exp[int(r['week_index'])] = int(r[pol_col] >= thr) \
                if pd.notna(r[pol_col]) else np.nan

    # 3'. Lagged primary outcome incidence
    inc_arr = df[PRIMARY_OUT].values / POPULATION
    if lag == 0:
        lag_arr = inc_arr.copy()
    else:
        lag_arr = np.concatenate([inc_arr[lag:], [np.nan]*lag])
    wk_arr  = df['week_index'].values.astype(int)
    yr_arr  = df['year'].values.astype(int)

    # 4'. Differential incidence with ±3/4/5 baseline expansion
    diffs = []
    for i in range(len(df)):
        t = int(wk_arr[i])
        if pd.isna(lag_arr[i]):
            continue
        ex_t = local_exp.get(t, np.nan)
        if pd.isna(ex_t):
            continue
        used_window = None; vals = []
        for W in BASELINE_WINDOWS:
            cand_idx = [j for j in range(len(df))
                        if int(wk_arr[j]) != t
                        and abs(int(wk_arr[j]) - t) <= W]
            vv = []
            for j in cand_idx:
                tj = int(wk_arr[j])
                if local_exp.get(tj) == 0 and not pd.isna(lag_arr[j]):
                    vv.append(float(lag_arr[j]))
            if len(vv) >= 2:
                used_window = W; vals = vv; break
        if used_window is None:
            continue
        baseline = float(np.median(vals))
        diffs.append({'wk': t, 'year': int(yr_arr[i]),
                      'exposed': int(ex_t),
                      'delta_inc': float(lag_arr[i] - baseline)})

    diffs_df = pd.DataFrame(diffs)
    if len(diffs_df) == 0:
        return np.nan

    # 5'. Local weights
    local_w = {}
    for yr, sub in df.groupby('year'):
        thr = local_thr[int(yr)]
        max_log = 1.0
        log_d_exp = []
        for _, r in sub.iterrows():
            v = r[pol_col]
            if pd.notna(v) and v >= thr:
                log_d_exp.append(np.log1p(v - thr))
        if log_d_exp:
            max_log = max(log_d_exp)
        for _, r in sub.iterrows():
            v = r[pol_col]
            if pd.isna(v):
                w = 0.01
            elif v >= thr:
                w = max(np.log1p(v - thr) / max_log if max_log > 0 else 1.0, 0.01)
            else:
                w = 0.01
            local_w[int(r['week_index'])] = w

    # 6'. Bootstrap MWU per year, then aggregate to city-level
    rs_all = []
    ps_all = []
    for yr, gg in diffs_df.groupby('year'):
        e  = gg[gg['exposed'] == 1]
        ne = gg[gg['exposed'] == 0]
        if len(e) == 0 or len(ne) == 0:
            continue
        n_each = min(len(e), len(ne))
        ev = e['delta_inc'].values
        nv = ne['delta_inc'].values
        ew = np.array([local_w[int(w)] for w in e['wk'].values])
        nw = np.array([local_w[int(w)] for w in ne['wk'].values])
        rs = []; ps = []
        for it in range(n_boot):
            ie = rng2.integers(0, len(e), size=n_each)
            ino = rng2.integers(0, len(ne), size=n_each)
            xe = ev[ie] * (1 + rng2.normal(0, NOISE_SD, size=n_each))
            xn = nv[ino] * (1 + rng2.normal(0, NOISE_SD, size=n_each))
            _, p, r, _ = weighted_mwu(xe, ew[ie], xn, nw[ino])
            if not np.isnan(r):
                rs.append(r); ps.append(p)
        if rs:
            w = (1 - np.array(ps)).clip(min=0)
            vul = float(np.average(rs, weights=w)) if w.sum() > 0 else float(np.mean(rs))
            sd_r = float(np.std(rs, ddof=1))
            rs_all.append((vul, sd_r))

    if not rs_all:
        return np.nan
    vuls = np.array([v for v, _ in rs_all])
    sds  = np.array([s for _, s in rs_all])
    # Floor sd_r at a small positive value to avoid divide-by-near-zero
    # explosions when bootstrap r happens to land on near-identical values.
    sds_safe = np.maximum(sds, 1e-3)
    se_signed = (vuls - 0.5) / sds_safe
    # Clip extreme values that suggest numerical instability
    se_signed = np.clip(se_signed, -10, 10)
    return float(np.nanmean(se_signed))

print("  Running pc x lag grid (NO2, PM2.5, PM10) ...")
sens_rows = []
for pol_col in POLLUTANTS:
    pol_lab = POL_LABELS[pol_col]
    for pc in PC_GRID:
        for L in LAGS:
            score = run_for_cell(pol_col, pc, L, n_boot=300)
            sens_rows.append({
                'pollutant': pol_lab, 'pc': pc, 'lag': L,
                'wmarm_like_se': round(score, 4) if not np.isnan(score) else np.nan,
            })
            print(f"    {pol_lab:6s}  pc={pc:.2f}  lag={L}  ->  se = {score:.4f}")

sens_df = pd.DataFrame(sens_rows)
sens_df.to_csv(os.path.join(SENSITIVITY, 'sensitivity_grid.csv'), index=False)

# Clean 3D surface + clean 2D heatmap per pollutant
for pol_lab in ['NO2', 'PM2.5', 'PM10']:
    sub = sens_df[sens_df['pollutant'] == pol_lab]

    pivot = sub.pivot(index='lag', columns='pc', values='wmarm_like_se')
    pivot = pivot.sort_index().sort_index(axis=1)

    pcs = pivot.columns.values.astype(float)
    lags_v = pivot.index.values.astype(float)
    Z = pivot.values.astype(float)

    fig = plt.figure(figsize=(13, 5.5))

    # -------------------------
    # 3D surface
    # -------------------------
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')

    X, Y = np.meshgrid(pcs, lags_v)

    surf = ax1.plot_surface(
        X, Y, Z,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.3,
        alpha=0.9
    )

    ax1.set_xlabel('pc threshold')
    ax1.set_ylabel('lag (weeks)')
    ax1.set_zlabel('WMARM-like se')
    ax1.set_title(f'WMARM-like surface - {pol_lab} -> respiratory')

    ax1.set_xticks(pcs)
    ax1.set_yticks(lags_v)
    ax1.view_init(elev=25, azim=-135)

    cbar1 = fig.colorbar(surf, ax=ax1, shrink=0.65, pad=0.08)
    cbar1.set_label('WMARM-like se')

    # -------------------------
    # 2D heatmap
    # -------------------------
    ax2 = fig.add_subplot(1, 2, 2)

    # Use integer positions so text is exactly centered
    im = ax2.imshow(Z, aspect='auto', origin='lower', cmap='viridis')

    ax2.set_xticks(np.arange(len(pcs)))
    ax2.set_xticklabels([f"{pc:.2f}" for pc in pcs])

    ax2.set_yticks(np.arange(len(lags_v)))
    ax2.set_yticklabels([str(int(L)) for L in lags_v])

    ax2.set_xlabel('pc threshold')
    ax2.set_ylabel('lag (weeks)')
    ax2.set_title(f'Heatmap - {pol_lab} -> respiratory')

    cbar2 = fig.colorbar(im, ax=ax2)
    cbar2.set_label('WMARM-like se')

    # Annotate heatmap cells exactly at the center
    z_min = np.nanmin(Z)
    z_max = np.nanmax(Z)
    mid = (z_min + z_max) / 2

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            value = Z[i, j]
            text = "NA" if np.isnan(value) else f"{value:.2f}"

            ax2.text(
                j, i, text,
                ha='center',
                va='center',
                fontsize=10,
                color='white' if value < mid else 'black'
            )

    fig.tight_layout()

    filename = f'WMARM_like_surface_{pol_lab.replace(".", "")}.png'
    fig.savefig(os.path.join(FIGURES, filename), dpi=160, bbox_inches='tight')
    plt.close(fig)

# Top pollutant compared to DLNM
top = sens_df.groupby('pollutant')['wmarm_like_se'].max().sort_values(ascending=False)
print("\n  Top WMARM-like se by pollutant (across the pc x lag grid):")
print(top.to_string())
print("  (P2 DLNM headline: NO2 -> respiratory significant; PM2.5 / PM10 null)")

with open(os.path.join(SENSITIVITY, 'sensitivity_summary.txt'), 'w') as f:
    f.write("Sensitivity grid (pc x lag) summary — primary outcome respiratory_disease\n")
    f.write("=========================================================================\n\n")
    f.write("Top WMARM-like se by pollutant (max across grid):\n")
    f.write(top.to_string() + "\n\n")
    f.write("Comparison with P2 DLNM:\n")
    f.write("  P2 found NO2 -> respiratory the only significant signal "
            "(cum RR 1.778; 1.052, 3.006).\n")
    f.write("  Higher WMARM-like se for NO2 here would be concordant with that result.\n")

print("\n[Done] All P3 deliverables written under p3_deliverables/")
