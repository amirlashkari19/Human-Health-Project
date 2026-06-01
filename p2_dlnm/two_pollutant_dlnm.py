"""
Extension to P2: 3 two-pollutant DLNMs (NO2 + PM10 in the same model)
crossed against respiratory disease, ILI, pneumonia.

Implements the dlnm cross-basis methodology from scratch (numpy only):
  - Natural cubic spline basis (matches the space spanned by R's splines::ns)
  - DLNM cross-basis as tensor product of variable basis * lag basis
  - Quasi-Poisson IRLS with Pearson dispersion-adjusted standard errors
  - crosspred-equivalent: cumulative RR (over lags 0-4) at p75 vs reference,
    with delta-method 95% CIs

Same spec as the P2 primary model:
  lag = 4, var df = 3, lag df = 3, ns(week_index, df=8),
  ns(temp_mean, df=3), ns(humidity_mean, df=3), quasi-Poisson family.

References for primary model (single-pollutant, from p2_deliverables 2):
  M1 NO2  -> Respiratory : RR = 1.778 (1.052, 3.006)
  M3 PM10 -> Respiratory : RR = 0.931 (0.777, 1.116)
  M4 NO2  -> ILI         : RR = 0.963 (0.427, 2.170)
  M6 PM10 -> ILI         : RR = 0.894 (0.679, 1.177)
  M7 NO2  -> Pneumonia   : RR = 0.768 (0.378, 1.560)
  M9 PM10 -> Pneumonia   : RR = 0.927 (0.729, 1.179)
"""

import numpy as np
import pandas as pd

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
INPUT  = os.path.join(_HERE, 'analysis_ready.csv')
OUTPUT = os.path.join(_HERE, 'two_pollutant_RR_table.csv')

# ---------------------------------------------------------------------------
# Natural cubic spline basis (ESL eq. 5.4-5.5, matches space of R splines::ns)
# ---------------------------------------------------------------------------
def ns_basis(x, df=3, boundary=None, internal=None):
    x = np.asarray(x, dtype=float)
    if boundary is None:
        boundary = (np.nanmin(x), np.nanmax(x))
    if internal is None:
        if df > 1:
            qs = np.linspace(0, 1, df + 1)[1:-1]
            internal = np.quantile(x[~np.isnan(x)], qs)
        else:
            internal = np.array([])
    knots = np.sort(np.concatenate([[boundary[0]], internal, [boundary[1]]]))
    K = len(knots)  # = df + 1 total knots

    def d(xv, k):
        num = np.maximum(xv - knots[k], 0)**3 - np.maximum(xv - knots[K-1], 0)**3
        den = knots[K-1] - knots[k]
        return num / den

    cols = [x.copy()]                                  # linear term
    for k in range(K - 2):                             # K-2 cubic terms
        cols.append(d(x, k) - d(x, K - 2))
    return np.column_stack(cols), {'boundary': boundary, 'internal': internal}

def ns_apply(x, info):
    return ns_basis(np.asarray(x, dtype=float), df=None,
                    boundary=info['boundary'], internal=info['internal'])[0] \
        if False else _ns_apply(x, info)

def _ns_apply(x, info):
    # apply ns with the same knots as another fit (so prediction uses same basis)
    boundary, internal = info['boundary'], info['internal']
    df = 1 + len(internal)
    knots = np.sort(np.concatenate([[boundary[0]], internal, [boundary[1]]]))
    K = len(knots)
    x = np.asarray(x, dtype=float)
    def d(xv, k):
        num = np.maximum(xv - knots[k], 0)**3 - np.maximum(xv - knots[K-1], 0)**3
        den = knots[K-1] - knots[k]
        return num / den
    cols = [x.copy()]
    for k in range(K - 2):
        cols.append(d(x, k) - d(x, K - 2))
    return np.column_stack(cols)

# ---------------------------------------------------------------------------
# DLNM cross-basis  (Gasparrini 2014)
#   For time t: W(t)_{j,k} = sum_{l=0..L} B_v(x_{t-l})_j * B_l(l)_k
#   Returns n x (vardf*lagdf) matrix; first L rows are NaN (lag padding).
# ---------------------------------------------------------------------------
def crossbasis(x, lag=4, var_df=3, lag_df=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    L = lag
    lag_grid = np.arange(L + 1)
    # Variable basis on full series (knots from observed values)
    Bv_full, vinfo = ns_basis(x, df=var_df)
    # Lag basis on lags 0..L (knots placed across the lag grid)
    Bl, linfo = ns_basis(lag_grid.astype(float), df=lag_df)
    vardf = Bv_full.shape[1]
    lagdf = Bl.shape[1]
    W = np.full((n, vardf * lagdf), np.nan)
    # For each l, take Bv at x_{t-l} (shifted) and accumulate
    for l in range(L + 1):
        Bv_lag = np.full((n, vardf), np.nan)
        if l == 0:
            Bv_lag = Bv_full.copy()
        else:
            Bv_lag[l:, :] = Bv_full[:-l, :]
        for j in range(vardf):
            for k in range(lagdf):
                W[:, j * lagdf + k] = np.where(
                    np.isnan(W[:, j * lagdf + k]), 0.0, W[:, j * lagdf + k]
                ) + Bv_lag[:, j] * Bl[l, k]
    # Set the first L rows to NaN (insufficient lag history)
    W[:L, :] = np.nan
    return W, {'vinfo': vinfo, 'linfo': linfo, 'L': L,
               'vardf': vardf, 'lagdf': lagdf, 'Bl': Bl}

# ---------------------------------------------------------------------------
# Quasi-Poisson GLM via IRLS
# ---------------------------------------------------------------------------
def quasipoisson_glm(X, y, max_iter=100, tol=1e-9):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    # Initial mu: y + 0.1 (avoid zeros for log)
    mu = np.maximum(y, 0.1) + 0.1
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    for it in range(max_iter):
        z = eta + (y - mu) / mu                # working response
        w = mu                                  # IRLS weights for Poisson
        WX = X * w[:, None]
        XtWX = X.T @ WX
        XtWz = X.T @ (w * z)
        try:
            beta_new = np.linalg.solve(XtWX, XtWz)
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(XtWX, XtWz, rcond=None)[0]
        eta = X @ beta_new
        eta = np.clip(eta, -30, 30)
        mu_new = np.exp(eta)
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            mu = mu_new
            break
        beta = beta_new
        mu = mu_new
    # Pearson residuals & dispersion
    n, p = X.shape
    pearson = (y - mu) / np.sqrt(mu)
    df_resid = n - p
    dispersion = np.sum(pearson**2) / df_resid
    # Coefficient covariance (quasi-Poisson: scale * (X' W X)^-1)
    XtWX = X.T @ (X * mu[:, None])
    Vbeta = dispersion * np.linalg.inv(XtWX)
    return {'beta': beta, 'mu': mu, 'eta': eta,
            'dispersion': dispersion, 'Vbeta': Vbeta,
            'df_resid': df_resid, 'n': n, 'p': p,
            'log_lik_proxy': float(np.sum(y * eta - mu)),
            'iters': it}

# ---------------------------------------------------------------------------
# Cumulative RR over lags 0..L for a single cross-basis at exposure x vs ref
# Builds the contrast vector c, then RR = exp(c' beta_cb), Var = c' V c.
# ---------------------------------------------------------------------------
def cumulative_rr(model, cb_info, beta_idx, x, ref):
    Bl = cb_info['Bl']
    vinfo = cb_info['vinfo']
    vardf = cb_info['vardf']
    lagdf = cb_info['lagdf']
    L = cb_info['L']
    Bv_x   = _ns_apply(np.array([x]),   vinfo)[0]   # vardf
    Bv_ref = _ns_apply(np.array([ref]), vinfo)[0]   # vardf
    diff_v = Bv_x - Bv_ref                          # vardf
    # Sum lag basis over l = 0..L (cumulative)
    sum_Bl = Bl.sum(axis=0)                         # lagdf
    # Tensor product diff_v outer sum_Bl, then flatten same order as crossbasis (j*lagdf+k)
    contrast = np.outer(diff_v, sum_Bl).reshape(-1) # vardf*lagdf
    # Slice the relevant block of beta and Vbeta
    b = model['beta'][beta_idx]
    V = model['Vbeta'][np.ix_(beta_idx, beta_idx)]
    log_rr = float(contrast @ b)
    se     = float(np.sqrt(contrast @ V @ contrast))
    rr     = np.exp(log_rr)
    return rr, np.exp(log_rr - 1.96*se), np.exp(log_rr + 1.96*se), log_rr, se

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(INPUT)
    df = df[df['outlier_flag'] == 0].sort_values('week_index').reset_index(drop=True)
    print(f"n (after outlier exclusion) = {len(df)}")
    print(f"NO2-PM10 correlation = {df[['NO2_mean','PM10_mean']].corr().iloc[0,1]:.3f}")

    outcomes = ['respiratory_disease', 'ILI', 'pneumonia']
    pol_a, pol_b = 'NO2_mean', 'PM10_mean'
    ref_a, ref_b = 10.0, 15.0

    # Build pollutant cross-bases (shared across outcomes since pollutants don't change)
    cbA, cbA_info = crossbasis(df[pol_a].values, lag=4, var_df=3, lag_df=3)
    cbB, cbB_info = crossbasis(df[pol_b].values, lag=4, var_df=3, lag_df=3)

    # Confounder splines
    Z_time, _ = ns_basis(df['week_index'].values, df=8)
    Z_temp, _ = ns_basis(df['temp_mean'].values,  df=3)
    Z_hum,  _ = ns_basis(df['humidity_mean'].values, df=3)

    # Reference quantiles for pollutants (75th percentile, on cleaned data)
    p75_NO2  = float(np.quantile(df[pol_a].values, 0.75))
    p75_PM10 = float(np.quantile(df[pol_b].values, 0.75))
    print(f"p75 NO2  = {p75_NO2:.2f} ug/m3 (vs ref {ref_a})")
    print(f"p75 PM10 = {p75_PM10:.2f} ug/m3 (vs ref {ref_b})")

    rows = []
    for outcome in outcomes:
        y = df[outcome].values.astype(float)
        # Drop the first 4 rows (NaN cross-basis padding)
        keep = ~np.isnan(cbA[:, 0])
        Xparts = [
            np.ones((keep.sum(), 1)),                 # intercept
            cbA[keep, :],
            cbB[keep, :],
            Z_time[keep, :],
            Z_temp[keep, :],
            Z_hum[keep, :],
        ]
        X = np.hstack(Xparts)
        ycut = y[keep]

        # Track column indices of each block
        ncols = [p.shape[1] for p in Xparts]
        offsets = np.cumsum([0] + ncols)
        idx_cbA = list(range(offsets[1], offsets[2]))
        idx_cbB = list(range(offsets[2], offsets[3]))

        m = quasipoisson_glm(X, ycut)
        rr_NO2, lo_NO2, hi_NO2, _, _ = cumulative_rr(
            m, cbA_info, idx_cbA, p75_NO2, ref_a)
        rr_PM10, lo_PM10, hi_PM10, _, _ = cumulative_rr(
            m, cbB_info, idx_cbB, p75_PM10, ref_b)

        print(f"\n--- Outcome: {outcome} ---")
        print(f"  n          = {m['n']}")
        print(f"  dispersion = {m['dispersion']:.2f}  (df residual = {m['df_resid']})")
        print(f"  IRLS iters = {m['iters']}")
        print(f"  Cum RR (NO2  p75={p75_NO2:.1f} vs {ref_a}, lags 0-4): "
              f"{rr_NO2:.3f}  (95% CI {lo_NO2:.3f}, {hi_NO2:.3f})")
        print(f"  Cum RR (PM10 p75={p75_PM10:.1f} vs {ref_b}, lags 0-4): "
              f"{rr_PM10:.3f}  (95% CI {lo_PM10:.3f}, {hi_PM10:.3f})")

        rows.append({
            'model_id': f"M2P_{outcome}",
            'outcome': outcome,
            'n': m['n'], 'dispersion': round(m['dispersion'], 2),
            'cum_RR_NO2': round(rr_NO2, 3),
            'cum_RR_NO2_low': round(lo_NO2, 3),
            'cum_RR_NO2_high': round(hi_NO2, 3),
            'cum_RR_PM10': round(rr_PM10, 3),
            'cum_RR_PM10_low': round(lo_PM10, 3),
            'cum_RR_PM10_high': round(hi_PM10, 3),
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT, index=False)
    print(f"\nSaved -> {OUTPUT}")
    print(out.to_string(index=False))

if __name__ == '__main__':
    main()
