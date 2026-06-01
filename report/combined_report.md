# Milan Air Pollution & Respiratory Morbidity — Combined Report
**Project:** HHLab | **Period:** W18/2023 – W16/2026 (155 weeks) | **Date:** May 2026

---

## Headline Finding

> **NO₂ is significantly associated with respiratory disease admissions in Milan (cumulative RR = 1.78, 95% CI: 1.05–3.01 at p75 vs. 10 µg/m³), with an estimated attributable fraction of ~35%. The association is robust across 144 sensitivity specifications.**

---

## Part 1 — Data Preparation (P1)

See full report: [`p1_data_prep/P1_Complete_Report.pdf`](../p1_data_prep/P1_Complete_Report.pdf)

### Sample
- **Raw dataset:** `data/raw/dataset_settimanale_finale 2.csv` — 155 ISO-weeks, W18/2023 → W16/2026
- **Cleaned dataset:** `data/processed/analysis_ready.csv`
- **Analytic sample:** n = 154 weeks (W8/2025 outlier removed: −64% visits vs. winter mean, likely ED under-reporting)
- W52/2025 flagged as partial week (5/7 measurement days); excluded in sensitivity

### Pollutants & Outcomes
| Exposure | Weekly mean (µg/m³) | p75 |
|---|---|---|
| NO₂ | — | ~42 |
| PM₂.₅ | — | ~24 |
| PM₁₀ | — | — |

Outcomes: **respiratory disease** (primary), ILI, pneumonia (weekly counts).

**Key constraint:** Single-pollutant models only — PM₂.₅ ~ PM₁₀ correlation ρ = 0.956.

---

## Part 2 — DLNM Analysis (P2)

See full section: [`p2_dlnm/report/dlnm_section_P2.md`](../p2_dlnm/report/dlnm_section_P2.md)

### Primary Model: NO₂ → Respiratory Disease

| Metric | Value |
|---|---|
| Cumulative RR (p75 vs. 10 µg/m³, lag 0–4 wks) | **1.778** (95% CI: 1.052–3.006) |
| Attributable Fraction | **~35%** |
| Attributable Number | See `p2_dlnm/attributable_risk/attributable_risk_table.csv` |
| Overdispersion (Pearson) | > 1.2 → quasi-Poisson justified |

### 9 Single-Pollutant Screening Models (3 pollutants × 3 outcomes)

See `p2_dlnm/model_reduction/cumulative_RR_table.csv` for full results.

| Model | Pollutant | Outcome | cum RR | 95% CI |
|---|---|---|---|---|
| M1 | NO₂ | Respiratory | **1.778** | 1.052–3.006 |
| M2 | PM₂.₅ | Respiratory | 0.928 | 0.677–1.272 |
| M3 | PM₁₀ | Respiratory | — | — |
| M4–M9 | All | ILI, Pneumonia | See table | — |

### Two-Pollutant Addendum

NO₂ + PM₁₀ model (respiratory disease): NO₂ cum RR = **2.80** (95% CI: 1.80–4.35).
See `p2_dlnm/two_pollutant_RR_table.csv`.

### Sensitivity Analysis (144 specifications)

- Varied: max lag (2/4/6 wks), var df (2/3), lag df (2/3), time df (6/8/10), meteorology (ns/linear), W52/2025 (in/out)
- Primary point estimate robust: CI excludes 1 in the majority of specifications
- Fourier vs. ns(week_index) comparison: consistent
- Winter interaction: stronger NO₂ effect in winter (exploratory)

See `p2_dlnm/sensitivity/dlnm_sensitivity_table.csv` and `dlnm_sensitivity_summary.png`.

---

## Part 3 — Rc / WMARM-like Analysis (P3)

See full section: [`p3_rc_wmarm/report/rc_wmarm_section_P3.md`](../p3_rc_wmarm/report/rc_wmarm_section_P3.md)

### City-Level Relevance (Rc)

- **108-row Rc table** covering 3 pollutants × 3 outcomes × 3 years × 4 lags
- **NO₂ dominates:** 14/36 cells have Rc_CI_low > 0.5 vs. PM₁₀ (6/36) and PM₂.₅ (8/36)

See `p3_rc_wmarm/tables/Rc_weighted_CI_table.csv`.

### WMARM-like Summary

Single-BSA city → WMARM = Rc (population weight = 1.0).
See `p3_rc_wmarm/tables/citylevel_WMARM_like_summary.csv`.

Key figures: `p3_rc_wmarm/figures/WMARM_like_surface_NO2.png` and peers.

---

## Methodological Cross-Check

| Finding | P2 (DLNM) | P3 (Rc/WMARM) | Consistent? |
|---|---|---|---|
| NO₂ strongest pollutant | ✅ Only model with CI excl. 1 | ✅ Most CI_low > 0.5 cells | ✅ Yes |
| PM₂.₅ / PM₁₀ weaker | ✅ RR CIs cross 1 | ✅ Fewer CI_low > 0.5 | ✅ Yes |
| Effect concentrated in winter | ✅ Winter interaction p < 0.05 | ✅ Higher Rc in winter lags | ✅ Yes |

---

## Reproducibility

```bash
# Full pipeline
make all       # P1 + P3 (Python)
make p2        # P2 DLNM (R, requires Rscript + packages)
make test      # pytest — 3 reproducibility checks
```

CI runs automatically on every push via `.github/workflows/ci.yml`.
