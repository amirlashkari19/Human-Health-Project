# Human Health Project — Milan Air Pollution & Respiratory Morbidity

> **Headline result:** NO₂ is significantly associated with respiratory disease in Milan (cumulative RR = **1.78**, 95% CI: 1.05–3.01, attributable fraction ≈ **35%**), based on a weekly ecological time-series DLNM over 155 weeks (W18/2023 – W16/2026).

---

## Overview

This repository integrates the full three-person analysis pipeline studying ambient air pollution (NO₂, PM₂.₅, PM₁₀) and respiratory morbidity (respiratory disease, ILI, pneumonia) in Milan.

| Person | Role | Folder |
|---|---|---|
| P1 | Data preparation & cleaning | `p1_data_prep/` |
| P2 | DLNM modelling, sensitivity, attributable risk | `p2_dlnm/` |
| P3 | Rc / WMARM-like city-level relevance analysis | `p3_rc_wmarm/` |

---

## Repository Structure

```
human-health/
├── data/
│   ├── raw/                        # dataset_settimanale_finale 2.csv (32 KB)
│   └── processed/                  # analysis_ready.csv (output of P1)
├── p1_data_prep/                   # P1_script.py + reports
├── p2_dlnm/
│   ├── scripts/                    # 01–04 R scripts
│   ├── rr_surfaces/                # 60 RR-surface PNGs
│   ├── model_reduction/            # cumulative/lag/predictor RR tables
│   ├── attributable_risk/          # AF/AN tables + caveats
│   ├── diagnostics/                # Cook's distance, DFBETA, residuals
│   ├── sensitivity/                # 144-spec sensitivity grid
│   └── report/                     # dlnm_section_P2.md
├── p3_rc_wmarm/
│   ├── scripts/                    # 01_run_p3_pipeline.py
│   ├── tables/                     # 10 Rc/WMARM tables
│   ├── figures/                    # 4 WMARM-like surface PNGs
│   ├── sensitivity/                # Rc sensitivity grid
│   └── report/                     # rc_wmarm_section_P3.md
├── report/
│   └── combined_report.md          # Integrated findings across P1/P2/P3
├── tests/                          # pytest reproducibility suite (3 tests)
├── requirements.txt                # Python dependencies
├── environment.R                   # R package installer
├── Makefile                        # make p1 / p2 / p3 / all / test
└── .github/workflows/ci.yml        # CI: reproducibility check on every push
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/amirlashkari19/Human-Health-Project.git
cd Human-Health-Project

# 2. Python dependencies
pip install -r requirements.txt

# 3. R dependencies
Rscript environment.R

# 4. Run full pipeline
make all       # P1 (preprocessing) + P3 (Rc/WMARM)
make p2        # P2 DLNM models (requires R)

# 5. Run tests
make test
```

---

## Key Results

### P2 — Primary DLNM (NO₂ → Respiratory Disease)

| Metric | Value |
|---|---|
| Cumulative RR at p75 (42 µg/m³) vs. 10 µg/m³ | **1.778** (95% CI: 1.052–3.006) |
| Attributable Fraction | **~35%** |
| Lag window | 0–4 weeks |
| Model | Quasi-Poisson DLNM, natural spline cross-basis |

PM₂.₅ and PM₁₀ single-pollutant models: CIs cross 1 (non-significant).
Two-pollutant model (NO₂ + PM₁₀): NO₂ cum RR = **2.80** (95% CI: 1.80–4.35).

### P3 — Rc / WMARM-like

NO₂ has the highest city-level relevance score (Rc), with 14/36 cells having CI_low > 0.5, compared to PM₁₀ (6/36) and PM₂.₅ (8/36) — consistent with the DLNM findings.

---

## Reproducibility & CI

- Every push triggers GitHub Actions CI (`.github/workflows/ci.yml`)
- P1 and P3 outputs are diffed against committed versions; build fails if any numeric value drifts > 1%
- Three pytest tests cover: P1 row count, P2 NO₂ RR (±5%), P3 Rc table shape and NO₂ dominance

---

## Data

Raw data (`data/raw/`) is committed — it is non-sensitive aggregate ecological data (weekly city-level counts, no individual records). See [`data/README.md`](data/README.md) for variable descriptions.

---

## Citation / Acknowledgements

HHLab — Milan Air Pollution & Respiratory Morbidity Study, 2023–2026.
Methods follow Gasparrini et al. (2010) DLNM framework and Gasparrini & Leone (2014) attributable risk approach.
