# Makefile — Human Health Project (Milan Air Pollution & Respiratory Morbidity)
# Usage:
#   make p1       # Run data preprocessing
#   make p3       # Run P3 Rc/WMARM pipeline
#   make all      # Run p1 then p3 (p2 requires R; see make p2)
#   make p2       # Run P2 DLNM models (requires R + packages)
#   make test     # Run pytest test suite
#   make clean    # Remove generated outputs

PYTHON   := python3
RSCRIPT  := Rscript
DATA_RAW := "data/raw/dataset_settimanale_finale 2.csv"
DATA_OUT := data/processed/analysis_ready.csv

.PHONY: all p1 p2 p3 test clean

all: p1 p3

## ── P1: data preprocessing ──────────────────────────────────────────────────
p1: $(DATA_OUT)

$(DATA_OUT): p1_data_prep/P1_script.py $(DATA_RAW)
	$(PYTHON) p1_data_prep/P1_script.py \
	  --input $(DATA_RAW) \
	  --output $(DATA_OUT)

## ── P2: DLNM models (R) ─────────────────────────────────────────────────────
p2: $(DATA_OUT)
	$(RSCRIPT) p2_dlnm/scripts/01_fit_dlnm_models.R
	$(RSCRIPT) p2_dlnm/scripts/02_sensitivity_analysis.R
	$(RSCRIPT) p2_dlnm/scripts/03_attributable_risk.R
	$(RSCRIPT) p2_dlnm/scripts/04_diagnostics_extended.R

## ── P3: Rc / WMARM pipeline ─────────────────────────────────────────────────
p3: $(DATA_OUT)
	$(PYTHON) p3_rc_wmarm/scripts/01_run_p3_pipeline.py

## ── Tests ────────────────────────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -v

## ── Clean generated outputs ─────────────────────────────────────────────────
clean:
	rm -f $(DATA_OUT)
	find p2_dlnm/rr_surfaces -name "*.png" -not -name ".gitkeep" -delete
	find p3_rc_wmarm/figures  -name "*.png" -not -name ".gitkeep" -delete
