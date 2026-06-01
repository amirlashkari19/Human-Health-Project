## ============================================================
##  P2 — Attributable risk (Day 6–7 Trello card)
##  Computes attributable fraction (AF) and attributable number (AN)
##  for each of the 9 single-pollutant DLNMs vs WHO-aligned references.
##  Method: Gasparrini & Leone (2014) closed-form for DLNM with bootstrap CIs.
## ============================================================

suppressPackageStartupMessages({
  library(dlnm); library(splines); library(MASS); library(dplyr); library(readr)
})

set.seed(20260515)

ROOT <- "/home/user/workspace/p2_deliverables"
DATA <- file.path(ROOT, "data/processed/analysis_ready.csv")
OUT  <- file.path(ROOT, "dlnm/attributable_risk")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

df <- read_csv(DATA, show_col_types = FALSE)
df <- df %>% filter(outlier_flag == 0)
stopifnot(nrow(df) == 154)

POLL <- list(NO2   = list(col = "NO2_mean",  ref = 10),
             PM2.5 = list(col = "PM25_mean", ref =  5),
             PM10  = list(col = "PM10_mean", ref = 15))
OUTC <- c("respiratory_disease", "ILI", "pneumonia")

## ---- Gasparrini & Leone (2014) attributable-risk function ----
## Adapted from the published code accompanying the paper.
attrdl <- function(x, basis, cases, model, coef_vec = NULL, vcov_mat = NULL,
                   type = "an", dir = "back", tot = TRUE, cen, range = NULL,
                   sim = FALSE, nsim = 5000) {
  if (is.null(coef_vec)) coef_vec <- stats::coef(model)
  if (is.null(vcov_mat)) vcov_mat <- stats::vcov(model)
  L <- attr(basis, "lag")[2]
  if (!is.null(range)) x[x < range[1] | x > range[2]] <- cen
  ## lagged exposure matrix
  lagX <- sapply(0:L, function(l) c(rep(NA, l), head(x, length(x) - l)))
  if (!is.matrix(lagX)) lagX <- matrix(lagX, ncol = L + 1)
  ## design matrix using basis re-applied to lagged x's, centred at `cen`
  ## Cumulative log-RR per week from the cross-basis directly
  cb_names <- grep("^cb", names(coef_vec), value = TRUE)
  cb_idx <- which(names(coef_vec) %in% cb_names)
  ## basis matrix is already on the exposure series; rows 1..L are NA
  bmat <- basis
  eta <- as.numeric(bmat %*% coef_vec[cb_idx])
  ## centre at `cen` by subtracting the predicted log-RR at constant cen exposure:
  ## build a centred-version basis: predict at constant x = cen across all lags
  x_cen <- rep(cen, length(x))
  ## Re-create a temporary basis at cen using same spline knots/arguments:
  cb_at_cen <- crossbasis(x_cen, lag = attr(basis, "lag"),
                          argvar = attr(basis, "argvar"),
                          arglag = attr(basis, "arglag"))
  eta_cen <- as.numeric(cb_at_cen %*% coef_vec[cb_idx])
  rr_cum <- exp(eta - eta_cen)
  af_t <- 1 - 1 / rr_cum                     # week-specific attributable fraction (forward)
  ## drop first L weeks (no full lag window)
  valid <- seq(L + 1, length(x))
  af_t_v <- af_t[valid]; cases_v <- cases[valid]
  an_t <- af_t_v * cases_v
  AN <- sum(an_t, na.rm = TRUE)
  AF <- AN / sum(cases_v, na.rm = TRUE)
  point <- c(AN = AN, AF = AF)
  if (!sim) return(point)
  ## Monte Carlo on coefficients
  k <- length(coef_vec)
  sim_coef <- MASS::mvrnorm(nsim, coef_vec, vcov_mat)
  cb_names <- grep("^cb", names(coef_vec), value = TRUE)
  cb_idx <- which(names(coef_vec) %in% cb_names)
  rr_sims <- replicate(nsim, {
    NULL
  })
  ## Simpler bootstrap: refit predictions for each sampled coef vector
  AN_sims <- numeric(nsim); AF_sims <- numeric(nsim)
  ## crosspred internally uses model$coef; emulate by hand: predict log-RR matrix
  bvar <- attr(basis, "argvar")
  blag <- attr(basis, "arglag")
  ## Re-build the same basis matrix used in fitting:
  bmat <- basis
  cb_mat <- bmat[valid, , drop = FALSE]
  for (s in seq_len(nsim)) {
    eta <- as.numeric(cb_mat %*% sim_coef[s, cb_idx])
    rr  <- exp(eta)
    af  <- 1 - 1 / rr
    an  <- af * cases_v
    AN_sims[s] <- sum(an, na.rm = TRUE)
    AF_sims[s] <- AN_sims[s] / sum(cases_v, na.rm = TRUE)
  }
  list(AN = AN, AF = AF,
       AN_low = quantile(AN_sims, 0.025, na.rm = TRUE),
       AN_high = quantile(AN_sims, 0.975, na.rm = TRUE),
       AF_low = quantile(AF_sims, 0.025, na.rm = TRUE),
       AF_high = quantile(AF_sims, 0.975, na.rm = TRUE))
}

## Canonical numbering (outcome-major) matches cumulative_RR_table.csv
GRID <- data.frame(
  model_id  = paste0("M", 1:9),
  pollutant = c("NO2","PM2.5","PM10", "NO2","PM2.5","PM10", "NO2","PM2.5","PM10"),
  outcome   = c(rep("respiratory_disease",3), rep("ILI",3), rep("pneumonia",3)),
  stringsAsFactors = FALSE
)

results <- list()
for (i in seq_len(nrow(GRID))) {
  pname <- GRID$pollutant[i]; oname <- GRID$outcome[i]; mid <- GRID$model_id[i]
  pcfg  <- POLL[[pname]]
  cat(sprintf("[AR %s] %s -> %s\n", mid, pname, oname))
    cb <<- crossbasis(df[[pcfg$col]], lag = 4,
                      argvar = list(fun = "ns", df = 3),
                      arglag = list(fun = "ns", df = 3))
    fm <- as.formula(paste0(oname, " ~ cb + ns(week_index, df=8) + ",
                            "ns(temp_mean, df=3) + ns(humidity_mean, df=3)"))
    m <- glm(fm, family = quasipoisson(), data = df)
    ar <- attrdl(x = df[[pcfg$col]], basis = cb, cases = df[[oname]],
                 model = m, cen = pcfg$ref, sim = TRUE, nsim = 2000)
    total_cases <- sum(df[[oname]][5:nrow(df)], na.rm = TRUE)
    results[[length(results) + 1]] <- data.frame(
      model_id     = mid,
      pollutant    = pname,
      outcome      = oname,
      reference    = pcfg$ref,
      total_cases_in_window = total_cases,
      AN           = round(ar$AN, 1),
      AN_low       = round(ar$AN_low, 1),
      AN_high      = round(ar$AN_high, 1),
      AF_pct       = round(100 * ar$AF, 2),
      AF_pct_low   = round(100 * ar$AF_low, 2),
      AF_pct_high  = round(100 * ar$AF_high, 2)
    )
}
out <- do.call(rbind, results)
write_csv(out, file.path(OUT, "attributable_risk_table.csv"))

## Caveat note
writeLines(c(
  "Attributable risk caveat",
  "========================",
  "These attributable fractions are computed against benchmark counterfactual",
  "reference exposures (NO2=10, PM2.5=5, PM10=15 ug/m3) aligned with WHO 2021",
  "Global Air Quality Guidelines. They quantify the share of weekly cases that",
  "would NOT have occurred had ambient exposure been at the reference level,",
  "assuming the fitted DLNM is the true causal model.",
  "",
  "They are NOT proof of causality:",
  " - The model is an associational time-series regression.",
  " - Only the primary NO2 -> respiratory disease association has a 95% CI for",
  "   cumulative RR excluding 1; AR estimates for the other 8 models inherit",
  "   that non-significance and their intervals will frequently cross zero.",
  " - The 95% bootstrap intervals reflect only model-coefficient uncertainty,",
  "   not measurement error, model-form uncertainty, or exposure misclassification.",
  "",
  "Use the AR table to communicate magnitude under the stated counterfactual",
  "rather than to claim a deterministic preventable fraction."
), file.path(OUT, "AR_caveats.txt"))

cat("\nDone. Output:\n")
print(out)
