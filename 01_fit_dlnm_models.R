# Phase 2 - DLNM models (Milan air pollution and respiratory ED study)
#
# Script 01: primary DLNM plus the 9 single-pollutant screening models.
#   - fit the primary model (NO2 -> respiratory disease)
#   - fit all 9 single-pollutant DLNMs (3 pollutants x 3 outcomes)
#   - produce RR surfaces, contour plots, cumulative and lag-response curves
#   - write the model-reduction tables and basic diagnostics
#
# Input : analysis_ready.csv (from Phase 1, in the same folder)
# Output: model_objects/*.rds and *_summary.txt
#         rr_surfaces/*.png
#         diagnostics/*.png
#         model_reduction/*.csv
#
# Key rules carried over from the preprocessing phase:
#   - single-pollutant models only (PM2.5 and PM10 correlate at rho = 0.96)
#   - drop the W8/2025 outlier (outlier_flag == 1) for the main analysis
#   - temperature is a within-season confounder, not the seasonality proxy
#   - seasonality is controlled separately with ns(week_index, df = 8)
#
# Note: dlnm::crosspred finds the cross-basis coefficients from the variable
# name through deparse(substitute(basis)), so the cross-basis must be called
# 'cb' in the formula. We keep one global 'cb' and rebuild it before each fit.

# Load packages
suppressPackageStartupMessages({
  library(dlnm)        # crossbasis, crosspred, crossreduce
  library(splines)     # ns()
  library(MASS)        # glm.nb fallback (not used; main model is quasi-Poisson)
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(broom)
})

set.seed(42)

# Load data
INPUT  <- "analysis_ready.csv"
OUTDIR <- "."

# Make sure the output folders exist
for (d in c("model_objects", "rr_surfaces", "diagnostics", "model_reduction")) {
  dir.create(file.path(OUTDIR, d), showWarnings = FALSE, recursive = TRUE)
}

df <- read_csv(INPUT, show_col_types = FALSE) %>%
  arrange(week_index) %>%
  as.data.frame()

cat("Loaded analysis_ready.csv\n")
cat("  Rows    :", nrow(df), "\n")
cat("  Columns :", ncol(df), "\n")
cat("  Period  :", df$year_week[1], "->", df$year_week[nrow(df)], "\n")
cat("  W52/2025 partial flag   :", sum(df$w52_2025_flag), "\n")
cat("  W8/2025  outlier flag   :", sum(df$outlier_flag), "\n")

# Main analysis subset: drop the W8/2025 outlier
df_clean <- df[df$outlier_flag == 0, ]
cat("  n (outlier excluded)    :", nrow(df_clean), "\n\n")

# Missing-value check on the key variables
key_vars <- c("NO2_mean", "PM25_mean", "PM10_mean",
              "temp_mean", "humidity_mean",
              "respiratory_disease", "ILI", "pneumonia")
miss_audit <- data.frame(
  variable = key_vars,
  n_missing = sapply(key_vars, function(v) sum(is.na(df_clean[[v]])))
)
print(miss_audit)
stopifnot(all(miss_audit$n_missing == 0))
cat("\nNo missing values in any key variable. PASS.\n\n")

# Model lists
outcomes   <- c("respiratory_disease", "ILI", "pneumonia")
pollutants <- c("NO2_mean", "PM25_mean", "PM10_mean")

pol_label  <- c(NO2_mean = "NO2",  PM25_mean = "PM2.5", PM10_mean = "PM10")
out_label  <- c(respiratory_disease = "Respiratory",
                ILI = "ILI",
                pneumonia = "Pneumonia")

# Counterfactual reference values for the attributable risk and RR centering
pol_ref    <- c(NO2_mean = 10, PM25_mean = 5, PM10_mean = 15)

# Primary model settings
LAG     <- 4
VARDF   <- 3
LAGDF   <- 3
TIME_DF <- 8
TEMP_DF <- 3
HUM_DF  <- 3

# The cross-basis and the model formula are built inside fit_one() below,
# because crosspred relies on the literal variable name 'cb'.

# Diagnostics helper
compute_overdispersion <- function(m) {
  rp <- residuals(m, type = "pearson")
  sum(rp^2) / m$df.residual
}

# Figure helpers
save_rr_surface <- function(pred, file, main = "RR surface") {
  png(file, width = 1200, height = 900, res = 130)
  plot(pred, xlab = "Pollutant (ug/m3)",
       zlab = "RR", main = main,
       theta = 220, phi = 25, ltheta = 170,
       col = "steelblue", shade = 0.6, ticktype = "detailed")
  dev.off()
}

save_rr_contour <- function(pred, file, main = "RR contour") {
  png(file, width = 1200, height = 900, res = 130)
  plot(pred, "contour", xlab = "Pollutant (ug/m3)",
       ylab = "Lag (weeks)", main = main,
       key.title = title("RR"))
  dev.off()
}

save_cum_rr <- function(pred, file, main = "Cumulative RR", ref = NULL) {
  png(file, width = 1200, height = 800, res = 130)
  plot(pred, "overall",
       xlab = "Pollutant (ug/m3)",
       ylab = "Cumulative RR (lag 0-4)",
       main = main, col = "darkred", lwd = 2, ci.arg = list(col = "grey80"))
  if (!is.null(ref)) abline(v = ref, lty = 2, col = "blue")
  abline(h = 1, lty = 3)
  dev.off()
}

# fit_one() runs the whole workflow for one pollutant/outcome pair:
#   1. build the cross-basis (named 'cb' in the global environment)
#   2. fit the quasi-Poisson DLNM
#   3. run crosspred for the RR surface and the lag-response at p75
#   4. compute the cumulative, lag-specific and predictor-specific summaries
#   5. save the figures and return the model plus the extracted tables
fit_one <- function(data, outcome, pollutant, mid,
                    lag = LAG, vardf = VARDF, lagdf = LAGDF,
                    time_df = TIME_DF, temp_df = TEMP_DF, hum_df = HUM_DF,
                    save_figs = TRUE) {

  # 1. Cross-basis (must be in the global env and named 'cb')
  cb <<- crossbasis(data[[pollutant]],
                    lag = lag,
                    argvar = list(fun = "ns", df = vardf),
                    arglag = list(fun = "ns", df = lagdf))

  # 2. Quasi-Poisson DLNM
  fm <- as.formula(paste0(
    outcome, " ~ cb + ns(week_index, df = ", time_df, ") + ",
    "ns(temp_mean, df = ", temp_df, ") + ",
    "ns(humidity_mean, df = ", hum_df, ")"
  ))
  m <- glm(fm, family = quasipoisson(), data = data)
  od <- compute_overdispersion(m)

  pol_lab <- pol_label[pollutant]
  out_lab <- out_label[outcome]
  ref     <- unname(pol_ref[pollutant])
  p75     <- as.numeric(quantile(data[[pollutant]], 0.75, na.rm = TRUE))

  # 3. Predictions
  pred       <- crosspred(cb, m, cen = ref, bylag = 0.2, cumul = TRUE)
  pred_p75   <- crosspred(cb, m, at = p75, cen = ref, bylag = 0.2)
  pred_cum   <- crosspred(cb, m, at = p75, cen = ref, cumul = TRUE)
  pred_lags  <- crosspred(cb, m, at = p75, cen = ref, bylag = 1, cumul = FALSE)
  pred_pct   <- crosspred(cb, m,
                          at = as.numeric(quantile(data[[pollutant]],
                                                   c(0.25, 0.5, 0.75, 0.95))),
                          cen = ref, cumul = TRUE)

  # 4. Model-reduction summaries
  cum_row <- data.frame(
    model_id    = mid,
    pollutant   = pol_lab,
    outcome     = out_lab,
    reference   = ref,
    p75_value   = round(p75, 2),
    cum_RR      = round(pred_cum$allRRfit,  3),
    cum_RR_low  = round(pred_cum$allRRlow,  3),
    cum_RR_high = round(pred_cum$allRRhigh, 3)
  )

  lags <- 0:lag
  lag_rows <- do.call(rbind, lapply(lags, function(L) {
    data.frame(
      model_id  = mid,
      pollutant = pol_lab,
      outcome   = out_lab,
      lag       = L,
      exposure  = round(p75, 2),
      RR        = round(pred_lags$matRRfit[1, paste0("lag", L)],  3),
      RR_low    = round(pred_lags$matRRlow[1, paste0("lag", L)],  3),
      RR_high   = round(pred_lags$matRRhigh[1, paste0("lag", L)], 3)
    )
  }))

  probs <- c(0.25, 0.50, 0.75, 0.95)
  qs    <- as.numeric(quantile(data[[pollutant]], probs, na.rm = TRUE))
  pred_rows <- data.frame(
    model_id    = mid,
    pollutant   = pol_lab,
    outcome     = out_lab,
    percentile  = paste0("p", round(probs * 100)),
    exposure    = round(qs, 2),
    reference   = ref,
    cum_RR      = round(as.numeric(pred_pct$allRRfit),  3),
    cum_RR_low  = round(as.numeric(pred_pct$allRRlow),  3),
    cum_RR_high = round(as.numeric(pred_pct$allRRhigh), 3)
  )

  # 5. Figures
  if (save_figs) {
    tag <- sprintf("%s_%s_%s", mid, pol_lab, outcome)
    save_rr_surface(pred,
                    file.path(OUTDIR, "rr_surfaces",
                              paste0(tag, "_surface.png")),
                    main = sprintf("RR surface - %s -> %s", pol_lab, out_lab))
    save_rr_contour(pred,
                    file.path(OUTDIR, "rr_surfaces",
                              paste0(tag, "_contour.png")),
                    main = sprintf("RR contour - %s -> %s", pol_lab, out_lab))
    save_cum_rr(pred,
                file.path(OUTDIR, "rr_surfaces",
                          paste0(tag, "_cumRR.png")),
                main = sprintf("Cumulative RR - %s -> %s", pol_lab, out_lab),
                ref = ref)
    # Lag-response at p75
    png(file.path(OUTDIR, "rr_surfaces", paste0(tag, "_lag_p75.png")),
        1200, 800, res = 130)
    plot(pred_p75, "slices", var = p75, col = "darkgreen", lwd = 2,
         xlab = "Lag (weeks)", ylab = "RR",
         main = sprintf("Lag-response at p75 (=%.1f) - %s -> %s",
                        p75, pol_lab, out_lab),
         ci.arg = list(col = "grey80"))
    abline(h = 1, lty = 3)
    dev.off()
    # Exposure-response at lag 0
    pred_l0 <- crosspred(cb, m, cen = ref, at = seq(min(data[[pollutant]]),
                                                    max(data[[pollutant]]),
                                                    length.out = 30))
    png(file.path(OUTDIR, "rr_surfaces", paste0(tag, "_exp_lag0.png")),
        1200, 800, res = 130)
    plot(pred_l0, "slices", lag = 0, col = "darkblue", lwd = 2,
         xlab = "Pollutant (ug/m3)", ylab = "RR at lag 0",
         main = sprintf("Exposure-response @ lag 0 - %s -> %s",
                        pol_lab, out_lab),
         ci.arg = list(col = "grey80"))
    abline(h = 1, lty = 3); abline(v = ref, lty = 2, col = "blue")
    dev.off()
    # Exposure-response at lag 2
    png(file.path(OUTDIR, "rr_surfaces", paste0(tag, "_exp_lag2.png")),
        1200, 800, res = 130)
    plot(pred_l0, "slices", lag = 2, col = "darkorange", lwd = 2,
         xlab = "Pollutant (ug/m3)", ylab = "RR at lag 2",
         main = sprintf("Exposure-response @ lag 2 - %s -> %s",
                        pol_lab, out_lab),
         ci.arg = list(col = "grey80"))
    abline(h = 1, lty = 3); abline(v = ref, lty = 2, col = "blue")
    dev.off()
  }

  list(model    = m,
       cb       = cb,
       od       = od,
       cum      = cum_row,
       lag_tbl  = lag_rows,
       pred_tbl = pred_rows,
       meta = list(model_id = mid, outcome = outcome, pollutant = pollutant,
                   pol_label = pol_lab, out_label = out_lab,
                   lag = lag, vardf = vardf, lagdf = lagdf,
                   time_df = time_df, temp_df = temp_df, hum_df = hum_df,
                   n = nrow(data)))
}

# Fit the primary model first (NO2 -> respiratory disease)
cat("\n========================================================\n")
cat("PRIMARY MODEL: NO2 -> respiratory disease\n")
cat("========================================================\n")

primary <- fit_one(df_clean,
                   outcome   = "respiratory_disease",
                   pollutant = "NO2_mean",
                   mid       = "PRIMARY",
                   save_figs = TRUE)

cat("Convergence              :", primary$model$converged, "\n")
cat("Coefficient aliasing     :",
    any(is.na(coef(primary$model))), "(should be FALSE)\n")
cat("Overdispersion (Pearson) :", round(primary$od, 3),
    ifelse(primary$od > 1.2,
           " (> 1.2 -> quasi-Poisson justified)", ""), "\n")
cat("df residual              :", primary$model$df.residual, "\n")
cat("Cumulative RR (p75 vs 10 ug/m3): ",
    sprintf("%.3f (%.3f, %.3f)\n", primary$cum$cum_RR,
            primary$cum$cum_RR_low, primary$cum$cum_RR_high))

# Save the primary model
saveRDS(primary,
        file.path(OUTDIR, "model_objects",
                  "primary_NO2_respiratory_disease.rds"))
cat("\nPrimary model object saved.\n")

# Fit the 9 single-pollutant models (3 pollutants x 3 outcomes)
cat("\n========================================================\n")
cat("9 SINGLE-POLLUTANT SCREENING DLNMs\n")
cat("========================================================\n")
cat("RULE: never include NO2, PM2.5, PM10 simultaneously\n")
cat("      (collinearity rho >= 0.77; PM2.5 ~ PM10 rho = 0.96)\n\n")

grid <- expand.grid(pollutant = pollutants, outcome = outcomes,
                    stringsAsFactors = FALSE)
grid$model_id <- paste0("M", seq_len(nrow(grid)))
print(grid)

fits <- list()
metadata_rows <- list()
warnings_log  <- character(0)
cum_list      <- list()
lag_list      <- list()
pred_list     <- list()

for (i in seq_len(nrow(grid))) {
  pol <- grid$pollutant[i]
  out <- grid$outcome[i]
  mid <- grid$model_id[i]

  cat(sprintf("\n %-3s | %-9s -> %-20s\n",
              mid, pol_label[pol], out))

  fit <- withCallingHandlers(
    fit_one(df_clean, outcome = out, pollutant = pol, mid = mid),
    warning = function(w) {
      warnings_log[[length(warnings_log) + 1]] <<-
        sprintf("[%s %s/%s] %s", mid, pol_label[pol], out, conditionMessage(w))
      invokeRestart("muffleWarning")
    })

  fits[[mid]]      <- fit
  cum_list[[mid]]  <- fit$cum
  lag_list[[mid]]  <- fit$lag_tbl
  pred_list[[mid]] <- fit$pred_tbl

  m  <- fit$model
  cat(sprintf("        overdisp = %.2f  cumRR(p75|ref) = %.3f (%.3f, %.3f)\n",
              fit$od, fit$cum$cum_RR,
              fit$cum$cum_RR_low, fit$cum$cum_RR_high))

  saveRDS(fit,
          file.path(OUTDIR, "model_objects",
                    sprintf("%s_%s_%s.rds", mid, pol_label[pol], out)))

  sink(file.path(OUTDIR, "model_objects",
                 sprintf("%s_%s_%s_summary.txt", mid, pol_label[pol], out)))
  cat("Model ID :", mid, "\n")
  cat("Pollutant:", pol_label[pol], "(", pol, ")\n")
  cat("Outcome  :", out_label[out], "(", out, ")\n")
  cat("n        :", fit$meta$n, "\n")
  cat("Lag/vardf/lagdf  :", fit$meta$lag, "/",
      fit$meta$vardf, "/", fit$meta$lagdf, "\n")
  cat("time_df/temp_df/hum_df :", fit$meta$time_df, "/",
      fit$meta$temp_df, "/", fit$meta$hum_df, "\n")
  cat("Overdispersion   :", round(fit$od, 3), "\n\n")
  print(summary(m))
  sink()

  metadata_rows[[mid]] <- data.frame(
    model_id  = mid,
    pollutant = pol_label[pol],
    outcome   = out_label[out],
    n         = fit$meta$n,
    lag       = fit$meta$lag,
    var_df    = fit$meta$vardf,
    lag_df    = fit$meta$lagdf,
    time_df   = fit$meta$time_df,
    temp_df   = fit$meta$temp_df,
    hum_df    = fit$meta$hum_df,
    overdisp  = round(fit$od, 3),
    converged = m$converged
  )
}

# Save the warning log and metadata
writeLines(if (length(warnings_log)) warnings_log else "No warnings.",
           file.path(OUTDIR, "model_objects", "warnings_log.txt"))

model_metadata <- do.call(rbind, metadata_rows)
write_csv(model_metadata,
          file.path(OUTDIR, "model_objects", "model_metadata.csv"))
cat("\nModel metadata:\n")
print(model_metadata, row.names = FALSE)

# Model reduction tables
cat("\n========================================================\n")
cat("Model reduction summaries\n")
cat("========================================================\n")

cum_tbl      <- do.call(rbind, cum_list)
lag_tbl      <- do.call(rbind, lag_list)
predspec_tbl <- do.call(rbind, pred_list)

write_csv(cum_tbl,
          file.path(OUTDIR, "model_reduction", "cumulative_RR_table.csv"))
write_csv(lag_tbl,
          file.path(OUTDIR, "model_reduction", "lag_specific_RR_table.csv"))
write_csv(predspec_tbl,
          file.path(OUTDIR, "model_reduction",
                    "predictor_specific_RR_table.csv"))

cat("\nCumulative RR table (overall cumulative association):\n")
print(cum_tbl, row.names = FALSE)
cat("\nLag-specific RR (head):\n"); print(head(lag_tbl, 10), row.names = FALSE)
cat("\nPredictor-specific RR (head):\n")
print(head(predspec_tbl, 12), row.names = FALSE)

# Primary model diagnostics
cat("\n========================================================\n")
cat("PRIMARY MODEL DIAGNOSTICS\n")
cat("========================================================\n")

diag_dir <- file.path(OUTDIR, "diagnostics")

# The cross-basis with lag=4 leaves NA padding in the first 4 rows
# (the model uses rows 5..n only). Align the observed values to the fitted ones.
fit_idx <- as.integer(names(fitted(primary$model)))
obs     <- df_clean$respiratory_disease[fit_idx]
fit_val <- as.numeric(fitted(primary$model))
wk_idx  <- df_clean$week_index[fit_idx]
res_p   <- residuals(primary$model, type = "pearson")

# Observed vs fitted
png(file.path(diag_dir, "primary_observed_vs_fitted.png"),
    1200, 800, res = 130)
plot(obs, fit_val,
     pch = 16, col = rgb(0, 0, 0, 0.5),
     xlab = "Observed (visits/week)",
     ylab = "Fitted (visits/week)",
     main = "Primary model - Observed vs fitted")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# Residuals over time
png(file.path(diag_dir, "primary_residuals_time.png"),
    1400, 700, res = 130)
plot(wk_idx, res_p,
     type = "l", col = "steelblue",
     xlab = "Week index (1-155)", ylab = "Pearson residual",
     main = "Primary model - Residuals over time")
abline(h = 0, col = "red", lty = 2)
dev.off()

# Residual histogram
png(file.path(diag_dir, "primary_residuals_histogram.png"),
    1000, 800, res = 130)
hist(res_p,
     breaks = 25, col = "lightblue", border = "white",
     xlab = "Pearson residual",
     main = "Primary model - Residual distribution")
dev.off()

# ACF of residuals
png(file.path(diag_dir, "primary_residuals_acf.png"),
    1100, 800, res = 130)
acf(res_p, lag.max = 30, main = "Primary model - Residual ACF")
dev.off()

cat("Primary model diagnostics saved.\n")

# Session info
sink(file.path(OUTDIR, "model_objects", "sessionInfo.txt"))
print(sessionInfo())
sink()

cat("\nAll outputs written to: ", normalizePath(OUTDIR), "\n", sep = "")
cat("Done.\n")
