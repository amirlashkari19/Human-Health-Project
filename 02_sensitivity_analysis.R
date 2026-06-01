## ============================================================
##  P2 — Sensitivity analysis (Day 9–10 Trello card)
##  Probes the primary model (NO2 -> respiratory_disease).
##
##  Components:
##   (A) Main sensitivity grid: max lag, var df, lag df, time df, meteorology,
##       partial-week (W52/2025 include/exclude)
##   (B) Fourier seasonality vs natural-spline time
##   (C) Winter interaction (secondary analysis)
##
##  Outputs:
##   dlnm/sensitivity/dlnm_sensitivity_table.csv
##   dlnm/sensitivity/dlnm_sensitivity_fourier.csv
##   dlnm/sensitivity/dlnm_sensitivity_winter.csv
##   dlnm/sensitivity/dlnm_sensitivity_summary.png
## ============================================================

suppressPackageStartupMessages({
  library(dlnm); library(splines); library(MASS); library(dplyr); library(readr); library(ggplot2)
})

set.seed(20260515)

ROOT <- "/home/user/workspace/p2_deliverables"
DATA <- file.path(ROOT, "data/processed/analysis_ready.csv")
OUT  <- file.path(ROOT, "dlnm/sensitivity")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

df_all <- read_csv(DATA, show_col_types = FALSE) %>% filter(outlier_flag == 0)
stopifnot(nrow(df_all) == 154)

REF_NO2 <- 10
p75_no2 <- as.numeric(quantile(df_all$NO2_mean, 0.75))

## helper: fit one specification and return cumulative RR at p75 vs ref
fit_spec <- function(d, maxlag, vardf, lagdf, timedf, met = c("ns", "linear")) {
  met <- match.arg(met)
  cb <<- crossbasis(d$NO2_mean, lag = maxlag,
                    argvar = list(fun = "ns", df = vardf),
                    arglag = list(fun = "ns", df = lagdf))
  if (met == "ns") {
    fm <- as.formula(paste0("respiratory_disease ~ cb + ns(week_index, df=",
                            timedf, ") + ns(temp_mean, df=3) + ns(humidity_mean, df=3)"))
  } else {
    fm <- as.formula(paste0("respiratory_disease ~ cb + ns(week_index, df=",
                            timedf, ") + temp_mean + humidity_mean"))
  }
  m <- tryCatch(glm(fm, family = quasipoisson(), data = d),
                error = function(e) NULL)
  if (is.null(m) || !m$converged) {
    return(list(cum_RR = NA, cum_RR_low = NA, cum_RR_high = NA,
                dispersion = NA, converged = FALSE, warn = "fit_failed_or_not_converged"))
  }
  p <- tryCatch(crosspred(cb, m, cen = REF_NO2,
                          at = as.numeric(quantile(d$NO2_mean, 0.75)),
                          bylag = 0.2, cumul = TRUE),
                error = function(e) NULL)
  if (is.null(p)) {
    return(list(cum_RR = NA, cum_RR_low = NA, cum_RR_high = NA,
                dispersion = summary(m)$dispersion, converged = TRUE,
                warn = "crosspred_failed"))
  }
  list(cum_RR     = as.numeric(p$allRRfit),
       cum_RR_low = as.numeric(p$allRRlow),
       cum_RR_high= as.numeric(p$allRRhigh),
       dispersion = summary(m)$dispersion,
       converged  = TRUE, warn = "")
}

## ---------------- (A) Main sensitivity grid ----------------
grid <- expand.grid(
  maxlag        = c(2, 4, 6),
  vardf         = c(2, 3),
  lagdf         = c(2, 3),
  timedf        = c(6, 8, 10),
  meteorology   = c("ns", "linear"),
  partial_week  = c("include_W52", "exclude_W52"),
  stringsAsFactors = FALSE
)
cat(sprintf("Sensitivity grid: %d specifications\n", nrow(grid)))

res <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]
  d <- if (g$partial_week == "exclude_W52") df_all %>% filter(w52_2025_flag == 0) else df_all
  r <- fit_spec(d, g$maxlag, g$vardf, g$lagdf, g$timedf, g$meteorology)
  res[[i]] <- cbind(g, as.data.frame(r))
  if (i %% 25 == 0) cat(sprintf("  ...%d/%d\n", i, nrow(grid)))
}
sens_tbl <- do.call(rbind, res)

## classify robustness vs the primary point estimate (1.778, CI 1.052-3.006)
PRIMARY <- list(rr = 1.778, lo = 1.052, hi = 3.006)
sens_tbl$rel_diff_pct <- round(100 * (sens_tbl$cum_RR - PRIMARY$rr) / PRIMARY$rr, 1)
sens_tbl$ci_excludes_1 <- with(sens_tbl, !is.na(cum_RR_low) & cum_RR_low > 1)
sens_tbl$classification <- with(sens_tbl,
  ifelse(!converged | is.na(cum_RR), "unstable",
  ifelse(abs(rel_diff_pct) <= 25 & ci_excludes_1, "robust",
  ifelse(abs(rel_diff_pct) <= 50, "partially_robust", "unstable"))))

sens_tbl$cum_RR      <- round(sens_tbl$cum_RR, 3)
sens_tbl$cum_RR_low  <- round(sens_tbl$cum_RR_low, 3)
sens_tbl$cum_RR_high <- round(sens_tbl$cum_RR_high, 3)
sens_tbl$dispersion  <- round(sens_tbl$dispersion, 2)

write_csv(sens_tbl, file.path(OUT, "dlnm_sensitivity_table.csv"))

## quick summary plot — forest by maxlag
png(file.path(OUT, "dlnm_sensitivity_summary.png"), width = 1100, height = 800, res = 110)
print(
  ggplot(sens_tbl %>% filter(!is.na(cum_RR)),
         aes(x = factor(maxlag), y = cum_RR, colour = classification)) +
    geom_jitter(width = 0.25, alpha = 0.75, size = 1.8) +
    geom_hline(yintercept = 1, linetype = "dashed") +
    geom_hline(yintercept = PRIMARY$rr, linetype = "dotted", colour = "blue") +
    facet_grid(meteorology ~ partial_week) +
    scale_colour_manual(values = c(robust = "forestgreen",
                                   partially_robust = "darkorange",
                                   unstable = "red")) +
    labs(title = "Sensitivity grid — cumulative RR (p75 NO2 vs ref=10)",
         subtitle = sprintf("Primary point estimate: %.3f (dotted blue)", PRIMARY$rr),
         x = "Max lag (weeks)", y = "Cumulative RR") +
    theme_minimal(base_size = 12)
)
dev.off()

## ---------------- (B) Fourier vs natural-spline time ----------------
fit_fourier <- function(d) {
  cb <<- crossbasis(d$NO2_mean, lag = 4,
                    argvar = list(fun = "ns", df = 3),
                    arglag = list(fun = "ns", df = 3))
  m <- glm(respiratory_disease ~ cb + sin52 + cos52 + sin26 + cos26 +
             ns(temp_mean, df=3) + ns(humidity_mean, df=3),
           family = quasipoisson(), data = d)
  p <- crosspred(cb, m, cen = REF_NO2,
                 at = as.numeric(quantile(d$NO2_mean, 0.75)),
                 bylag = 0.2, cumul = TRUE)
  data.frame(seasonality = "fourier_sin52_cos52_sin26_cos26",
             cum_RR     = round(as.numeric(p$allRRfit), 3),
             cum_RR_low = round(as.numeric(p$allRRlow), 3),
             cum_RR_high= round(as.numeric(p$allRRhigh), 3),
             dispersion = round(summary(m)$dispersion, 2),
             AIC_proxy  = round(m$deviance + 2 * length(coef(m)), 1))
}

fit_spline_time <- function(d, dfv) {
  cb <<- crossbasis(d$NO2_mean, lag = 4,
                    argvar = list(fun = "ns", df = 3),
                    arglag = list(fun = "ns", df = 3))
  m <- glm(as.formula(paste0("respiratory_disease ~ cb + ns(week_index, df=", dfv,
                             ") + ns(temp_mean, df=3) + ns(humidity_mean, df=3)")),
           family = quasipoisson(), data = d)
  p <- crosspred(cb, m, cen = REF_NO2,
                 at = as.numeric(quantile(d$NO2_mean, 0.75)),
                 bylag = 0.2, cumul = TRUE)
  data.frame(seasonality = paste0("ns_week_index_df", dfv),
             cum_RR     = round(as.numeric(p$allRRfit), 3),
             cum_RR_low = round(as.numeric(p$allRRlow), 3),
             cum_RR_high= round(as.numeric(p$allRRhigh), 3),
             dispersion = round(summary(m)$dispersion, 2),
             AIC_proxy  = round(m$deviance + 2 * length(coef(m)), 1))
}

fourier_tbl <- rbind(
  fit_spline_time(df_all, 6),
  fit_spline_time(df_all, 8),
  fit_spline_time(df_all, 10),
  fit_fourier(df_all)
)
write_csv(fourier_tbl, file.path(OUT, "dlnm_sensitivity_fourier.csv"))

## ---------------- (C) Winter interaction (secondary) ----------------
d <- df_all
cb <<- crossbasis(d$NO2_mean, lag = 4,
                  argvar = list(fun = "ns", df = 3),
                  arglag = list(fun = "ns", df = 3))

## winter-stratified cross-basis: cb_winter = cb * winter_flag (Gasparrini-style interaction)
cb_winter <- cb * d$winter_flag
colnames(cb_winter) <- paste0("cbW", seq_len(ncol(cb_winter)))

m_winter <- glm(respiratory_disease ~ cb + cb_winter + winter_flag +
                  ns(week_index, df=8) + ns(temp_mean, df=3) + ns(humidity_mean, df=3),
                family = quasipoisson(), data = d)

## non-winter cumulative RR (cb only)
p_nonwint <- crosspred(cb, m_winter, cen = REF_NO2,
                       at = p75_no2, bylag = 0.2, cumul = TRUE)

## winter cumulative RR: combine cb + cb_winter
## (refit a smaller representative model just to get a winter-only point estimate)
m_winteronly <- tryCatch({
  d_w <- d %>% filter(winter_flag == 1)
  cb_w <<- crossbasis(d_w$NO2_mean, lag = 4,
                      argvar = list(fun = "ns", df = 3),
                      arglag = list(fun = "ns", df = 3))
  glm(respiratory_disease ~ cb_w + ns(week_index, df = 4) +
        ns(temp_mean, df = 3) + ns(humidity_mean, df = 3),
      family = quasipoisson(), data = d_w)
}, error = function(e) NULL)

if (!is.null(m_winteronly) && m_winteronly$converged) {
  pw <- crosspred(cb_w, m_winteronly, cen = REF_NO2,
                  at = p75_no2, bylag = 0.2, cumul = TRUE)
  winter_rr <- as.numeric(pw$allRRfit)
  winter_lo <- as.numeric(pw$allRRlow)
  winter_hi <- as.numeric(pw$allRRhigh)
  winter_n  <- nrow(m_winteronly$model)
  winter_note <- "winter-only refit (winter_flag==1 sub-sample)"
} else {
  winter_rr <- winter_lo <- winter_hi <- NA
  winter_n <- 0
  winter_note <- "winter-only refit failed; interaction-coef LRT only"
}

interaction_lrt <- tryCatch({
  m_main <- glm(respiratory_disease ~ cb + winter_flag +
                  ns(week_index, df=8) + ns(temp_mean, df=3) + ns(humidity_mean, df=3),
                family = quasipoisson(), data = d)
  anova(m_main, m_winter, test = "F")
}, error = function(e) NULL)

winter_tbl <- data.frame(
  scenario = c("non_winter_(cb_only_in_interaction_model)", "winter_only_refit"),
  n        = c(nrow(d), winter_n),
  cum_RR   = c(round(as.numeric(p_nonwint$allRRfit), 3),
               round(winter_rr, 3)),
  cum_RR_low = c(round(as.numeric(p_nonwint$allRRlow), 3),  round(winter_lo, 3)),
  cum_RR_high= c(round(as.numeric(p_nonwint$allRRhigh), 3), round(winter_hi, 3)),
  note     = c("Reference: cb_winter set to 0 (non-winter weeks)", winter_note)
)
write_csv(winter_tbl, file.path(OUT, "dlnm_sensitivity_winter.csv"))

if (!is.null(interaction_lrt)) {
  capture.output(print(interaction_lrt),
                 file = file.path(OUT, "dlnm_sensitivity_winter_interaction_F.txt"))
}

cat("Sensitivity analysis complete.\n")
cat(sprintf("  Grid n=%d, robust=%d, partially_robust=%d, unstable=%d\n",
            nrow(sens_tbl),
            sum(sens_tbl$classification == "robust"),
            sum(sens_tbl$classification == "partially_robust"),
            sum(sens_tbl$classification == "unstable")))
print(fourier_tbl)
print(winter_tbl)
