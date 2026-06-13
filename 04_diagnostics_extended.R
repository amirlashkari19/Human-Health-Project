# Phase 2 - Extended diagnostics for the primary model (NO2 -> respiratory disease)
#
# Adds a Cook's distance influence check, a DFBETA check on the cross-basis
# coefficients, and a comparison of including vs excluding the partial week
# W52/2025.
#
# Input : analysis_ready.csv (same folder)
# Output: diagnostics/*.png and *.csv

suppressPackageStartupMessages({
  library(dlnm); library(splines); library(MASS); library(dplyr); library(readr); library(ggplot2)
})

DATA <- "analysis_ready.csv"
OUT  <- "diagnostics"
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

df_full <- read_csv(DATA, show_col_types = FALSE) %>% filter(outlier_flag == 0)
stopifnot(nrow(df_full) == 154)

# Influence: Cook's distance on the primary model
df <- df_full
cb <<- crossbasis(df$NO2_mean, lag = 4,
                  argvar = list(fun = "ns", df = 3),
                  arglag = list(fun = "ns", df = 3))
m_primary <- glm(respiratory_disease ~ cb + ns(week_index, df=8) +
                   ns(temp_mean, df=3) + ns(humidity_mean, df=3),
                 family = quasipoisson(), data = df)

cooks <- cooks.distance(m_primary)
fit_idx <- as.integer(names(fitted(m_primary)))
infl_df <- data.frame(row = fit_idx,
                      week_index = df$week_index[fit_idx],
                      year_week  = df$year_week[fit_idx],
                      cooks      = cooks)

png(file.path(OUT, "primary_cooks_distance.png"), width = 1000, height = 500, res = 110)
print(
  ggplot(infl_df, aes(x = week_index, y = cooks)) +
    geom_segment(aes(xend = week_index, yend = 0), colour = "steelblue") +
    geom_point(colour = "steelblue4", size = 1.2) +
    geom_hline(yintercept = 4 / nrow(infl_df), linetype = "dashed", colour = "red") +
    labs(title = "Cook's distance - primary model (NO2 -> respiratory)",
         subtitle = "Dashed line: 4/n rule-of-thumb threshold",
         x = "Week index", y = "Cook's distance") +
    theme_minimal(base_size = 12)
)
dev.off()

top10 <- infl_df %>% arrange(desc(cooks)) %>% head(10)
write_csv(top10, file.path(OUT, "primary_cooks_top10.csv"))

# Compare including vs excluding the partial week W52/2025
df_in  <- df_full                     # includes W52/2025 (partial week)
df_out <- df_full %>% filter(w52_2025_flag == 0)

fit_no2_resp <- function(d) {
  cb <<- crossbasis(d$NO2_mean, lag = 4,
                    argvar = list(fun = "ns", df = 3),
                    arglag = list(fun = "ns", df = 3))
  m <- glm(respiratory_disease ~ cb + ns(week_index, df=8) +
             ns(temp_mean, df=3) + ns(humidity_mean, df=3),
           family = quasipoisson(), data = d)
  p <- crosspred(cb, m, cen = 10, at = quantile(d$NO2_mean, 0.75),
                 bylag = 0.2, cumul = TRUE)
  data.frame(n = nrow(d),
             dispersion = round(summary(m)$dispersion, 3),
             cum_RR     = round(as.numeric(p$allRRfit), 3),
             cum_RR_low = round(as.numeric(p$allRRlow), 3),
             cum_RR_high= round(as.numeric(p$allRRhigh), 3))
}

cmp <- rbind(
  cbind(scenario = "include_W52_2025", fit_no2_resp(df_in)),
  cbind(scenario = "exclude_W52_2025", fit_no2_resp(df_out))
)
write_csv(cmp, file.path(OUT, "primary_W52_2025_in_vs_out.csv"))

# DFBETAs on the cross-basis coefficients
dfb <- dfbetas(m_primary)
cb_cols <- grep("^cb", colnames(dfb), value = TRUE)
maxdfb <- apply(abs(dfb[, cb_cols, drop = FALSE]), 1, max)
mdf <- data.frame(row = fit_idx,
                  week_index = df$week_index[fit_idx],
                  year_week  = df$year_week[fit_idx],
                  max_abs_dfbeta_cb = maxdfb)
png(file.path(OUT, "primary_max_dfbeta_cb.png"), width = 1000, height = 500, res = 110)
print(
  ggplot(mdf, aes(x = week_index, y = max_abs_dfbeta_cb)) +
    geom_line(colour = "darkorange") +
    geom_hline(yintercept = 2 / sqrt(nrow(mdf)), linetype = "dashed", colour = "red") +
    labs(title = "Max |DFBETA| on cross-basis coefficients - primary model",
         subtitle = "Dashed line: 2/sqrt(n) rule-of-thumb",
         x = "Week index", y = "max |DFBETA| across cb coefs") +
    theme_minimal(base_size = 12)
)
dev.off()

cat("Extended diagnostics complete.\n")
print(cmp)
cat("\nTop 10 Cook's distance weeks:\n")
print(top10)
