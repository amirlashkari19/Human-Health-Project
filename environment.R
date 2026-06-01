# environment.R — R package dependencies for P2 DLNM scripts
# Run once to install all required packages:
#   Rscript environment.R

required_packages <- c(
  "dlnm",     # Distributed Lag Non-Linear Models
  "splines",  # ns() natural splines
  "MASS",     # glm.nb, mvrnorm
  "dplyr",    # data manipulation
  "ggplot2",  # figures
  "readr",    # read_csv / write_csv
  "broom"     # tidy model summaries
)

missing <- required_packages[!required_packages %in% installed.packages()[, "Package"]]

if (length(missing) > 0) {
  message("Installing missing packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
} else {
  message("All required packages are already installed.")
}

invisible(lapply(required_packages, library, character.only = TRUE))
message("All packages loaded successfully.")
