# Install the exact package versions used in the analysis.
# install.packages() has no version argument, so remotes::install_version() is used instead.
# Note: survival is used in models.R but ships with R as a recommended package.
if (!requireNamespace('remotes', quietly = TRUE)) install.packages('remotes')

pkgs <- c(readr   = '2.1.5',
          ggplot2 = '3.5.2',
          mice    = '3.18.0',
          mgcv    = '1.9-3',
          dplyr   = '1.1.4')

for (p in names(pkgs)) {
  remotes::install_version(p, version = pkgs[[p]], repos = 'https://cloud.r-project.org', upgrade = 'never')
}
