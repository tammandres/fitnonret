# ------------------
# Sensitivity check: FIT 70-day non-return (type 2) logistic model,
# full data vs most recent year (fit_request_date_corrected >= 2023-01-01)
#
# Fits the same logistic model as models.R (section 2), but only for the
# nonret2_days70 outcome, and compares odds ratios between:
#   (1) the full analysis dataset
#   (2) the dataset restricted to requests from 2023-01-01 onwards
#
# Predicted probabilities are not retrieved. Output is a single table of
# odds ratios (point estimate, 95% CI, p value) side by side.
#
# This script was generated with Claude Code and manually verified.
# ------------------

# Packages
library(readr)

setwd("Z:/fit_nonreturn_paper_20250417")

out_path <- "Z:/fit_nonreturn_paper_20250417/results"
dir.create(file.path(out_path), showWarnings = FALSE)

# When TRUE, the full-dataset model is fitted on the training set only
# (test_set == 0), matching models.R section 2. When FALSE, it is fitted on
# the entire dataset. The >= 2023 model is always fitted on the full 2023+
# data (no train/test split) regardless of this flag.
data_split <- TRUE


# ---- 1. Read and prepare data ----
# Uses first_fit_nonret.csv (one row per patient), matching the main
# logistic analysis in models.R section 2.
df <- read_csv("Z:/fit_nonreturn_paper_20250417/data/first_fit_nonret.csv")
nrow(df)

# Set White as reference for ethnicity
df$ethnicity <- factor(df$ethnicity)
df$ethnicity <- relevel(df$ethnicity, ref = 'White')

# Set youngest age group as reference
df$age_group <- factor(df$age_group)
df$age_group <- relevel(df$age_group, ref = '18-39')

# IMD to factor, missing coded as 'Not known', least deprived (5) as reference
df$imd_quintile_factor <- df$imd_quintile
df$imd_quintile_factor[is.na(df$imd_quintile)] <- 'Not known'
df$imd_quintile_factor <- factor(df$imd_quintile_factor)
df$imd_quintile_factor <- relevel(df$imd_quintile_factor, ref = '5')

# Request month to factor, January (1) as reference
df$request_month_factor <- factor(df$request_month)
df$request_month_factor <- relevel(df$request_month_factor, ref = '1')

# Drop patients without 70-day follow-up (same as models.R section 2)
nrow(df)
df <- df[df$fit_request_date_fu >= 70, ]
nrow(df)

# Drop patients who died before 70-day follow-up without returning a test
#  (matches models.R section 2; the censoring indicator in dataprep.py does
#   not account for deaths occurring within the follow-up window)
mask <- (df$censored == 1) & (!is.na(df$days_to_death)) & (df$days_to_death <= 70)
sum(mask)
df <- df[!mask, ]
nrow(df)

# Guard: outcome should be fully observed in this population
stopifnot(sum(is.na(df$nonret2_days70)) == 0)

# Parse request date so the 2023 subset can be defined
df$fit_request_date_corrected <- as.Date(df$fit_request_date_corrected)


# ---- 2. Helper: fit model and extract odds ratios ----
# The request_year reference is set to the earliest year *present* in the
# supplied data, so the model works both on the full data (ref 2017) and on
# the 2023+ subset (ref 2023) without an empty reference level.
fit_and_extract <- function(data) {

  data$request_year_factor <- factor(data$request_year)
  data$request_year_factor <- relevel(data$request_year_factor,
                                       ref = as.character(min(data$request_year)))

  formula <- paste("nonret2_days70 ~ gender_male + age_group + ethnicity +",
                   "imd_quintile_factor + request_year_factor + request_month_factor")
  fit <- glm(as.formula(formula), data = data, family = binomial(link = "logit"))

  tab <- data.frame(coef(summary(fit)), check.names = FALSE)
  tab$variable <- rownames(tab)
  rownames(tab) <- NULL

  est <- tab$Estimate
  se  <- tab$`Std. Error`
  pval <- tab$`Pr(>|z|)`

  or     <- round(exp(est), 2)
  or_low <- round(exp(est - 1.96 * se), 2)
  or_upp <- round(exp(est + 1.96 * se), 2)
  or_ci  <- paste0(or_low, "-", or_upp)

  # p value: round to 3 dp, floor small values to <0.001
  or_pval <- round(pval, 3)
  or_pval <- format(or_pval, nsmall = 3, scientific = FALSE, trim = TRUE)
  or_pval[pval < 0.001] <- "<0.001"

  data.frame(variable = tab$variable,
             or = or,
             or_ci = or_ci,
             or_pval = or_pval,
             stringsAsFactors = FALSE)
}


# ---- 3. Fit to full data and to 2023+ subset ----
# Full-dataset model: training set only if data_split is TRUE, else all data.
if (data_split) {
  df_fit <- df[df$test_set == 0, ]
} else {
  df_fit <- df
}
nrow(df_fit)
res_full <- fit_and_extract(df_fit)

# 2023+ model: always the full 2023+ data, no train/test split
df_2023 <- df[df$fit_request_date_corrected >= as.Date("2023-01-01"), ]
nrow(df_2023)
res_2023 <- fit_and_extract(df_2023)

# Rename the 2023 columns before joining
colnames(res_2023) <- c("variable", "or2023", "or2023_ci", "or2023_pval")


# ---- 4. Combine into a single table ----
# Full outer join keeps coefficients that appear in only one model (e.g. year
# levels that exist only in the full data). Ordering follows the full model,
# with any subset-only coefficients appended at the end.
res <- merge(res_full, res_2023, by = "variable", all = TRUE, sort = FALSE)

order_idx <- match(res_full$variable, res$variable)
extra_idx <- setdiff(seq_len(nrow(res)), order_idx)
res <- res[c(order_idx, extra_idx), ]
rownames(res) <- NULL

res <- res[, c("variable", "or", "or_ci", "or_pval",
               "or2023", "or2023_ci", "or2023_pval")]

print(res)

out_name <- paste0("logistic2023_or_comparison_datasplit-", tolower(as.character(data_split)), ".csv")
write.csv(res, file.path(out_path, out_name), row.names = FALSE)
