# ------------------
# Analysis of FIT test non-return
# 5 May 2025
# Andres Tamm
# ------------------

# Packages
library(survival)
library(readr)
library(ggplot2)
library(mice)
library(mgcv)
library(dplyr)


setwd("Z:/fit_nonreturn_paper_20250417")

out_path <- "Z:/fit_nonreturn_paper_20250417/results"
dir.create(file.path(out_path), showWarnings = FALSE)

run_on_imputed_data <- FALSE


# ---- 1. Read and prepare data ----

# Read data
df <- read_csv("Z:/fit_nonreturn_paper_20250417/data/first_fit_nonret.csv")
nrow(df)

# Drop patients without deprivation? No 
sum(is.na(df$imd_quintile))
sum(is.na(df[df$test_set==0, 'imd_quintile']))
#df <- df[!is.na(df$imd_quintile),]
#nrow(df)

# IMD to factor
df$imd_quintile_factor <- df$imd_quintile
df$imd_quintile_factor[is.na(df$imd_quintile)] <- 'Not known'
df$imd_quintile_factor <- factor(df$imd_quintile_factor)
df$imd_quintile_factor <- relevel(df$imd_quintile_factor, ref='5')

# Set White as reference
df$ethnicity <- factor(df$ethnicity)
df$ethnicity <- relevel(df$ethnicity, ref='White')

# Set youngest age group as ref
df$age_group <- factor(df$age_group)
df$age_group <- relevel(df$age_group, ref='18-39')

# Time to factor
df$request_year_factor <- factor(df$request_year)
df$request_year_factor <- relevel(df$request_year_factor, ref='2017')
df$request_month_factor <- factor(df$request_month)
df$request_month_factor <- relevel(df$request_month_factor, ref='1')

# Use time with daily precision (more precision not needed)
df$days_to_return <- round(df$days_to_return, digits=0)
df$days_to_return_type1 <- round(df$days_to_return_type1, digits=0)

# Drop patients without 70-day follow-up (2204)
# But before that, store full data for time to event analysis
df_full <- df

n0 <- nrow(df)
df <- df[df$fit_request_date_fu >= 70,]
n1 <- nrow(df)
sum(is.na(df$nonret2_days70))
sum(is.na(df$nonret2_days14))

# Drop patients who died before 70-day follow-up without returning a test (116)
#  This is necessary, because the censoring indicator that was created in dataprep.py
#  only considers time passed from FIT request to datacut date, not deaths occurring in between.
#  Ideally, the censoring indicator could have been updated to account for deaths.
mask <- (df$censored == 1) & (!is.na(df$days_to_death)) & (df$days_to_death <= 70)
sum(mask)
nrow(df)
df <- df[!mask,]
nrow(df)

# Train test split
df_train <- df[df$test_set == 0,]
df_test <- df[df$test_set == 1,]
df_train <- df_train[,colnames(df_train) != 'test_set']

# Dbl check that the few >365 returns are dropped
max(df[df$fit_return_type1 == 1,]$days_to_return_type1)

nrow(df_train)  ## 62093
nrow(df_test)   ## 7147
mean(df_train$nonret2_days70) ## 11.8%
mean(df_test$nonret2_days70)  ## 14.8% 
sum(df_train$nonret2_days70) ## 7304
sum(df_test$nonret2_days70)  ## 1059 
0.9 * nrow(df)


# ---- 2. Fit logistic models (outcome is test nonreturn within 7, 14, 28 or 70 days) ----
# Code for fitting models is partly from Madison Luick
coef_logistic <- data.frame()
pred_logistic <- data.frame()

for(outcome_col in c('nonret2_days7', 'nonret2_days14', 'nonret2_days28', 'nonret2_days70',
                     'nonret1_days7', 'nonret1_days14', 'nonret1_days28', 'nonret1_days70')){
  print(outcome_col)
  
  # Fit simplest logistic model
  formula = paste(outcome_col, "~ gender_male + age_group + ethnicity + imd_quintile_factor + request_year_factor + request_month_factor")
  fit_logistic <- glm(as.formula(formula), data=df_train, family = binomial(link="logit"))
  summary(fit_logistic)
  
  # Reformat the model summary
  tab_logistic <- data.frame(coef(summary(fit_logistic)))
  tab_logistic$Predictor <- rownames(tab_logistic)
  tab_logistic <- tab_logistic[,c(5, 1,2,3,4)]
  rownames(tab_logistic) <- NULL
  tab_logistic$OR <- round(exp(tab_logistic$Estimate),2)
  tab_logistic$`OR low` <- round(exp(tab_logistic$Estimate - 1.96 * tab_logistic$`Std..Error`),2)
  tab_logistic$`OR upp` <- round(exp(tab_logistic$Estimate + 1.96 * tab_logistic$`Std..Error`),2)
  tab_logistic <- tab_logistic[,c('Predictor', 'OR', 'OR low', 'OR upp', 'Pr...z..')]
  colnames(tab_logistic)[5] <- 'P value'
  
  tab_logistic$`P value sig` <- ''
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.05] <- '*'
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.01] <- '**'
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.001] <- '***'
  
  tab_logistic$`P value` <- round(tab_logistic$`P value`, 4)
  tab_logistic$`P value`[tab_logistic$`P value` == 0] <- '<0.001'
  tab_logistic$`P value`[tab_logistic$`P value` == 1e-4] <- '0.001'
  print(tab_logistic)
  
  tab_logistic$outcome <- outcome_col
  coef_logistic <- rbind(coef_logistic, tab_logistic)
  
  # Create predictions on the held out test set
  # What to do with year 2024 not in factor level? Atm just assigning it the highest possible value (2023)
  df_new <- df
  df_new$request_year_factor[df_new$request_year_factor == 2024] <- 2023
  pred <- predict(fit_logistic, newdata=df_new, type='response')
  pred <- data.frame(patient_id=df$patient_id, 
                     icen=df$icen, 
                     y_pred=pred, 
                     y_true=df_new[[outcome_col]],
                     outcome=outcome_col,
                     test_set=df$test_set)
  pred_logistic <- rbind(pred_logistic, pred)
  
}

## Save
write.csv(coef_logistic, file.path(out_path, 'logistic-coef.csv'), row.names=FALSE)
write.csv(pred_logistic, file.path(out_path, 'logistic-pred.csv'), row.names=FALSE)


# ---- 3. Fit GAMs (outcome is test nonreturn within 7, 14, 28 or 70 days) ----
# Note: the GAM is fitted, but only to complete cases (IMD has missing values) implicitly.
table_gam_linear <- data.frame()   # Stores a summary of linear effects
table_gam_smooth <- data.frame()   # Stores a summary of smooths
plot_gam_link <- data.frame()      # Stores data for plotting smooths on link function scale
plot_gam_response <- data.frame()  # Stores data for plotting smooths on response scale
pred_gam <- data.frame()

for(outcome_col in c('nonret2_days7', 'nonret2_days14', 'nonret2_days28', 'nonret2_days70',
                     'nonret1_days7', 'nonret1_days14', 'nonret1_days28', 'nonret1_days70')){
  print(outcome_col)
  
  # Fit GAM
  formula = paste(outcome_col, "~ gender_male + s(age_at_request) + ethnicity + s(imd_quintile, k=4) + s(request_year, k=7) + s(request_month, k=11)")
  fit_gam <- gam(as.formula(formula), data=df_train, family = binomial(link="logit"))
  
  # Get data for plotting smooths on link function scale
  # In the plot object, se is already multiplied by se.mul
  plot_gam <- plot.gam(fit_gam, rug=FALSE, pages=1)
  df_gam <- data.frame()
  for(i in 1:length(plot_gam)){
    plot_sub <- plot_gam[[i]]
    print(plot_sub$se.mul)
    dfsub <- data.frame(x=plot_sub$x, y=plot_sub$fit, se=plot_sub$se, var=plot_sub$xlab)
    dfsub$y_low <- dfsub$y - dfsub$se
    dfsub$y_upp <- dfsub$y + dfsub$se
    dfsub$outcome <- outcome_col
    df_gam <- rbind(df_gam, dfsub)
  }
  plot_gam_link <- rbind(plot_gam_link, df_gam)

  ## Sanity check
  p <- ggplot(data=df_gam)
  p <- p + geom_line(aes(x=x, y=y)) + geom_ribbon(aes(x=x, ymin=y_low, ymax=y_upp), alpha=0.2)
  p <- p + facet_wrap(vars(var), ncol=2, scales='free')
  p <- p + expand_limits(y=0)
  p <- p + ylab("Contribution to linear predictor") + xlab("Value of the predictor variable")
  p
  
  # Get data for plotting smooths on response scale
  df_gam <- data.frame()
  
  ## For each smooth, the other variables are fixed at these values
  values <- list(age_at_request = median(df$age_at_request),
                 gender_male = 0, 
                 ethnicity='White', 
                 imd_quintile=median(df$imd_quintile, na.rm=TRUE),
                 request_year=2024,
                 request_month=1)
  
  ## Create a dataframe that contains a grid of values for the variable to be plotted
  ## while values of other variables are fixed
  values_new <- values
  values_new$age_at_request <- seq(min(df$age_at_request), max(df$age_at_request), length = 200)
  new_data <- expand.grid(values_new)
  pred <- predict(fit_gam, new_data, type = "response", se.fit = TRUE)
  df_pred = data.frame(x=new_data$age_at_request, y=pred$fit, se=pred$se.fit)
  df_pred$var <- 'age_at_request'
  df_gam <- rbind(df_gam, df_pred)
  
  values_new <- values
  values_new$imd_quintile <- seq(1, 5, 1)
  new_data <- expand.grid(values_new)
  pred <- predict(fit_gam, new_data, type = "response", se.fit = TRUE)
  df_pred = data.frame(x=new_data$imd_quintile, y=pred$fit, se=pred$se.fit)
  df_pred$var <- 'imd_quintile'
  df_gam <- rbind(df_gam, df_pred)
  
  values_new <- values
  values_new$request_year <- seq(min(df$request_year), max(df$request_year), length=length(unique(df$request_year)))
  new_data <- expand.grid(values_new)
  pred <- predict(fit_gam, new_data, type = "response", se.fit = TRUE)
  df_pred = data.frame(x=new_data$request_year, y=pred$fit, se=pred$se.fit)
  df_pred$var <- 'request_year'
  df_gam <- rbind(df_gam, df_pred)
  
  values_new <- values
  values_new$request_month <- seq(1, 12, 1)
  new_data <- expand.grid(values_new)
  pred = predict(fit_gam, new_data, type = "response", se.fit = TRUE)
  df_pred = data.frame(x=new_data$request_month, y=pred$fit, se=pred$se.fit)
  df_pred$var <- 'request_month'
  df_gam <- rbind(df_gam, df_pred)
  
  df_gam$y_low <- df_gam$y - 2 * df_gam$se
  df_gam$y_upp <- df_gam$y + 2 * df_gam$se
  df_gam$outcome <- outcome_col
  
  plot_gam_response <- rbind(plot_gam_response, df_gam)
  
  ## Sanity check
  p <- ggplot(data=df_gam)
  p <- p + geom_line(aes(x=x, y=y)) + geom_ribbon(aes(x=x, ymin=y_low, ymax=y_upp), alpha=0.2)
  p <- p + facet_wrap(vars(var), ncol=2, scales='free')
  p <- p + expand_limits(y=0)
  p <- p + ylab("Probability of test nonreturn") + xlab("Value of the predictor variable")
  p
  
  # Summarise in a table
  sum_gam <- summary(fit_gam)
  
  tab_lin <- data.frame(sum_gam$p.table)
  tab_lin$Predictor <- rownames(tab_lin)
  tab_lin <- tab_lin[,c(5, 1,2,3,4)]
  rownames(tab_lin) <- NULL
  tab_lin$OR <- round(exp(tab_lin$Estimate),2)
  tab_lin$`OR low` <- round(exp(tab_lin$Estimate - 1.96 * tab_lin$`Std..Error`),2)
  tab_lin$`OR upp` <- round(exp(tab_lin$Estimate + 1.96 * tab_lin$`Std..Error`),2)
  tab_lin <- tab_lin[,c('Predictor', 'OR', 'OR low', 'OR upp', 'Pr...z..')]
  colnames(tab_lin)[5] <- 'P value'
  
  tab_lin$`P value sig` <- ''
  tab_lin$`P value sig`[tab_lin$`P value` < 0.05] <- '*'
  tab_lin$`P value sig`[tab_lin$`P value` < 0.01] <- '**'
  tab_lin$`P value sig`[tab_lin$`P value` < 0.001] <- '***'
  
  tab_lin$`P value` <- round(tab_lin$`P value`, 4)
  tab_lin$`P value`[tab_lin$`P value` == 0] <- '<0.001'
  tab_lin$`P value`[tab_lin$`P value` == 1e-4] <- '0.001'
  tab_lin$outcome <- outcome_col
  
  print(tab_lin)
  
  tab_smooth <- data.frame(sum_gam$s.table)
  tab_smooth$Predictor <- rownames(tab_smooth)
  tab_smooth <- tab_smooth[,c(5,1,2,3,4)]
  colnames(tab_smooth)[5] <- 'P value'
  tab_smooth$`P value sig` <- ''
  tab_smooth$`P value sig`[tab_smooth$`P value` < 0.05] <- '*'
  tab_smooth$`P value sig`[tab_smooth$`P value` < 0.01] <- '**'
  tab_smooth$`P value sig`[tab_smooth$`P value` < 0.001] <- '***'
  tab_smooth$`P value` <- round(tab_smooth$`P value`, 4)
  tab_smooth$`P value`[tab_smooth$`P value` == 0] <- '<0.001'
  tab_smooth$`P value`[tab_smooth$`P value` == 1e-4] <- '0.001'
  rownames(tab_smooth) <- NULL
  tab_smooth$outcome <- outcome_col
  
  print(tab_smooth)
  
  table_gam_linear <- rbind(table_gam_linear, tab_lin)
  table_gam_smooth <- rbind(table_gam_smooth, tab_smooth)
  
  # Get model predictions
  pred <- predict.gam(fit_gam, newdata=df, type='response')
  pred <- data.frame(patient_id=df$patient_id, 
                     icen=df$icen, 
                     y_pred=pred,
                     y_true=df[[outcome_col]], 
                     outcome=outcome_col,
                     test_set=df$test_set
                     )
  pred_gam <- rbind(pred_gam, pred)
}

## Save to disk
write.csv(table_gam_linear, file.path(out_path, 'gam-effect_linear.csv'), row.names=FALSE)
write.csv(table_gam_smooth, file.path(out_path, 'gam-effect_smooth.csv'), row.names=FALSE)
write.csv(plot_gam_link, file.path(out_path, 'gam-plot_link.csv'), row.names=FALSE)
write.csv(plot_gam_response, file.path(out_path, 'gam-plot_resp.csv'), row.names=FALSE)
write.csv(pred_gam, file.path(out_path, 'gam-pred.csv'), row.names=FALSE)


# ---- 4. Time-to-event analysis (analysing type 1 and type 2 nonreturn separately) ----
# Note: results from this section are not included in the publication. 
# Given the short required follow up of 70 days, and a small number of deaths (116) during the 70-day period,
# it is unlikely that Cox models would provide notably better estimates or insight over logistic models.
# The logistic models are also simpler and easier to interpret.
# The KM-curves computed here use training set data only (KM-curves on full data are generated in section 7)

# Train test split
df_full_train <- df_full[df_full$test_set == 0,]
df_full_test <- df_full[df_full$test_set == 1,]

nrow(df_full_train)  ## 62,200
nrow(df_full_test)   ## 9,360
0.9 * nrow(df_full)

# Result containers
return_time <- data.frame()
km_curve <- data.frame()
km_curve_grouped <- data.frame()
km_curve_return_grouped <- data.frame()
coef_cox <- data.frame()
pred_cox <- data.frame()


# Time to event analysis (type 2 and type 1)
for(out_col in c('days_to_return', 'days_to_return_type1')){
  
  if(out_col == 'days_to_return'){
    return_col <- 'fit_return'
    return_type <- 2
    } else {
    return_col <- 'fit_return_type1'
    return_type <- 1
  }
  
  # Analyse time to return for those who returned the test 
  t <- df_full_train[df_full_train[return_col] == 1,]
  t <- t[[out_col]]
  times <-  c(seq(1, 180, 1)) #, max(t))
  perc <- c()
  nret <- c()
  for(time in times){
    perc <- c(perc, mean(t <= time) * 100)
    nret <- c(nret, sum(t <= time))
  }
  df_perc <- data.frame(time=times, perc=perc, nret=nret)
  df_perc$return_type <- return_type
  df_perc$perc <- round(df_perc$perc, 2)
  return_time <- rbind(return_time, df_perc)
  
  test <- sum(t > 180)
  print(test)
  if(test < 10){
    break
  }
  
  # KM curve
  f <- paste('Surv(df_full_train$', out_col, ', df_full_train$', return_col, ') ~ 1', sep='')
  s <- survfit(as.formula(f), data = df_full_train)
  
  df_km <- data.frame(time=s$time, surv=s$surv, std_err=s$std.err, cumhaz=s$cumhaz,
                      std_chaz=s$std.chaz, surv_low=s$lower, surv_upp=s$upper,
                      n_event=s$n.event, n_risk=s$n.risk, n=s$n, n_censor=s$n.censor)
  
  # Compute conditional probabilities of return
  s14 <- df_km[df_km$time == 14, 'surv']
  s28 <- df_km[df_km$time == 28, 'surv']
  s70 <- df_km[df_km$time == 70, 'surv']
  smax <- df_km[which.max(df_km$time), 'surv']
  
  df_km$p14 <- 1 - s14 / df_km$surv
  df_km$p28 <- 1 - s28 / df_km$surv
  df_km$p70 <- 1 - s70 / df_km$surv
  df_km$pmax <- 1 - smax / df_km$surv
  
  df_km$p14[df_km$time > 14] <- NA
  df_km$p28[df_km$time > 28] <- NA
  df_km$p70[df_km$time > 70] <- NA
  df_km$return_type <- return_type
  
  # Sanity check of conditional probabilities
  print(df_km[df_km$time == 14, 'p28'])
  
  df_sub <- df_full_train[df_full_train$days_to_return > 14, 
                          c('days_to_return', 'fit_request_date_fu', 'fit_return')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return <= 28) & (df_sub$fit_return == 1) 
  mean(mask)
  mean(df_sub$fit_return)
  
  df_sub <- df_full_train[df_full_train$days_to_return_type1 > 14, 
                          c('days_to_return_type1', 'fit_request_date_fu', 'fit_return_type1')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return_type1 <= 28) & (df_sub$fit_return_type1 == 1) 
  mean(mask)
  mean(df_sub$fit_return_type1)
  
  sum(df_full_train$fit_return)
  mean(df_full_train$fit_return)
  
  sum(df_full_train$fit_return_type1)
  mean(df_full_train$fit_return_type1)
  
  df_ret <- df_full_train[df_full_train$fit_return_type1 == 1, c('days_to_return_type1', 'fit_request_date_fu')]
  sum(df_ret$days_to_return_type1 > 14)
  nrow(df_ret)
  
  # Sanity check of KM curves
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=p70), color='red')
  p <- p + geom_line(aes(x=time, y=p28), color='green')
  p <- p + geom_line(aes(x=time, y=p14), color='blue')
  p <- p + geom_line(aes(x=time, y=pmax), color='orange')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test within 70 days\nif not yet returned")
  p <- p + xlab("Days from FIT test request")
  p
  
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=surv), color='red')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test after t days")
  p <- p + xlab("Days from FIT test request")
  p
  
  km_curve <- rbind(km_curve, df_km)
  
  # KM curves by gender, age, ethnicity and deprivation
  df_km_grouped <- data.frame()
  for(var in c('gender_male', 'age_group', 'ethnicity', 'imd_quintile_factor')){
    values <- unique(df[[var]])
    for(val in values){
      print(paste(var, val))
      dfsub <- df_full_train[df_full_train[[var]] == val,]
      f <- paste('Surv(dfsub$', out_col, ', dfsub$', return_col, ') ~ 1', sep='')
      s <- survfit(as.formula(f), data = dfsub)
      df_km <- data.frame(time=s$time, surv=s$surv, std_err=s$std.err, cumhaz=s$cumhaz,
                          std_chaz=s$std.chaz, surv_low=s$lower, surv_upp=s$upper,
                          n_event=s$n.event, n_risk=s$n.risk, n=s$n, n_censor=s$n.censor)
      df_km$variable <- var
      df_km$value <- val
      df_km_grouped = rbind(df_km_grouped, df_km)
    }
  }
  df_km_grouped$st <- 1 - df_km_grouped$surv
  df_km_grouped$return_type <- return_type
  km_curve_grouped <- rbind(km_curve_grouped, df_km_grouped)
  
  p <- ggplot(data=df_km_grouped) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70) + facet_wrap(vars(variable))
  p <- ggplot(data=df_km_grouped[df_km_grouped$var=='age_group',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  p <- ggplot(data=df_km_grouped[df_km_grouped$var=='ethnicity',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  p <- ggplot(data=df_km_grouped[df_km_grouped$var=='imd_quintile_factor',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  
  # KM curves only for those who return test: by gender, age, ethnicity, and deprivation
  df_km_grouped_return <- data.frame()
  for(var in c('gender_male', 'age_group', 'ethnicity', 'imd_quintile_factor')){
    values <- unique(df[[var]])
    for(val in values){
      print(paste(var, val))
      dfsub <- df_full_train[df_full_train[[var]] == val,]
      dfsub <- dfsub[dfsub[[return_col]] == 1,]
      f <- paste('Surv(dfsub$', out_col, ', dfsub$', return_col, ') ~ 1', sep='')
      s <- survfit(as.formula(f), data = dfsub)
      df_km <- data.frame(time=s$time, surv=s$surv, std_err=s$std.err, cumhaz=s$cumhaz,
                          std_chaz=s$std.chaz, surv_low=s$lower, surv_upp=s$upper,
                          n_event=s$n.event, n_risk=s$n.risk, n=s$n, n_censor=s$n.censor)
      df_km$variable <- var
      df_km$value <- val
      df_km_grouped_return = rbind(df_km_grouped_return, df_km)
    }
  }
  df_km_grouped_return$st <- 1 - df_km_grouped_return$surv
  df_km_grouped_return$return_type <- return_type
  km_curve_return_grouped <- rbind(km_curve_return_grouped, df_km_grouped_return)
  
  p <- ggplot(data=df_km_grouped_return) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70) + facet_wrap(vars(variable))
  p <- ggplot(data=df_km_grouped_return[df_km_grouped_return$var=='age_group',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  p <- ggplot(data=df_km_grouped_return[df_km_grouped_return$var=='ethnicity',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  p <- ggplot(data=df_km_grouped_return[df_km_grouped_return$var=='imd_quintile_factor',]) + geom_line(aes(x=time, y=st, color=value)) + xlim(0, 70)
  
  
  # Cox model
  f <- paste('Surv(df_full_train$', out_col, ', df_full_train$', return_col, ') ~  1 + gender_male + age_group + ethnicity + imd_quintile_factor + request_year_factor + request_month_factor', sep='')
  fit_cox <- coxph(as.formula(f), data = df_full_train)
  
  tab_cox <- data.frame(coef(summary(fit_cox)))
  tab_cox$Predictor <- rownames(tab_cox)
  rownames(tab_cox) <- NULL
  tab_cox <- tab_cox[,colnames(tab_cox) != 'exp.coef.']
  tab_cox <- tab_cox[,c(5,1,2,3,4)]
  tab_cox$HR <- round(exp(tab_cox$coef),2)
  tab_cox$`HR low` <- round(exp(tab_cox$coef - 1.96 * tab_cox$se.coef.),2)
  tab_cox$`HR upp` <- round(exp(tab_cox$coef + 1.96 * tab_cox$se.coef.),2)
  tab_cox <- tab_cox[,c('Predictor', 'HR', 'HR low', 'HR upp', 'Pr...z..')]
  colnames(tab_cox)[5] <- 'P value'
  tab_cox$`P value sig` <- ''
  tab_cox$`P value sig`[tab_cox$`P value` < 0.05] <- '*'
  tab_cox$`P value sig`[tab_cox$`P value` < 0.01] <- '**'
  tab_cox$`P value sig`[tab_cox$`P value` < 0.001] <- '***'
  tab_cox$`P value` <- round(tab_cox$`P value`, 4)
  tab_cox$`P value`[tab_cox$`P value` == 0] <- '<0.001'
  tab_cox$`P value`[tab_cox$`P value` == 1e-4] <- '0.001'
  print(tab_cox)
  
  tab_cox$return_type <- return_type
  coef_cox <- rbind(coef_cox, tab_cox)
  
  # predict 14, 28 and 70 day return from cox model
  pred <- df_full[,c('patient_id', 'icen', 'test_set')]

  df_full_new <- df_full
  df_full_new$request_year_factor[df_full_new$request_year_factor == 2024] <- 2023
  lp <- predict(fit_cox, type='lp', newdata=df_full_new)
  hazard_ratio <- exp(lp)
  bhaz <- basehaz(fit_cox, centered=FALSE)
  bhaz$base_surv <- exp(-bhaz$hazard)
  
  for(days in c(14, 28, 70)){
    baseline_survival <- bhaz[bhaz$time == days,]$base_surv
    p_nonreturn <- baseline_survival ^ hazard_ratio
    pred$y_pred <- p_nonreturn
    outcome_col <- paste('nonret', return_type, '_days', days, sep='')
    pred$y_true <- df_full[[outcome_col]]
    pred$outcome <- outcome_col
    pred_cox <- rbind(pred_cox, pred)
  }
}
pred_cox <- pred_cox[!is.na(pred_cox$y_true),]
  
write.csv(coef_cox, file.path(out_path, 'cox-coef.csv'), row.names=FALSE)
write.csv(pred_cox, file.path(out_path, 'cox-pred.csv'), row.names=FALSE)
write.csv(return_time, file.path(out_path, 'return_time.csv'), row.names=FALSE)
write.csv(km_curve, file.path(out_path, 'km_curve.csv'), row.names=FALSE)
write.csv(km_curve_grouped, file.path(out_path, 'km_curve_grouped.csv'), row.names=FALSE)
write.csv(km_curve_return_grouped, file.path(out_path, 'km_curve_return_grouped.csv'), row.names=FALSE)


# Sanity check
p_logistic <- pred_logistic[,c('patient_id', 'icen', 'y_pred', 'outcome', 'test_set')]
p_cox <- pred_cox[,c('patient_id', 'icen', 'y_pred', 'outcome')]
colnames(p_cox)[3] <- 'y_pred_cox'
p_join <- inner_join(p_logistic, p_cox, by=c('patient_id', 'icen', 'outcome'))
p_sub <- p_join[p_join$outcome=='nonret1_days70',]
p_sub <- p_sub[p_sub$test_set == 0,]
plot(p_sub$y_pred, p_sub$y_pred_cox)


# ---- 5. Logistic models on imputed data ----
# This section imputes data using MICE before fitting logistic models
# This is not done by default, as data may not be missing at random.
if(run_on_imputed_data){
    
  coef_logistic_imp <- data.frame()
  
  # Impute ethnicity and imd quintile 
  #outcome_col <- "nonret2_days28"
  for(outcome_col in  c('nonret2_days7', 'nonret2_days14', 'nonret2_days28', 'nonret2_days70',
                        'nonret1_days7', 'nonret1_days14', 'nonret1_days28', 'nonret1_days70')){
    print(outcome_col)
    
    df_train_sub <- df_train[,c(outcome_col, 'gender_male', 'age_group', 'ethnicity', 'imd_quintile_factor', 'request_year_factor',
                                'request_month_factor')]
    df_train_sub$ethnicity[df_train_sub$ethnicity %in% c('Not stated', 'Not known')] <- NA
    df_train_sub$imd_quintile_factor[df_train_sub$imd_quintile_factor %in% c('Not known')] <- NA
    df_train_sub$imd_quintile_factor <- factor(df_train_sub$imd_quintile_factor, ordered=TRUE, levels=c("5", "4", "3", "2", "1"))
    mean(is.na(df_train_sub$imd_quintile_factor))  # 10% missing
    mean(is.na(df_train_sub$ethnicity))  # 22% missing
    
    
    df_train_imputed <- mice(df_train_sub, m=5, maxit=5) #, method=c("", "", "", "polyreg", "polr", "", ""))
    print(df_train_imputed$method)
    
    df_complete <- complete(df_train_imputed, action=1)
    
    table(df_complete$imd_quintile_factor) / nrow(df_complete) * 100
    table(df_train_sub$imd_quintile_factor) / sum(!is.na(df_train_sub$imd_quintile_factor)) * 100
    
    table(df_complete$ethnicity) / nrow(df_complete) * 100
    table(df_train_sub$ethnicity) / sum(!is.na(df_train_sub$ethnicity)) * 100
    
    
    f <- paste(outcome_col, " ~ gender_male + age_group + ethnicity + ",
               "relevel(factor(imd_quintile_factor, ordered=FALSE, levels=c('1', '2', '3', '4', '5')), ref='5')",
               "+ request_year_factor + request_month_factor",
               sep='')
    fit_logistic_imp <- with(data=df_train_imputed, 
                             glm(as.formula(f), family = binomial(link="logit")))
    
    fit_logistic_pool <- pool(fit_logistic_imp)
    summary(fit_logistic_pool)
    
    # Reformat the model summary
    tab_logistic <- summary(fit_logistic_pool)
    tab_logistic$OR <- round(exp(tab_logistic$estimate),2)
    tab_logistic$`OR low` <- round(exp(tab_logistic$estimate - 1.96 * tab_logistic$std.error),2)
    tab_logistic$`OR upp` <- round(exp(tab_logistic$estimate + 1.96 * tab_logistic$std.error),2)
    tab_logistic <- tab_logistic[,c('term', 'OR', 'OR low', 'OR upp', 'p.value')]
    colnames(tab_logistic)[5] <- 'P value'
    colnames(tab_logistic)[1] <- 'Predictor'
    
    tab_logistic$`P value sig` <- ''
    tab_logistic$`P value sig`[tab_logistic$`P value` < 0.05] <- '*'
    tab_logistic$`P value sig`[tab_logistic$`P value` < 0.01] <- '**'
    tab_logistic$`P value sig`[tab_logistic$`P value` < 0.001] <- '***'
    
    tab_logistic$`P value` <- round(tab_logistic$`P value`, 4)
    tab_logistic$`P value`[tab_logistic$`P value` == 0] <- '<0.001'
    tab_logistic$`P value`[tab_logistic$`P value` == 1e-4] <- '0.001'
    print(tab_logistic)
    
    pat <- 'relevel\\(factor\\(imd_quintile_factor, ordered = FALSE, levels = c\\("1", "2", "3", "4", "5"\\)\\), ref = "5"\\)'
    tab_logistic$Predictor <- gsub(pat, 'imd_quintile_factor', tab_logistic$Predictor)
    tab_logistic$outcome <- outcome_col
    
    coef_logistic_imp <- rbind(coef_logistic_imp, tab_logistic)
  }
  write.csv(coef_logistic_imp, file.path(out_path, 'logistic-coef-imp.csv'), row.names=FALSE)
}


# ---- 6. Logistic models with multiple requests per patient ----

# Read data
df <- read_csv("Z:/fit_nonreturn_paper_20250417/data/fit_nonret.csv")
nrow(df)

# IMD to factor
df$imd_quintile_factor <- df$imd_quintile
df$imd_quintile_factor[is.na(df$imd_quintile)] <- 'Not known'
df$imd_quintile_factor <- factor(df$imd_quintile_factor)
df$imd_quintile_factor <- relevel(df$imd_quintile_factor, ref='5')

# Set White as reference
df$ethnicity <- factor(df$ethnicity)
df$ethnicity <- relevel(df$ethnicity, ref='White')

# Set youngest age group as ref
df$age_group <- factor(df$age_group)
df$age_group <- relevel(df$age_group, ref='18-39')

# Time to factor
df$request_year_factor <- factor(df$request_year)
df$request_year_factor <- relevel(df$request_year_factor, ref='2017')
df$request_month_factor <- factor(df$request_month)
df$request_month_factor <- relevel(df$request_month_factor, ref='1')

# Use time with daily precision (more precision not needed)
df$days_to_return <- round(df$days_to_return, digits=0)
df$days_to_return_type1 <- round(df$days_to_return_type1, digits=0)

# Drop patients without 70-day follow-up
nrow(df)
df <- df[df$fit_request_date_fu >= 70,]
nrow(df)
sum(is.na(df$nonret2_days70))
sum(is.na(df$nonret2_days14))

# Drop requests where patients died before 70-day follow-up without returning a test (162)
mask <- (df$censored == 1) & (!is.na(df$days_to_death)) & (df$days_to_death <= 70)
sum(mask)
nrow(df)
df <- df[!mask,]
nrow(df)

# Train test split
df_train <- df[df$test_set == 0,]
df_test <- df[df$test_set == 1,]
df_train <- df_train[,colnames(df_train) != 'test_set']

# Dbl check that the few >365 returns are dropped
max(df[df$fit_return_type1 == 1,]$days_to_return_type1)

# Fit models
coef_logistic_mult <- data.frame()
pred_logistic_mult <- data.frame()

for(outcome_col in c('nonret2_days7', 'nonret2_days14', 'nonret2_days28', 'nonret2_days70',
                     'nonret1_days7', 'nonret1_days14', 'nonret1_days28', 'nonret1_days70')){
  print(outcome_col)
  
  # Fit simplest logistic model
  formula = paste(outcome_col, "~ gender_male + age_group + ethnicity + imd_quintile_factor + request_year_factor + request_month_factor")
  fit_logistic <- glm(as.formula(formula), data=df_train, family = binomial(link="logit"))
  summary(fit_logistic)
  
  # Reformat the model summary
  tab_logistic <- data.frame(coef(summary(fit_logistic)))
  tab_logistic$Predictor <- rownames(tab_logistic)
  tab_logistic <- tab_logistic[,c(5, 1,2,3,4)]
  rownames(tab_logistic) <- NULL
  tab_logistic$OR <- round(exp(tab_logistic$Estimate),2)
  tab_logistic$`OR low` <- round(exp(tab_logistic$Estimate - 1.96 * tab_logistic$`Std..Error`),2)
  tab_logistic$`OR upp` <- round(exp(tab_logistic$Estimate + 1.96 * tab_logistic$`Std..Error`),2)
  tab_logistic <- tab_logistic[,c('Predictor', 'OR', 'OR low', 'OR upp', 'Pr...z..')]
  colnames(tab_logistic)[5] <- 'P value'
  
  tab_logistic$`P value sig` <- ''
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.05] <- '*'
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.01] <- '**'
  tab_logistic$`P value sig`[tab_logistic$`P value` < 0.001] <- '***'
  
  tab_logistic$`P value` <- round(tab_logistic$`P value`, 4)
  tab_logistic$`P value`[tab_logistic$`P value` == 0] <- '<0.001'
  tab_logistic$`P value`[tab_logistic$`P value` == 1e-4] <- '0.001'
  print(tab_logistic)
  
  tab_logistic$outcome <- outcome_col
  coef_logistic_mult <- rbind(coef_logistic_mult, tab_logistic)
  
  # Create predictions on the held out test set
  # What to do with year 2024 not in factor level?
  df_new <- df
  df_new$request_year_factor[df_new$request_year_factor == 2024] <- 2023
  pred <- predict(fit_logistic, newdata=df_new, type='response')
  pred <- data.frame(patient_id=df$patient_id, 
                     icen=df$icen, 
                     y_pred=pred, 
                     y_true=df[[outcome_col]],
                     outcome=outcome_col,
                     test_set=df$test_set)
  pred_logistic_mult <- rbind(pred_logistic_mult, pred)
  
}

## Save
write.csv(coef_logistic_mult, file.path(out_path, 'logistic-coef-mult.csv'), row.names=FALSE)
write.csv(pred_logistic_mult, file.path(out_path, 'logistic-pred-mult.csv'), row.names=FALSE)


# ---- 7. KM curves and conditional probability curves without train-test split ----
# These are included in the publication. The train-test split is only for evaluating performance of models

# Result containers
return_time <- data.frame()
km_curve <- data.frame()

# KM curves and return times
for(out_col in c('days_to_return', 'days_to_return_type1')){
  
  if(out_col == 'days_to_return'){
    return_col <- 'fit_return'
    return_type <- 2
  } else {
    return_col <- 'fit_return_type1'
    return_type <- 1
  }
  
  # Analyse time to return for those who returned the test 
  # (Is this meaningful given variable length of follow-up?
  #  Or should you choose, say, 70-day fu, and then compute 
  #  time to return only up to 70 days? Yes.)
  #t <- df_full[df_full[return_col] == 1,]
  t <- df[df[return_col] == 1,]
  t <- t[[out_col]]
  #times <-  c(seq(1, 180, 1)) #, max(t))
  times <- c(seq(1, 70, 1))
  perc <- c()
  nret <- c()
  for(time in times){
    perc <- c(perc, mean(t <= time) * 100)
    nret <- c(nret, sum(t <= time))
  }
  df_perc <- data.frame(time=times, perc=perc, nret=nret)
  df_perc$return_type <- return_type
  df_perc$perc <- round(df_perc$perc, 2)
  return_time <- rbind(return_time, df_perc)
  
  test <- sum(t > 180)
  print(test)
  if(test < 10){
    break
  }
  
  # KM curve
  # Note: the conditional probabilities seem intuitively too high when using max time period as reference
  f <- paste('Surv(df_full$', out_col, ', df_full$', return_col, ') ~ 1', sep='')
  s <- survfit(as.formula(f), data = df_full)
  
  df_km <- data.frame(time=s$time, surv=s$surv, std_err=s$std.err, cumhaz=s$cumhaz,
                      std_chaz=s$std.chaz, surv_low=s$lower, surv_upp=s$upper,
                      n_event=s$n.event, n_risk=s$n.risk, n=s$n, n_censor=s$n.censor)
  
  # Compute conditional probabilities of return
  s14 <- df_km[df_km$time == 14, 'surv']
  s28 <- df_km[df_km$time == 28, 'surv']
  s70 <- df_km[df_km$time == 70, 'surv']
  smax <- df_km[which.max(df_km$time), 'surv']
  
  df_km$p14 <- 1 - s14 / df_km$surv
  df_km$p28 <- 1 - s28 / df_km$surv
  df_km$p70 <- 1 - s70 / df_km$surv
  df_km$pmax <- 1 - smax / df_km$surv
  
  df_km$p14[df_km$time > 14] <- NA
  df_km$p28[df_km$time > 28] <- NA
  df_km$p70[df_km$time > 70] <- NA
  df_km$return_type <- return_type
  
  # Sanity check of conditional probabilities
  print(df_km[df_km$time == 14, 'p28'])
  
  df_sub <- df_full[df_full$days_to_return > 14, c('days_to_return', 'fit_request_date_fu', 'fit_return')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return <= 28) & (df_sub$fit_return == 1) 
  mean(mask)
  mean(df_sub$fit_return)
  
  df_sub <- df_full[df_full$days_to_return_type1 > 14, c('days_to_return_type1', 'fit_request_date_fu', 'fit_return_type1')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return_type1 <= 28) & (df_sub$fit_return_type1 == 1) 
  mean(mask)
  mean(df_sub$fit_return_type1)
  
  # Sanity check of KM curves
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=p70), color='red')
  p <- p + geom_line(aes(x=time, y=p28), color='green')
  p <- p + geom_line(aes(x=time, y=p14), color='blue')
  p <- p + geom_line(aes(x=time, y=pmax), color='orange')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test within 70 days\nif not yet returned")
  p <- p + xlab("Days from FIT test request")
  p
  
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=surv), color='red')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test after t days")
  p <- p + xlab("Days from FIT test request")
  p
  
  km_curve <- rbind(km_curve, df_km)
}
  
write.csv(return_time, file.path(out_path, 'return_time_data-full_fu-70.csv'), row.names=FALSE)
write.csv(km_curve, file.path(out_path, 'km_curve_data-full.csv'), row.names=FALSE)



# .... Also create KM curves for test set, to see how different they may be due to changes over time

df$fit_request_year <- substr(df$fit_request_date_corrected, start = 1, stop = 4)
df_full$fit_request_year <- substr(df_full$fit_request_date_corrected, start = 1, stop = 4)

df <- df[df$fit_request_year %in% c("2023", "2024"),]
df_full <- df_full[df_full$fit_request_year %in% c("2023", "2024"),]

# Result containers
return_time <- data.frame()
km_curve <- data.frame()

# Time to event analysis (type 2 and type 1)
for(out_col in c('days_to_return', 'days_to_return_type1')){
  
  if(out_col == 'days_to_return'){
    return_col <- 'fit_return'
    return_type <- 2
  } else {
    return_col <- 'fit_return_type1'
    return_type <- 1
  }
  
  # Analyse time to return for those who returned the test 
  # (Is this meaningful given variable length of follow-up?
  #  Or should you choose, say, 70-day fu, and then compute 
  #  time to return only up to 70 days? Yes.)
  #t <- df_full[df_full[return_col] == 1,]
  t <- df[df[return_col] == 1,]
  t <- t[[out_col]]
  #times <-  c(seq(1, 180, 1)) #, max(t))
  times <- c(seq(1, 70, 1))
  perc <- c()
  nret <- c()
  for(time in times){
    perc <- c(perc, mean(t <= time) * 100)
    nret <- c(nret, sum(t <= time))
  }
  df_perc <- data.frame(time=times, perc=perc, nret=nret)
  df_perc$return_type <- return_type
  df_perc$perc <- round(df_perc$perc, 2)
  return_time <- rbind(return_time, df_perc)
  
  test <- sum(t > 180)
  print(test)
  if(test < 10){
    break
  }
  
  # KM curve
  # Note: the conditional probabilities seem intuitively too high when using max time period as reference
  f <- paste('Surv(df_full$', out_col, ', df_full$', return_col, ') ~ 1', sep='')
  s <- survfit(as.formula(f), data = df_full)
  
  df_km <- data.frame(time=s$time, surv=s$surv, std_err=s$std.err, cumhaz=s$cumhaz,
                      std_chaz=s$std.chaz, surv_low=s$lower, surv_upp=s$upper,
                      n_event=s$n.event, n_risk=s$n.risk, n=s$n, n_censor=s$n.censor)
  
  # Compute conditional probabilities of return
  s14 <- df_km[df_km$time == 14, 'surv']
  s28 <- df_km[df_km$time == 28, 'surv']
  s70 <- df_km[df_km$time == 70, 'surv']
  smax <- df_km[which.max(df_km$time), 'surv']
  
  df_km$p14 <- 1 - s14 / df_km$surv
  df_km$p28 <- 1 - s28 / df_km$surv
  df_km$p70 <- 1 - s70 / df_km$surv
  df_km$pmax <- 1 - smax / df_km$surv
  
  df_km$p14[df_km$time > 14] <- NA
  df_km$p28[df_km$time > 28] <- NA
  df_km$p70[df_km$time > 70] <- NA
  df_km$return_type <- return_type
  
  # Sanity check of conditional probabilities
  print(df_km[df_km$time == 14, 'p28'])
  
  df_sub <- df_full[df_full$days_to_return > 14, c('days_to_return', 'fit_request_date_fu', 'fit_return')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return <= 28) & (df_sub$fit_return == 1) 
  mean(mask)
  mean(df_sub$fit_return)
  
  df_sub <- df_full[df_full$days_to_return_type1 > 14, c('days_to_return_type1', 'fit_request_date_fu', 'fit_return_type1')]
  df_sub <- df_sub[df_sub$fit_request_date_fu >= 28,]
  print(nrow(df_sub))
  mask <- (df_sub$days_to_return_type1 <= 28) & (df_sub$fit_return_type1 == 1) 
  mean(mask)
  mean(df_sub$fit_return_type1)
  
  # Sanity check of KM curves
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=p70), color='red')
  p <- p + geom_line(aes(x=time, y=p28), color='green')
  p <- p + geom_line(aes(x=time, y=p14), color='blue')
  p <- p + geom_line(aes(x=time, y=pmax), color='orange')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test within 70 days\nif not yet returned")
  p <- p + xlab("Days from FIT test request")
  p
  
  p <- ggplot(data=df_km)
  p <- p + geom_line(aes(x=time, y=surv), color='red')
  p <- p + xlim(0, 70)
  p <- p + scale_y_continuous(breaks = seq(0, 1, 0.1), limits=c(0, 1))
  p <- p + ylab("Probability of returning the test after t days")
  p <- p + xlab("Days from FIT test request")
  p
  
  km_curve <- rbind(km_curve, df_km)
}

write.csv(return_time, file.path(out_path, 'return_time_data-full_fu-70_years-23-24.csv'), row.names=FALSE)
write.csv(km_curve, file.path(out_path, 'km_curve_data-full_years-23-24.csv'), row.names=FALSE)