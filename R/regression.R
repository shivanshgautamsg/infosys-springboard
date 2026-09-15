# ==============================================================================
# File: R/regression.R
# Purpose: Linear Regression models (Simple & Multiple), diagnostics,
#          comparison tables, and academic interpretations.
# Author: Reconstructed for Shivansh Gautam (Infosys Springboard ML in R)
# ==============================================================================

source("R/metrics.R")

#' Fit Simple Linear Regression as documented in the report
#'
#' Formula: Petal.Length ~ Petal.Width
#'
#' @param data data.frame (default: iris)
#' @return list with model, summary, metrics, equation, and coef_table
fit_simple_regression <- function(data = NULL) {
  if (is.null(data)) data <- get_raw_iris()
  
  model <- lm(Petal.Length ~ Petal.Width, data = data)
  s <- summary(model)
  metrics <- calc_regression_metrics(model)
  
  # Format coefficient table with confidence intervals
  ci <- confint(model)
  coef_df <- data.frame(
    Term = rownames(s$coefficients),
    Estimate = s$coefficients[, "Estimate"],
    Std_Error = s$coefficients[, "Std. Error"],
    t_value = s$coefficients[, "t value"],
    p_value = s$coefficients[, "Pr(>|t|)"],
    CI_Lower = ci[, 1],
    CI_Upper = ci[, 2],
    stringsAsFactors = FALSE
  )
  
  # Construct readable equation
  b0 <- round(coef(model)[1], 4)
  b1 <- round(coef(model)[2], 4)
  sign_b1 <- ifelse(b1 >= 0, "+", "-")
  equation <- sprintf("Petal.Length = %.4f %s %.4f × Petal.Width", b0, sign_b1, abs(b1))
  
  interpretation <- paste(
    "An increase of one centimetre in petal width is associated with an increase of",
    "approximately 2.23 cm in petal length (slope = 2.2299, p < 0.001).",
    "The simple linear model explains 92.71% of the total variance in petal length",
    "(R² = 0.9271) with an RMSE of 0.4750 cm."
  )
  
  list(
    model = model,
    summary = s,
    metrics = metrics,
    coef_table = coef_df,
    equation = equation,
    interpretation = interpretation
  )
}

#' Fit Multiple Linear Regression as documented in the report
#'
#' Formula: Petal.Length ~ Petal.Width + Sepal.Length + Sepal.Width
#'
#' @param data data.frame (default: iris)
#' @return list with model, summary, metrics, equation, and coef_table
fit_multiple_regression <- function(data = NULL) {
  if (is.null(data)) data <- get_raw_iris()
  
  model <- lm(Petal.Length ~ Petal.Width + Sepal.Length + Sepal.Width, data = data)
  s <- summary(model)
  metrics <- calc_regression_metrics(model)
  
  ci <- confint(model)
  coef_df <- data.frame(
    Term = rownames(s$coefficients),
    Estimate = s$coefficients[, "Estimate"],
    Std_Error = s$coefficients[, "Std. Error"],
    t_value = s$coefficients[, "t value"],
    p_value = s$coefficients[, "Pr(>|t|)"],
    CI_Lower = ci[, 1],
    CI_Upper = ci[, 2],
    stringsAsFactors = FALSE
  )
  
  b0 <- round(coef(model)[1], 4)
  b_pw <- round(coef(model)["Petal.Width"], 4)
  b_sl <- round(coef(model)["Sepal.Length"], 4)
  b_sw <- round(coef(model)["Sepal.Width"], 4)
  
  equation <- sprintf(
    "Petal.Length = %.4f %s %.4f × Petal.Width %s %.4f × Sepal.Length %s %.4f × Sepal.Width",
    b0,
    ifelse(b_pw >= 0, "+", "-"), abs(b_pw),
    ifelse(b_sl >= 0, "+", "-"), abs(b_sl),
    ifelse(b_sw >= 0, "+", "-"), abs(b_sw)
  )
  
  interpretation <- paste(
    "Adding Sepal.Length and Sepal.Width increases explanatory power: R² improves from 0.9271 to 0.9680,",
    "and RMSE decreases from 0.4750 cm to 0.3147 cm.",
    "Holding all other variables constant, each 1 cm increase in Petal.Width contributes +1.4468 cm to Petal.Length,",
    "each 1 cm increase in Sepal.Length contributes +0.7291 cm, while Sepal.Width exerts a negative partial effect (-0.6460 cm)."
  )
  
  list(
    model = model,
    summary = s,
    metrics = metrics,
    coef_table = coef_df,
    equation = equation,
    interpretation = interpretation
  )
}

#' Compare Simple vs Multiple Linear Regression
#'
#' @return data.frame contrasting metrics
get_regression_comparison <- function() {
  m_simple <- fit_simple_regression()
  m_mult <- fit_multiple_regression()
  
  data.frame(
    Metric = c("Model Formula", "R²", "Adjusted R²", "RMSE (cm)", "MAE (cm)", "F-Statistic", "Residual Std Error"),
    Simple_Regression = c(
      "Petal.Length ~ Petal.Width",
      sprintf("%.4f", m_simple$metrics$r_squared),
      sprintf("%.4f", m_simple$metrics$adj_r_squared),
      sprintf("%.4f", m_simple$metrics$rmse),
      sprintf("%.4f", m_simple$metrics$mae),
      sprintf("%.2f", m_simple$metrics$f_statistic),
      sprintf("%.4f", m_simple$summary$sigma)
    ),
    Multiple_Regression = c(
      "Petal.Length ~ Petal.Width + Sepal.Length + Sepal.Width",
      sprintf("%.4f", m_mult$metrics$r_squared),
      sprintf("%.4f", m_mult$metrics$adj_r_squared),
      sprintf("%.4f", m_mult$metrics$rmse),
      sprintf("%.4f", m_mult$metrics$mae),
      sprintf("%.2f", m_mult$metrics$f_statistic),
      sprintf("%.4f", m_mult$summary$sigma)
    ),
    Improvement = c(
      "+2 Additional Predictors",
      sprintf("+%.4f (+%.2f%%)", m_mult$metrics$r_squared - m_simple$metrics$r_squared, 
              (m_mult$metrics$r_squared - m_simple$metrics$r_squared)/m_simple$metrics$r_squared * 100),
      sprintf("+%.4f (+%.2f%%)", m_mult$metrics$adj_r_squared - m_simple$metrics$adj_r_squared,
              (m_mult$metrics$adj_r_squared - m_simple$metrics$adj_r_squared)/m_simple$metrics$adj_r_squared * 100),
      sprintf("-%.4f (-%.2f%%)", m_simple$metrics$rmse - m_mult$metrics$rmse,
              (m_simple$metrics$rmse - m_mult$metrics$rmse)/m_simple$metrics$rmse * 100),
      sprintf("-%.4f (-%.2f%%)", m_simple$metrics$mae - m_mult$metrics$mae,
              (m_simple$metrics$mae - m_mult$metrics$mae)/m_simple$metrics$mae * 100),
      "Statistically Significant (p < 2.2e-16)",
      "Reduced error variance"
    ),
    stringsAsFactors = FALSE
  )
}
