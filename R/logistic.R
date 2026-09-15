# ==============================================================================
# File: R/logistic.R
# Purpose: Binary Logistic Regression (versicolor vs virginica),
#          interactive decision thresholding, confusion matrix, ROC/sensitivity.
# Author: Reconstructed for Shivansh Gautam (Infosys Springboard ML in R)
# ==============================================================================

source("R/data.R")
source("R/metrics.R")

#' Fit Logistic Regression Model on Versicolor vs Virginica
#'
#' Evaluates on the 100 observations of the binary subset.
#'
#' @return list containing model, summary, probabilities, coefficients, odds ratios
fit_logistic_model <- function() {
  binary_df <- get_binary_iris()
  model <- glm(Species ~ Petal.Length + Petal.Width, family = binomial, data = binary_df)
  s <- summary(model)
  probs <- predict(model, type = "response")
  
  coef_df <- data.frame(
    Term = rownames(s$coefficients),
    Estimate_Log_Odds = s$coefficients[, "Estimate"],
    Std_Error = s$coefficients[, "Std. Error"],
    z_value = s$coefficients[, "z value"],
    p_value = s$coefficients[, "Pr(>|z|)"],
    Odds_Ratio = exp(s$coefficients[, "Estimate"]),
    stringsAsFactors = FALSE
  )
  
  list(
    model = model,
    summary = s,
    probabilities = probs,
    data = binary_df,
    coefficients = coef_df
  )
}

#' Evaluate Logistic Classification at a Specific Threshold
#'
#' @param threshold numeric decision threshold between 0.01 and 0.99 (default: 0.5)
#' @return list containing binary classification metrics and confusion matrix
evaluate_logistic_threshold <- function(threshold = 0.5) {
  fit <- fit_logistic_model()
  binary_df <- fit$data
  probs <- fit$probabilities
  
  predicted <- ifelse(probs >= threshold, "virginica", "versicolor")
  predicted <- factor(predicted, levels = c("versicolor", "virginica"))
  
  metrics <- calc_binary_metrics(binary_df$Species, predicted, positive_class = "virginica")
  metrics$threshold <- threshold
  metrics$probabilities <- probs
  metrics$actual <- binary_df$Species
  metrics$predicted <- predicted
  
  metrics
}

#' Generate Threshold Sensitivity Curves (0.05 to 0.95)
#'
#' @return data.frame with Threshold, Accuracy, Precision, Recall, F1
get_threshold_sensitivity_curve <- function() {
  thresholds <- seq(0.1, 0.9, by = 0.02)
  curve_list <- lapply(thresholds, function(th) {
    m <- evaluate_logistic_threshold(th)
    data.frame(
      Threshold = th,
      Accuracy = m$accuracy,
      Precision = m$precision,
      Recall = m$recall,
      F1_Score = m$f1,
      TN = m$tn,
      FP = m$fp,
      FN = m$fn,
      TP = m$tp
    )
  })
  do.call(rbind, curve_list)
}

#' Get Report Benchmark Threshold Performance
#'
#' Compares computed metrics against the report documented values at 0.3, 0.5, 0.7.
#'
#' @return data.frame
get_reported_threshold_benchmarks <- function() {
  targets <- c(0.3, 0.5, 0.7)
  res <- lapply(targets, function(th) {
    m <- evaluate_logistic_threshold(th)
    data.frame(
      Threshold = th,
      Computed_Accuracy = m$accuracy,
      Reported_Accuracy = ifelse(th == 0.3, 0.9600, ifelse(th == 0.5, 0.9400, 0.9500)),
      Computed_F1 = m$f1,
      TN = m$tn,
      FP = m$fp,
      FN = m$fn,
      TP = m$tp,
      Match = abs(m$accuracy - ifelse(th == 0.3, 0.9600, ifelse(th == 0.5, 0.9400, 0.9500))) < 1e-4
    )
  })
  do.call(rbind, res)
}
