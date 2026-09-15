# ==============================================================================
# File: R/validation.R
# Purpose: Comprehensive validation suite checking live computed R models
#          against the documented benchmarks in the Industrial Training report.
# Author: Reconstructed for Shivansh Gautam (Infosys Springboard ML in R)
# ==============================================================================

source("R/data.R")
source("R/metrics.R")
source("R/regression.R")
source("R/multicollinearity.R")
source("R/logistic.R")
source("R/decision_tree.R")
source("R/knn.R")

#' Run Full Pipeline Validation against Documented Targets
#'
#' @param tolerance numeric tolerance for floating-point comparisons (default: 0.005)
#' @return data.frame containing Model, Metric, Target, Computed, Difference, Status
validate_all_results <- function(tolerance = 0.005) {
  checks <- list()
  
  add_check <- function(model_name, metric_name, target, computed, tol = tolerance) {
    diff_val <- abs(computed - target)
    passed <- diff_val <= tol
    data.frame(
      Model = model_name,
      Metric = metric_name,
      Report_Target = target,
      Computed_Value = computed,
      Absolute_Diff = diff_val,
      Tolerance = tol,
      Status = ifelse(passed, "PASS", "FLAG"),
      stringsAsFactors = FALSE
    )
  }
  
  # 1. Simple Linear Regression
  m_simple <- fit_simple_regression()
  checks[[length(checks) + 1]] <- add_check("Simple Linear Regression", "R²", 0.9271, m_simple$metrics$r_squared)
  checks[[length(checks) + 1]] <- add_check("Simple Linear Regression", "Adjusted R²", 0.9266, m_simple$metrics$adj_r_squared)
  checks[[length(checks) + 1]] <- add_check("Simple Linear Regression", "RMSE (cm)", 0.4750, m_simple$metrics$rmse)
  checks[[length(checks) + 1]] <- add_check("Simple Linear Regression", "Slope (Petal.Width)", 2.2299, coef(m_simple$model)["Petal.Width"])
  
  # 2. Multiple Linear Regression
  m_mult <- fit_multiple_regression()
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "R²", 0.9680, m_mult$metrics$r_squared)
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "Adjusted R²", 0.9674, m_mult$metrics$adj_r_squared)
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "RMSE (cm)", 0.3147, m_mult$metrics$rmse)
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "Coef: Petal.Width", 1.4468, coef(m_mult$model)["Petal.Width"])
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "Coef: Sepal.Length", 0.7291, coef(m_mult$model)["Sepal.Length"])
  checks[[length(checks) + 1]] <- add_check("Multiple Linear Regression", "Coef: Sepal.Width", -0.6460, coef(m_mult$model)["Sepal.Width"])
  
  # 3. Multicollinearity / VIF
  vif_data <- calc_vif()
  pw_vif <- vif_data$VIF[vif_data$Predictor == "Petal.Width"]
  sl_vif <- vif_data$VIF[vif_data$Predictor == "Sepal.Length"]
  sw_vif <- vif_data$VIF[vif_data$Predictor == "Sepal.Width"]
  checks[[length(checks) + 1]] <- add_check("Multicollinearity", "VIF: Petal.Width", 3.890, pw_vif)
  checks[[length(checks) + 1]] <- add_check("Multicollinearity", "VIF: Sepal.Length", 3.416, sl_vif)
  checks[[length(checks) + 1]] <- add_check("Multicollinearity", "VIF: Sepal.Width", 1.306, sw_vif)
  
  # 4. Logistic Regression
  log_05 <- evaluate_logistic_threshold(0.5)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.5)", "Accuracy", 0.9400, log_05$accuracy)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.5)", "F1-Score", 0.9400, log_05$f1)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.5)", "TN Count", 47, log_05$tn, tol = 0)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.5)", "TP Count", 47, log_05$tp, tol = 0)
  
  log_03 <- evaluate_logistic_threshold(0.3)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.3)", "Accuracy", 0.9600, log_03$accuracy)
  
  log_07 <- evaluate_logistic_threshold(0.7)
  checks[[length(checks) + 1]] <- add_check("Logistic (th=0.7)", "Accuracy", 0.9500, log_07$accuracy)
  
  # 5. Decision Tree
  split_data <- get_stratified_split()
  tree_d3 <- fit_decision_tree(max_depth = 3, split_data = split_data)
  checks[[length(checks) + 1]] <- add_check("Decision Tree (Depth 3)", "Test Accuracy", 0.9778, tree_d3$test_metrics$accuracy)
  checks[[length(checks) + 1]] <- add_check("Decision Tree (Depth 3)", "Test Macro F1", 0.9778, tree_d3$test_metrics$macro_f1)
  
  tree_unr <- fit_decision_tree(max_depth = NULL, split_data = split_data)
  checks[[length(checks) + 1]] <- add_check("Decision Tree (Unrestricted)", "Train Accuracy", 1.0000, tree_unr$train_metrics$accuracy)
  checks[[length(checks) + 1]] <- add_check("Decision Tree (Unrestricted)", "Test Accuracy", 0.9333, tree_unr$test_metrics$accuracy)
  
  # 6. KNN
  knn_raw <- run_knn_classifier(k = 5, normalized = FALSE, split_data = split_data)
  checks[[length(checks) + 1]] <- add_check("KNN (Raw, k=5)", "Test Accuracy", 0.9778, knn_raw$metrics$accuracy)
  
  knn_norm <- run_knn_classifier(k = 5, normalized = TRUE, split_data = split_data)
  checks[[length(checks) + 1]] <- add_check("KNN (Normalised, k=5)", "Test Accuracy", 0.9333, knn_norm$metrics$accuracy)
  checks[[length(checks) + 1]] <- add_check("KNN (Normalised, k=5)", "Test Macro F1", 0.9327, knn_norm$metrics$macro_f1, tol = 0.005)
  
  do.call(rbind, checks)
}
