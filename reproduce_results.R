# ==============================================================================
# Script: reproduce_results.R
# Purpose: Standalone execution script running the complete ML pipeline
#          from a clean R session and exporting results to CSV and JSON.
# Author: Shivansh Gautam (Infosys Springboard ML in R Demonstration)
# ==============================================================================

cat("======================================================================\n")
cat(" INDUSTRIAL TRAINING REPRODUCIBILITY PIPELINE\n")
cat(" 'Machine Learning Fundamentals: A Study of Predictive Modelling'\n")
cat(" Author: Shivansh Gautam | Infosys Springboard ML using R\n")
cat("======================================================================\n\n")

suppressPackageStartupMessages({
  library(rpart)
  library(class)
  library(jsonlite)
})

# Source modular components
source("R/data.R")
source("R/metrics.R")
source("R/regression.R")
source("R/multicollinearity.R")
source("R/logistic.R")
source("R/decision_tree.R")
source("R/knn.R")
source("R/validation.R")

dir.create("results", showWarnings = FALSE)

cat("[1/6] Running Simple & Multiple Linear Regression...\n")
simple_res <- fit_simple_regression()
mult_res <- fit_multiple_regression()
reg_comp <- get_regression_comparison()

cat(sprintf("   - Simple:   R² = %.4f, RMSE = %.4f cm\n", simple_res$metrics$r_squared, simple_res$metrics$rmse))
cat(sprintf("   - Multiple: R² = %.4f, RMSE = %.4f cm\n", mult_res$metrics$r_squared, mult_res$metrics$rmse))

cat("[2/6] Running Multicollinearity & VIF Analysis...\n")
vif_table <- calc_vif()
coef_shift <- get_coefficient_shift_analysis()
for (i in 1:nrow(vif_table)) {
  cat(sprintf("   - Predictor %-13s : VIF = %.3f\n", vif_table$Predictor[i], vif_table$VIF[i]))
}

cat("[3/6] Running Logistic Regression & Threshold Sensitivity...\n")
log_05 <- evaluate_logistic_threshold(0.5)
log_benchmarks <- get_reported_threshold_benchmarks()
cat(sprintf("   - Threshold 0.5: Accuracy = %.4f, F1 = %.4f (TN=%d, TP=%d, FP=%d, FN=%d)\n",
            log_05$accuracy, log_05$f1, log_05$tn, log_05$tp, log_05$fp, log_05$fn))

cat("[4/6] Running Decision Tree & Overfitting Experiment...\n")
split_data <- get_stratified_split()
tree_d3 <- fit_decision_tree(3, split_data)
tree_unr <- fit_decision_tree(NULL, split_data)
overfit_table <- run_tree_overfitting_experiment(split_data)
cat(sprintf("   - Depth 3:       Test Acc = %.4f, Macro F1 = %.4f (Leaves = %d)\n",
            tree_d3$test_metrics$accuracy, tree_d3$test_metrics$macro_f1, tree_d3$leaf_count))
cat(sprintf("   - Unrestricted:  Train Acc = %.4f, Test Acc = %.4f (Overfitting Divergence = %.4f)\n",
            tree_unr$train_metrics$accuracy, tree_unr$test_metrics$accuracy,
            tree_unr$train_metrics$accuracy - tree_unr$test_metrics$accuracy))

cat("[5/6] Running K-Nearest Neighbours (Raw vs Normalised)...\n")
knn_raw <- run_knn_classifier(k = 5, normalized = FALSE, split_data = split_data)
knn_norm <- run_knn_classifier(k = 5, normalized = TRUE, split_data = split_data)
knn_comp <- get_raw_vs_normalized_comparison(split_data)
cat(sprintf("   - Raw KNN (k=5):        Accuracy = %.4f, Macro F1 = %.4f\n",
            knn_raw$metrics$accuracy, knn_raw$metrics$macro_f1))
cat(sprintf("   - Normalised KNN (k=5): Accuracy = %.4f, Macro F1 = %.4f\n",
            knn_norm$metrics$accuracy, knn_norm$metrics$macro_f1))

cat("[6/6] Compiling Master Comparison Table & Validation Checks...\n")
validation_table <- validate_all_results()

# Master comparison summary
model_comp_table <- data.frame(
  Model = c("Logistic Regression", "Decision Tree (Depth 3)", "Decision Tree (Unrestricted)", "KNN (k=5, Raw)", "KNN (k=5, Normalised)"),
  Task = c("Binary: versicolor vs virginica", "Three-class Iris", "Three-class Iris", "Three-class Iris", "Three-class Iris"),
  R_Function = c("glm(family = binomial)", "rpart(maxdepth = 3)", "rpart(unrestricted)", "class::knn(raw)", "class::knn(min-max)"),
  Accuracy = c(log_05$accuracy, tree_d3$test_metrics$accuracy, tree_unr$test_metrics$accuracy, knn_raw$metrics$accuracy, knn_norm$metrics$accuracy),
  F_Score = c(log_05$f1, tree_d3$test_metrics$macro_f1, tree_unr$test_metrics$macro_f1, knn_raw$metrics$macro_f1, knn_norm$metrics$macro_f1),
  Reported_Accuracy = c(0.9400, 0.9778, 0.9333, 0.9778, 0.9333),
  stringsAsFactors = FALSE
)

# Export to CSV
write.csv(validation_table, "results/results.csv", row.names = FALSE)
write.csv(model_comp_table, "results/model_comparison.csv", row.names = FALSE)

# Export to JSON
results_json <- list(
  metadata = list(
    author = "Shivansh Gautam",
    training = "Infosys Springboard - Explore Machine Learning using R",
    timestamp = as.character(Sys.time()),
    r_version = R.version.string,
    seed_configured = 1606
  ),
  simple_regression = simple_res$metrics,
  multiple_regression = mult_res$metrics,
  vif = vif_table,
  logistic_regression = list(
    threshold_0.5 = log_05[c("accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp")],
    benchmarks = log_benchmarks
  ),
  decision_tree = list(
    depth_3 = list(
      test_accuracy = tree_d3$test_metrics$accuracy,
      macro_f1 = tree_d3$test_metrics$macro_f1,
      leaves = tree_d3$leaf_count
    ),
    unrestricted = list(
      train_accuracy = tree_unr$train_metrics$accuracy,
      test_accuracy = tree_unr$test_metrics$accuracy
    ),
    overfitting_experiment = overfit_table
  ),
  knn = list(
    raw_k5 = knn_raw$metrics[c("accuracy", "macro_f1")],
    norm_k5 = knn_norm$metrics[c("accuracy", "macro_f1")],
    interpretation = knn_comp$interpretation
  ),
  validation_summary = list(
    total_checks = nrow(validation_table),
    passed_checks = sum(validation_table$Status == "PASS")
  )
)

write_json(results_json, "results/results.json", pretty = TRUE, auto_unbox = TRUE)

cat("\n======================================================================\n")
cat(" REPRODUCTION COMPLETE: Results saved to results/results.csv & results.json\n")
cat(sprintf(" Validation Checks: %d / %d PASSED\n", sum(validation_table$Status == "PASS"), nrow(validation_table)))
cat("======================================================================\n")
