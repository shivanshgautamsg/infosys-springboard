# ==============================================================================
# File: tests/testthat/test_models.R
# Purpose: Unit tests for dataset integrity, regression, logistic, decision tree,
#          and KNN implementations.
# Author: Shivansh Gautam (Infosys Springboard ML in R Demonstration)
# ==============================================================================

library(testthat)

if (file.exists("../../R/data.R")) setwd("../..")
source("R/data.R")
source("R/metrics.R")
source("R/regression.R")
source("R/multicollinearity.R")
source("R/logistic.R")
source("R/decision_tree.R")
source("R/knn.R")

test_that("Dataset dimensions and metadata match report specifications", {
  meta <- get_dataset_metadata()
  df <- get_raw_iris()
  
  expect_equal(nrow(df), 150)
  expect_equal(ncol(df), 5)
  expect_equal(as.numeric(table(df$Species)), c(50, 50, 50))
  expect_equal(meta$training_split, 105)
  expect_equal(meta$testing_split, 45)
})

test_that("Stratified split creates equal representation", {
  split_data <- get_stratified_split(1606)
  expect_equal(nrow(split_data$train), 105)
  expect_equal(nrow(split_data$test), 45)
  expect_equal(as.numeric(table(split_data$train$Species)), c(35, 35, 35))
  expect_equal(as.numeric(table(split_data$test$Species)), c(15, 15, 15))
  expect_length(intersect(split_data$train_idx, split_data$test_idx), 0)
})

test_that("Min-Max normalization prevents data leakage", {
  split_data <- get_stratified_split(1606)
  scaled <- min_max_scale_features(split_data$train[, 1:4], split_data$test[, 1:4])
  
  # Training data bounds must be strictly [0, 1]
  expect_true(all(scaled$train >= 0 & scaled$train <= 1))
  
  # Scaling parameters must match training min and max
  expect_equal(scaled$min_vals, apply(split_data$train[, 1:4], 2, min))
  expect_equal(scaled$max_vals, apply(split_data$train[, 1:4], 2, max))
})

test_that("Simple and Multiple Linear Regression metrics match report targets", {
  simple <- fit_simple_regression()
  expect_equal(simple$metrics$r_squared, 0.9271, tolerance = 0.001)
  expect_equal(simple$metrics$adj_r_squared, 0.9266, tolerance = 0.001)
  expect_equal(simple$metrics$rmse, 0.4750, tolerance = 0.002)
  expect_equal(as.numeric(coef(simple$model)["Petal.Width"]), 2.2299, tolerance = 0.001)
  
  multiple <- fit_multiple_regression()
  expect_equal(multiple$metrics$r_squared, 0.9680, tolerance = 0.001)
  expect_equal(multiple$metrics$adj_r_squared, 0.9674, tolerance = 0.001)
  expect_equal(multiple$metrics$rmse, 0.3147, tolerance = 0.002)
  expect_equal(as.numeric(coef(multiple$model)["Petal.Width"]), 1.4468, tolerance = 0.001)
})

test_that("VIF values remain below 4.0 as reported", {
  vif_df <- calc_vif()
  expect_true(all(vif_df$VIF < 4.0))
  expect_equal(vif_df$VIF[vif_df$Predictor == "Petal.Width"], 3.890, tolerance = 0.01)
  expect_equal(vif_df$VIF[vif_df$Predictor == "Sepal.Length"], 3.416, tolerance = 0.01)
  expect_equal(vif_df$VIF[vif_df$Predictor == "Sepal.Width"], 1.306, tolerance = 0.01)
})

test_that("Logistic regression matches binary evaluation targets", {
  res_05 <- evaluate_logistic_threshold(0.5)
  expect_equal(res_05$accuracy, 0.9400, tolerance = 0.001)
  expect_equal(res_05$f1, 0.9400, tolerance = 0.001)
  expect_equal(res_05$tn, 47)
  expect_equal(res_05$tp, 47)
  expect_equal(res_05$fp, 3)
  expect_equal(res_05$fn, 3)
  
  res_03 <- evaluate_logistic_threshold(0.3)
  expect_equal(res_03$accuracy, 0.9600, tolerance = 0.001)
  
  res_07 <- evaluate_logistic_threshold(0.7)
  expect_equal(res_07$accuracy, 0.9500, tolerance = 0.001)
})

test_that("Decision tree and overfitting divergence reproduce expected behavior", {
  split_data <- get_stratified_split(1606)
  tree_d3 <- fit_decision_tree(3, split_data)
  expect_equal(tree_d3$test_metrics$accuracy, 0.9778, tolerance = 0.005)
  
  tree_unr <- fit_decision_tree(NULL, split_data)
  expect_equal(tree_unr$train_metrics$accuracy, 1.0000, tolerance = 0.001)
  expect_equal(tree_unr$test_metrics$accuracy, 0.9333, tolerance = 0.01)
})

test_that("KNN scaling demonstrates counterintuitive performance drop", {
  split_data <- get_stratified_split(1606)
  knn_raw <- run_knn_classifier(k = 5, normalized = FALSE, split_data = split_data)
  knn_norm <- run_knn_classifier(k = 5, normalized = TRUE, split_data = split_data)
  
  expect_equal(knn_raw$metrics$accuracy, 0.9778, tolerance = 0.005)
  expect_equal(knn_norm$metrics$accuracy, 0.9333, tolerance = 0.005)
  expect_true(knn_raw$metrics$accuracy > knn_norm$metrics$accuracy)
})
