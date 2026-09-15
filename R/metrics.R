# ==============================================================================
# File: R/metrics.R
# Purpose: Statistical and Machine Learning evaluation metrics
#          (Regression diagnostics, Classification accuracy, Macro F1, etc.)
# Author: Reconstructed for Shivansh Gautam (Infosys Springboard ML in R)
# ==============================================================================

#' Calculate Regression Diagnostics
#'
#' @param model fitted lm model object
#' @return list of metrics: r_squared, adj_r_squared, rmse, mae, f_stat, p_value
calc_regression_metrics <- function(model) {
  s <- summary(model)
  res <- residuals(model)
  n <- length(res)
  
  rmse <- sqrt(mean(res^2))
  mae <- mean(abs(res))
  r2 <- s$r.squared
  adj_r2 <- s$adj.r.squared
  
  f_stat <- s$fstatistic[1]
  f_df1 <- s$fstatistic[2]
  f_df2 <- s$fstatistic[3]
  p_val <- pf(f_stat, f_df1, f_df2, lower.tail = FALSE)
  
  list(
    r_squared = as.numeric(r2),
    adj_r_squared = as.numeric(adj_r2),
    rmse = as.numeric(rmse),
    mae = as.numeric(mae),
    f_statistic = as.numeric(f_stat),
    p_value = as.numeric(p_val),
    n_obs = n
  )
}

#' Calculate Multiclass Classification Metrics (Macro-averaged)
#'
#' Evaluates accuracy, per-class precision, recall, and F1,
#' and returns macro-averaged values.
#'
#' @param actual factor of true class labels
#' @param predicted factor of predicted class labels
#' @return list containing accuracy, macro_f1, macro_precision, macro_recall,
#'         per_class_table, and confusion_matrix
calc_multiclass_metrics <- function(actual, predicted) {
  classes <- levels(actual)
  cm <- table(Actual = actual, Predicted = predicted)
  total <- sum(cm)
  acc <- sum(diag(cm)) / total
  
  per_class <- data.frame(
    Class = classes,
    TP = numeric(length(classes)),
    FP = numeric(length(classes)),
    FN = numeric(length(classes)),
    TN = numeric(length(classes)),
    Precision = numeric(length(classes)),
    Recall = numeric(length(classes)),
    F1 = numeric(length(classes)),
    stringsAsFactors = FALSE
  )
  
  for (i in seq_along(classes)) {
    c <- classes[i]
    tp <- cm[c, c]
    fp <- sum(cm[, c]) - tp
    fn <- sum(cm[c, ]) - tp
    tn <- total - (tp + fp + fn)
    
    prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
    rec <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
    f1 <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
    
    per_class$TP[i] <- tp
    per_class$FP[i] <- fp
    per_class$FN[i] <- fn
    per_class$TN[i] <- tn
    per_class$Precision[i] <- prec
    per_class$Recall[i] <- rec
    per_class$F1[i] <- f1
  }
  
  macro_prec <- mean(per_class$Precision)
  macro_rec <- mean(per_class$Recall)
  macro_f1 <- mean(per_class$F1)
  
  list(
    accuracy = as.numeric(acc),
    macro_precision = as.numeric(macro_prec),
    macro_recall = as.numeric(macro_rec),
    macro_f1 = as.numeric(macro_f1),
    per_class = per_class,
    confusion_matrix = cm
  )
}

#' Calculate Binary Classification Metrics
#'
#' @param actual factor with 2 levels (ref, positive)
#' @param predicted factor with 2 levels (ref, positive)
#' @param positive_class character name of positive class (default: "virginica")
#' @return list containing accuracy, precision, recall, f1, tn, fp, fn, tp, cm
calc_binary_metrics <- function(actual, predicted, positive_class = "virginica") {
  classes <- levels(actual)
  ref_class <- setdiff(classes, positive_class)
  
  cm <- table(Actual = actual, Predicted = predicted)
  total <- sum(cm)
  
  tp <- cm[positive_class, positive_class]
  fp <- cm[ref_class, positive_class]
  fn <- cm[positive_class, ref_class]
  tn <- cm[ref_class, ref_class]
  
  acc <- (tp + tn) / total
  prec <- ifelse(tp + fp == 0, 0, tp / (tp + fp))
  rec <- ifelse(tp + fn == 0, 0, tp / (tp + fn))
  f1 <- ifelse(prec + rec == 0, 0, 2 * prec * rec / (prec + rec))
  
  list(
    accuracy = as.numeric(acc),
    precision = as.numeric(prec),
    recall = as.numeric(rec),
    f1 = as.numeric(f1),
    tp = as.integer(tp),
    fp = as.integer(fp),
    fn = as.integer(fn),
    tn = as.integer(tn),
    total = as.integer(total),
    confusion_matrix = cm
  )
}
