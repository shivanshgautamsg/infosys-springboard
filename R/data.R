# ==============================================================================
# File: R/data.R
# Purpose: Dataset loading, stratified train/test partitioning, feature scaling,
#          and binary subset preparation for Industrial Training ML Experiments.
# Author: Reconstructed for Shivansh Gautam (Infosys Springboard ML in R)
# ==============================================================================

#' Load and format the raw Iris dataset
#'
#' @return data.frame containing the 150 observations of Iris flowers
get_raw_iris <- function() {
  data(iris, envir = environment())
  return(iris)
}

#' Get Iris dataset factual metadata
#'
#' @return list containing summary metadata documented in the report
get_dataset_metadata <- function() {
  list(
    name = "Fisher's Iris Flower Dataset",
    total_observations = 150,
    classes = c("setosa", "versicolor", "virginica"),
    observations_per_class = 50,
    features = c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"),
    target = "Species",
    units = "centimetres (cm)",
    training_split = 105,
    testing_split = 45,
    stratification = "Equal representation: 35 train / 15 test per species"
  )
}

#' Perform stratified 70/30 train/test split
#'
#' Partitions 150 observations into 105 training (35 per class) and
#' 45 testing (15 per class) maintaining equal representation.
#'
#' @param seed integer random seed. Default 3310.
#' @return list with train, test, train_idx, and test_idx
get_stratified_split <- function(seed = 1606) {
  df <- get_raw_iris()
  set.seed(seed)
  
  idx_setosa <- sample(which(df$Species == "setosa"), 35)
  idx_versicolor <- sample(which(df$Species == "versicolor"), 35)
  idx_virginica <- sample(which(df$Species == "virginica"), 35)
  
  train_idx <- c(idx_setosa, idx_versicolor, idx_virginica)
  test_idx <- setdiff(1:150, train_idx)
  
  train_data <- df[train_idx, ]
  test_data <- df[test_idx, ]
  
  list(
    train = train_data,
    test = test_data,
    train_idx = train_idx,
    test_idx = test_idx,
    seed = seed
  )
}

#' Min-Max Normalization (Scaling derived strictly from Training Data)
#'
#' Crucial ML best practice: Normalization parameters (min and max) are
#' calculated ONLY on the training split, and then applied to transform
#' both training and test sets. This prevents data leakage.
#'
#' @param train_x data.frame or matrix of training features
#' @param test_x data.frame or matrix of test features
#' @return list containing scaled train and test data, plus scaling parameters
min_max_scale_features <- function(train_x, test_x) {
  min_vals <- apply(train_x, 2, min)
  max_vals <- apply(train_x, 2, max)
  diff_vals <- max_vals - min_vals
  diff_vals[diff_vals == 0] <- 1
  
  train_scaled <- as.data.frame(scale(train_x, center = min_vals, scale = diff_vals))
  test_scaled <- as.data.frame(scale(test_x, center = min_vals, scale = diff_vals))
  
  list(
    train = train_scaled,
    test = test_scaled,
    min_vals = min_vals,
    max_vals = max_vals
  )
}

#' Extract Binary Classification Dataset (versicolor vs virginica)
#'
#' Used for Logistic Regression analysis (100 observations total,
#' 50 versicolor and 50 virginica).
#'
#' @return data.frame with factor levels 'versicolor' (reference) and 'virginica'
get_binary_iris <- function() {
  df <- get_raw_iris()
  binary_df <- subset(df, Species %in% c("versicolor", "virginica"))
  binary_df$Species <- factor(binary_df$Species, levels = c("versicolor", "virginica"))
  return(binary_df)
}
