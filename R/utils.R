library(pROC)
library(caret)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(doParallel)
library(foreach)
library(DMwR2)

plot_pca_elbow <- function(data, max_components = 50) {
  data_scaled <- scale(data)
  
  var_explained <- numeric(max_components)
  pca <- prcomp(data_scaled, center = TRUE, scale. = TRUE)
  total_variance <- sum(pca$sdev^2)
  
  for (k in 1:max_components) {
    var_explained[k] <- sum(pca$sdev[1:k]^2) / total_variance
  }
  
  df <- data.frame(
    Components = 1:max_components,
    VarianceExplained = 1 - var_explained
  )
  
  ggplot(df, aes(x = Components, y = VarianceExplained)) +
    geom_line(color = "skyblue") +
    geom_point(color = "orange") +
    xlab("Number of Components") +
    ylab("1 - Cumulative Variance Explained") +
    ggtitle("PCA on Signals")
}

train_eval <- function(signals, stats, labels) {
  
  set.seed(123)
  labels <- as.factor(labels)
  
  train_idx <- createDataPartition(labels, p = 0.85, list = FALSE)
  test_idx <- setdiff(seq_len(nrow(signals)), train_idx)
  
  y_train <- labels[train_idx]
  y_test <- labels[test_idx]
  
  signals_train <- as.matrix(signals[train_idx, , drop = FALSE])
  signals_test <- as.matrix(signals[test_idx, , drop = FALSE])
  stats_train <- stats[train_idx, , drop = FALSE]
  stats_test <- stats[test_idx, , drop = FALSE]
  
  class_counts <- table(y_train)
  minority <- names(which.min(class_counts))
  weight_ratio <- as.numeric(max(class_counts) / min(class_counts))
  w <- ifelse(y_train == minority, weight_ratio, 1)
  
  folds <- createFolds(y_train, k = 10)
  foldid <- rep(NA, length(y_train))
  for (i in seq_along(folds)) {
    foldid[folds[[i]]] <- i
  }
  
  compute_auc <- function(x_train, x_test) {
    cv_model <- cv.glmnet(
      x = x_train,
      y = y_train,
      family = "binomial",
      alpha = 1,
      weights = w,
      foldid = foldid,
      type.measure = "auc"
    )
    
    best_lambda <- cv_model$lambda.min
    
    model <- glmnet(
      x = x_train,
      y = y_train,
      family = "binomial",
      alpha = 1,
      lambda = best_lambda,
      weights = w
    )
    
    probs <- predict(model, newx = x_test, type = "response")[,1]
    roc_test <- roc(as.numeric(as.character(y_test)), probs)
    
    list(
      test_auc = as.numeric(auc(roc_test)),
      train_auc_mean = max(cv_model$cvm),
      probs = probs,
      model = model
    )
  }
  
  results <- list()
  res_signals <- compute_auc(signals_train, signals_test)
  results[["Signals Only"]] <- res_signals$test_auc
  
  n_cores <- parallel::detectCores() - 1
  cl <- makeCluster(n_cores)
  registerDoParallel(cl)
  
  stat_results <- foreach(col = colnames(stats_train),
                          .combine = rbind,
                          .packages = c("glmnet", "pROC")) %dopar% {
                            x_train <- cbind(signals_train, stats_train[, col])
                            x_test <- cbind(signals_test, stats_test[, col])
                            res <- compute_auc(as.matrix(x_train), as.matrix(x_test))
                            
                            data.frame(
                              Predictor = col,
                              AUC = res$test_auc
                            )
                          }
  
  stopCluster(cl)
  registerDoSEQ()
  
  auc_df <- rbind(
    data.frame(Predictor = "Signals Only", AUC = results[["Signals Only"]]),
    stat_results
  )
  
  # Visualization
  bar_plot <- ggplot(auc_df, aes(x = reorder(Predictor, AUC), y = AUC)) +
    geom_bar(stat = "identity", fill = "skyblue") +
    geom_text(aes(label = round(AUC, 3)), 
              hjust = 1.1, color = "black", size = 4) +
    coord_flip() +
    theme_minimal() +
    ggtitle("Test AUC Comparison") +
    ylim(0, 1)
  
  print(bar_plot)
  
  best_name <- auc_df$Predictor[which.max(auc_df$AUC)]
  
  if (best_name == "Signals Only") {
    best_train <- signals_train
    best_test <- signals_test
  } else {
    best_train <- cbind(signals_train, stats_train[, best_name])
    best_test <- cbind(signals_test, stats_test[, best_name])
  }
  
  final_res <- compute_auc(as.matrix(best_train), as.matrix(best_test))
  
  preds_class <- ifelse(final_res$probs > 0.5, 1, 0)
  cm <- table(Predicted = preds_class, Actual = as.numeric(as.character(y_test)))
  
  roc_obj <- roc(as.numeric(as.character(y_test)), final_res$probs)
  auc_value <- as.numeric(auc(roc_obj))
  
  roc_plot <- ggplot() +
    geom_line(aes(x = 1 - roc_obj$specificities,
                  y = roc_obj$sensitivities),
              linewidth = 1) +
    geom_abline(linetype = "dashed") +
    theme_minimal() +
    ggtitle(paste0("ROC - ", best_name, " | AUC = ", round(auc_value, 3))) +
    xlab("False Positive Rate") +
    ylab("True Positive Rate")
  
  cm_df <- as.data.frame(cm)
  cm_plot <- ggplot(cm_df, aes(x = Actual, y = Predicted)) +
    geom_tile(fill = "white", color = "black") +
    geom_text(aes(label = Freq), size = 6) +
    theme_minimal() +
    ggtitle("Confusion Matrix")
  
  grid.arrange(cm_plot, roc_plot, ncol = 2)
  
  return(list(
    AUC_table = auc_df,
    Best_Predictor = best_name,
    Mean_Train_AUC = final_res$train_auc_mean,
    Test_AUC = final_res$test_auc,
    TP = cm["1","1"],
    TN = cm["0","0"],
    FP = cm["1","0"],
    FN = cm["0","1"]
  ))
}