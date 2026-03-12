source("R/utils.R")

# load data
data <- read.csv("/Users/kwin/Master/SL/Project/data/Ketamine_icp.csv")
dim(data)
# observations: 32760, predictors: 1033 ; id, label, x0 to x999, statistical predictors like min, max, median, ...

# set label 0 as risky and label 1 as normal signal
data$label <- ifelse(data$label == 0, 1, 0)

# remove id column
data$id <- NULL

# cleaning data : checking for missing values 
# we can see in the report figure 1, bar chart that 25% of the observations that labeled as risky has missing values
# add a new predictor called has_missing in case there is at least one missing value inside the observation
data$has_missing <- as.integer(rowSums(is.na(data)) > 0)

# next we fill these missing value using data imputation using KNN with k = 10 
# install.packages("VIM") single core useless on this size of data
#library(VIM)
#data_imputed <- kNN(data, k = 1)

if (!require("BiocManager", quietly = TRUE)) 
  install.packages("BiocManager")
  BiocManager::install("impute")

library(impute)
data_matrix <- as.matrix(data)
data_imputed <- impute.knn(data_matrix, k = 5)$data
data_imputed <- as.data.frame(data_imputed)
# check number of missing values in data_imputed to make sure it's correctly filled.
sum(is.na(data_imputed))

# check observations based on label
table(data_imputed$label)

#     0     1 
#.    951   31809 
# based on the data we can see that we have an imbalanced data  
# to solve this issue we use scale_weight for each label based on it's proportion during training and only apply on train data. 

# Predictor Selection
# We have 2 different types of predictors, Signals, and Statistical Predictors of the Signals

# Signal selection: We use PCA and we try to find only predictors that contain more information 
# so we are looking for predictors that maximizes the sum of explained variance.

signals <- data_imputed[, 2:1001]
stats <- data_imputed[, 1002:1034]
labels <- data_imputed[, 1]

plot_pca_elbow(signals, 50)


# plot shows that using elbow method by considering only 12 predictors we can explain 
# 95% of the information of 1000 signal predictors to the model. There is a 
# multicollinearity (high correlation - linear relationship) between signals


n_comp <- 12 
pca_result <- prcomp(signals, center = TRUE, scale. = TRUE)
pca_signals <- pca_result$x[, 1:n_comp]













