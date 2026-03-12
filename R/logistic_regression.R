
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
# based on the data we can see that we have an imbalanced dataset 
# to solve this issue we use scale_weight for each label based on it's proportion during training and only apply on train data. 

















