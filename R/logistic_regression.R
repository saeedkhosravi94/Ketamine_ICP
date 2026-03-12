
# load data
data <- read.csv("/Users/kwin/Master/SL/Project/data/Ketamine_icp.csv")
dim(data)
# observations: 32760, predictors: 1033 ; id, label, x0 to x999, statistical predictors like min, max, median, ...

# set label 0 as risky and label 1 as normal signal
data$label <- ifelse(data$label == 0, 1, 0)

# remove id column
data$id <- NULL

# cleaning data : checking for missing values 
# add a new predictor called has_missing in case there is at least one missing value inside the observation
data$has_missing <- as.integer(rowSums(is.na(data)) > 0)

# next we fill these missing value using data imputation using KNN with k = 10 
# install.packages("VIM")
library(VIM)
data_imputed <- kNN(data, k = 10)

# check number of missing values in data_imputed to make sure it's correctly filled.
sum(is.na(data_imputed))

# check observations based on label
table(data_imputed$label)









