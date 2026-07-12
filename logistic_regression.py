import pandas as pd
from train import train_eval
from data_prep import split_data, pca_transform_signals


from sklearn.linear_model import LogisticRegression

data = pd.read_csv("./data/Ketamine_icp_knn_imputed_k_20_missing.csv")

data_pca = pca_transform_signals(data, n_components=2)

train_data, test_data = split_data(data_pca, test_size=0.2, random_state=42, stratify=True)

model = LogisticRegression(max_iter=100, penalty="l1", solver="liblinear")

lgr_results = train_eval(model, train_data, test_data, kfolds=10)

print(lgr_results)