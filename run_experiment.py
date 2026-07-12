import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_prep import split_data, pca_transform_signals
from train import train_eval


data = pd.read_csv("./data/Ketamine_icp_knn_imputed_k_20_missing.csv")

data_pca = pca_transform_signals(data, n_components=4)


train_data, test_data = split_data(data, test_size=0.2, random_state=42, stratify=True)

model = LogisticRegression(max_iter=100)





lgr_results = train_eval(model, train_data, test_data, kfolds=10)

print(lgr_results)