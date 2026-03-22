import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data import Dataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, precision_score
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (Input, Conv1D, MaxPooling1D, Flatten, Dense, 
                                     Dropout, Add, LSTM, Bidirectional, GlobalAveragePooling1D, 
                                     MultiHeadAttention, LayerNormalization)

dataset = Dataset('./data/Ketamine_icp.csv', impute=True, k=5)
label = dataset.data['label']
features = dataset.data.drop('label', axis=1)
features_pca = dataset.apply_pca(features, variance_threshold=0.98)
signals = dataset.data.drop('label', axis=1).iloc[:, :1000]
stats = dataset.data.drop('label', axis=1).iloc[:, 1000:]
signals_pca = dataset.apply_pca(signals, variance_threshold=0.98)
# stats_pca = dataset.apply_pca(stats, variance_threshold=0.98)

df_signals_pca = pd.DataFrame(signals_pca, index=signals.index)
# df_stats_pca = pd.DataFrame(stats_pca, index=signals.index)
data = pd.concat([df_signals_pca, stats, label], axis=1) 
data.columns = data.columns.astype(str) 

X = data.drop('label', axis=1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_train_cnn = X_train_s.reshape(-1, X_train_s.shape[1], 1)
X_test_cnn = X_test_s.reshape(-1, X_test_s.shape[1], 1)

weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw_dict = dict(enumerate(weights))
w_ratio = weights[1] / weights[0]

def build_simple_cnn(shape):
    model = Sequential([
        Conv1D(32, 3, activation='relu', input_shape=shape),
        MaxPooling1D(2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

def build_resnet(shape):
    i = Input(shape=shape)
    x = Conv1D(32, 3, padding='same', activation='relu')(i)
    s = x
    x = Conv1D(32, 3, padding='same', activation='relu')(x)
    x = Conv1D(32, 3, padding='same')(x)
    x = Add()([x, s])
    x = tf.keras.layers.Activation('relu')(x)
    x = GlobalAveragePooling1D()(x)
    return Model(i, Dense(1, activation='sigmoid')(x))

def build_cnn_lstm(shape):
    i = Input(shape=shape)
    x = Conv1D(64, 3, activation='relu')(i)
    x = MaxPooling1D(2)(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dropout(0.3)(x)
    return Model(i, Dense(1, activation='sigmoid')(x))

def build_dilated_cnn(shape):
    i = Input(shape=shape)
    x = Conv1D(32, 3, padding='same', dilation_rate=1, activation='relu')(i)
    x = Conv1D(32, 3, padding='same', dilation_rate=2, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)
    return Model(i, Dense(1, activation='sigmoid')(x))

def build_attn_cnn(shape):
    i = Input(shape=shape)
    x = Conv1D(64, 3, padding='same', activation='relu')(i)
    a = MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = LayerNormalization()(Add()([x, a]))
    x = GlobalAveragePooling1D()(x)
    return Model(i, Dense(1, activation='sigmoid')(x))

sk_models = {
    'Logistic Regression': LogisticRegression(class_weight='balanced', max_iter=5000),
    'Random Forest': RandomForestClassifier(class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(scale_pos_weight=w_ratio, eval_metric='logloss', random_state=42),
    'LightGBM': LGBMClassifier(class_weight='balanced', verbosity=-1, random_state=42),
    'CatBoost': CatBoostClassifier(auto_class_weights='Balanced', verbose=0, random_state=42),
    'LDA': LinearDiscriminantAnalysis(),
    'MLP': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
    'Bayesian LR': LogisticRegression(solver='liblinear', penalty='l1', C=0.1, class_weight='balanced')
}

dl_models = {
    '1D-CNN (Base)': build_simple_cnn,
    'ResNet-1D': build_resnet,
    'CNN-LSTM': build_cnn_lstm,
    'Dilated-CNN': build_dilated_cnn,
    'Attention-CNN': build_attn_cnn
}

results_table = []
perf_profile_data = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

plt.figure(figsize=(12, 8))

for name, model in sk_models.items():
    print(f"Evaluating {name}...")
    cv_aucs = []
    for tr_idx, val_idx in cv.split(X_train_s, y_train):
        model.fit(X_train_s[tr_idx], y_train.iloc[tr_idx])
        p = model.predict_proba(X_train_s[val_idx])[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_train_s[val_idx])
        cv_aucs.append(roc_auc_score(y_train.iloc[val_idx], p))
    perf_profile_data[name] = cv_aucs
    
    model.fit(X_train_s, y_train)
    y_prob = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test_s)
    y_pred = model.predict(X_test_s)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results_table.append({'Model': name, 'AUC': auc, 'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(1-fpr, tpr, label=f'{name} (AUC={auc:.2f})')

for name, build_func in dl_models.items():
    print(f"Evaluating {name}...")
    cv_aucs = []
    for tr_idx, val_idx in cv.split(X_train_cnn, y_train):
        m = build_func((X_train_cnn.shape[1], 1))
        m.compile(optimizer='adam', loss='binary_crossentropy')
        m.fit(X_train_cnn[tr_idx], y_train.iloc[tr_idx], epochs=15, batch_size=16, verbose=0, class_weight=cw_dict)
        cv_aucs.append(roc_auc_score(y_train.iloc[val_idx], m.predict(X_train_cnn[val_idx])))
    perf_profile_data[name] = cv_aucs
    
    final_m = build_func((X_train_cnn.shape[1], 1))
    final_m.compile(optimizer='adam', loss='binary_crossentropy')
    final_m.fit(X_train_cnn, y_train, epochs=20, batch_size=16, verbose=0, class_weight=cw_dict)
    y_prob = final_m.predict(X_test_cnn).flatten()
    y_pred = (y_prob > 0.5).astype(int)
    auc = roc_auc_score(y_test, y_prob)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    results_table.append({'Model': name, 'AUC': auc, 'Sensitivity': tp/(tp+fn), 'Specificity': tn/(tn+fp), 'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn})
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(1-fpr, tpr, label=f'{name} (AUC={auc:.2f})', linestyle='--')

plt.plot([1, 0], [0, 1], 'k--', alpha=0.5)
plt.xlabel('Specificity')
plt.ylabel('Sensitivity')
plt.title('ROC Curve Comparison (13 Models)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.gca().invert_xaxis()
plt.tight_layout()
plt.show()

print("\n" + "="*90)
print(pd.DataFrame(results_table).to_string(index=False))

def plot_dolan_more(perf_dict):
    df_perf = 1 - pd.DataFrame(perf_dict)
    r_ps = df_perf.div(df_perf.min(axis=1), axis=0)
    taus = np.linspace(1, r_ps.values.max(), 100)
    plt.figure(figsize=(12, 7))
    for col in r_ps.columns:
        rho = [(r_ps[col] <= t).mean() for t in taus]
        plt.step(taus, rho, label=col, where='post')
    plt.title('Performance Profile (Dolan-Moré Study)')
    plt.xlabel('tau')
    plt.ylabel('rho')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_dolan_more(perf_profile_data)