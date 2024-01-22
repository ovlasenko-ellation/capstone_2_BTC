# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, mean_squared_error, roc_auc_score, roc_curve, auc, f1_score
import xgboost as xgb
from sklearn.model_selection import KFold
from tqdm import tqdm

data = 'https://github.com/ovlasenko-ellation/capstone_2_BTC/raw/main/Data_input/BTC-USD_daily.csv'

btc = pd.read_csv(data)
btc.columns = btc.columns.str.lower().str.replace(' ', '_')

"""##Feature Engineering 

Adding columns for calculationg RSI index
"""
# Calculate daily price changes with previous day
N = 14  # Period for calculating averages
btc['price_change'] = btc['close'].diff()

# Separate gains and losses
btc['gain'] = btc['price_change'].apply(lambda x: max(x, 0))
btc['loss'] = btc['price_change'].apply(lambda x: max(-x, 0))

# Calculate average gain and average loss
avg_gain = btc['gain'].rolling(window=N, min_periods=1).mean()
avg_loss = btc['loss'].rolling(window=N, min_periods=1).mean()

# Calculate relative strength (RS)
rs = avg_gain / avg_loss

# Calculate RSI
btc['rsi'] = 100 - (100 / (1 + rs))

btc['price_diff_tomorrow'] = - btc['close'].diff(periods=-1)  # difference with the following day
btc['price_diff_tmr_is_up'] = btc['price_diff_tomorrow'] > 0 #new feature to underestand trend will be target

"""Adding additional features to identify if market is bull or bear, overfitted"""

btc['is_bull'] = btc['rsi'] > 50
btc['overbought'] = btc['rsi'] > 70
btc['oversold'] = btc['rsi'] < 30

btc['trend_change'] = ((btc['is_bull'] == True) & (btc['overbought'] == True)) | ((btc['is_bull'] == False) & (btc['oversold'] == True))
btc

btc = btc.fillna(0)

"""##Split the data

Since trading data has temporal patters and not evenly split around the year, the approach for data split will be to perform a chronological split, where the earlier part of the time series will be used for training, the middle part for validation, and the latest part for testing. This ensures that the model is trained on past data and validated on a more recent period
"""

len(btc)

btc = btc.sort_values('date')  # Sort the DataFrame by date

# Set the proportions for train, validation, and test sets
train_ratio = 0.6
val_ratio = 0.2

# Calculating the number of samples for each set
num_samples = len(btc)
n_train = int(train_ratio * num_samples)
n_val = int(val_ratio * num_samples)

# Splitting the data
df_train = btc[:n_train]
df_val = btc[n_train:n_train + n_val]
df_test = btc[n_train + n_val:]
df_full_train = pd.concat([df_train, df_val])

# Resetting the index after splitting
df_train.reset_index(drop=True, inplace=True)
df_val.reset_index(drop=True, inplace=True)
df_test.reset_index(drop=True, inplace=True)

# Resetting the index after splitting
df_train.reset_index(drop=True, inplace=True)

"""Separating the dataset from the target values"""

y_train = df_train.price_diff_tmr_is_up.astype('int').values
y_val = df_val.price_diff_tmr_is_up.astype('int').values
y_test = df_test.price_diff_tmr_is_up.astype('int').values

#For the final model
y_full_train = df_full_train.price_diff_tmr_is_up.values

#Removing target columns from the dataset
del df_train['price_diff_tmr_is_up']
del df_val['price_diff_tmr_is_up']
del df_test['price_diff_tmr_is_up']

del df_train['price_diff_tomorrow']
del df_val['price_diff_tomorrow']
del df_test['price_diff_tomorrow']

#For the final model
del df_full_train['price_diff_tmr_is_up']
del df_full_train['price_diff_tomorrow']

"""#Training the final selected model"""

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

features = dv.get_feature_names_out()
features_list = features.tolist()  # Convert to a list of strings

dtrain = xgb.DMatrix(X_full_train, label=y_full_train, feature_names=features_list)
dval = xgb.DMatrix(X_test, label=y_test, feature_names=features_list)

watchlist = [(dtrain, 'train'), (dval, 'eval')]

# Assuming you have X_full_train and y_full_train

# Define the number of splits for K-fold
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

# Initialize lists to store scores
scores_accuracy = []
scores_f1 = []

# Define XGBoost parameters
xgb_params = {
    'eta': 0.3,
    'max_depth': 3,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

# Loop through K-fold splits
for train_idx, val_idx in tqdm(kf.split(X_full_train)):
    X_train_fold, X_val_fold = X_full_train[train_idx], X_full_train[val_idx]
    y_train_fold, y_val_fold = y_full_train[train_idx], y_full_train[val_idx]

    # Create DMatrix for XGBoost

    dtrain_fold = xgb.DMatrix(X_train_fold, label=y_train_fold, feature_names=features_list)
    dval_fold = xgb.DMatrix(X_val_fold, label=y_val_fold, feature_names=features_list)

    # Train the XGBoost model
    model = xgb.train(xgb_params, dtrain_fold, num_boost_round=60, verbose_eval=5, evals=[(dtrain_fold, 'train'), (dval_fold, 'eval')])

    # Predict on the validation fold
    y_pred_fold = model.predict(dval_fold)

    # Convert probabilities to binary predictions
    y_pred_fold_binary = (y_pred_fold > 0.5).astype(int)

    # Evaluate accuracy and f1_score
    accuracy_fold = accuracy_score(y_val_fold, y_pred_fold_binary)
    f1_fold = f1_score(y_val_fold, y_pred_fold_binary)

    # Store scores
    scores_accuracy.append(accuracy_fold)
    scores_f1.append(f1_fold)

# Calculate and print mean scores
mean_accuracy = np.mean(scores_accuracy)
mean_f1 = np.mean(scores_f1)
print(f'Mean Accuracy: {mean_accuracy:.4f}')
print(f'Mean F1 Score: {mean_f1:.4f}')

"""##Saving model to a file"""

import pickle
output_file = 'model_new.bin'
output_file

# Save the final model to a file using pickle
with open(output_file, 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the DictVectorizer to a file using pickle
with open('dict_vectorizer.pickle', 'wb') as dv_file:
    pickle.dump(dv, dv_file)
