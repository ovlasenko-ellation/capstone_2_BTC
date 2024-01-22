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


data = 'https://github.com/ovlasenko-ellation/capstone_2_BTC/raw/main/BTC-USD_daily.csv'

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

"""Adding vectorization"""

dv = DictVectorizer(sparse=True)

train_dict = df_train.to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

val_dict = df_val.to_dict(orient='records')
X_val = dv.transform(val_dict)

"""#Training the final selected model"""

dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

rf = RandomForestClassifier(n_estimators=80, max_depth=10,  random_state=1)
rf.fit(X_full_train, y_full_train)
y_pred = rf.predict(X_test)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

"""##Saving model to a file"""

import pickle

output_file = 'model.bin'

f_out = open(output_file, 'wb')
pickle.dump((dv, rf), f_out)
f_out.close()

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, rf), f_out)