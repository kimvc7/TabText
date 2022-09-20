import pandas as pd
import numpy as np
from itertools import product
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
import json
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]

import sys

sys.path.insert(0, DIR_NAME + 'TabText/src/modules')
sys.path.insert(1, DIR_NAME + 'TabText/src/utils')


def load_embeddings(path, num_files=1):
    df = None
    for i in range( num_files):

        if i == 0:
            df = pd.read_csv(path + str(i) + ".csv")
        else:
            df_i = pd.read_csv(path + str(i) + ".csv" )
            df = pd.concat([df, df_i])

    return df


def train_xgb(train_df, test_df, target, n_est, max_param, lr, λ):
    ######################################
    #X_train, X_test, y_train and y_test are pandas dataframes. The number of rows corresponds to
    #the data size, and the number of columns corresponds to the number of features per data point.
    ######################################

    X_train, X_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]
    
    try:
        clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, eval_metric='logloss', reg_lambda=λ, tree_method='gpu_hist', gpu_id=0)
        clf.fit(X_train, y_train)
    except:
        clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, eval_metric='logloss', reg_lambda=λ)
        clf.fit(X_train, y_train)
    y_pred = np.array(clf.predict_proba(X_test)[:,1])

    auc = roc_auc_score(y_test, y_pred)
    return auc


def train_nn(train_df, test_df, target, dim, num_epochs, lr, batch_size, λ):
    X_train, X_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]

        
    model = Sequential()
    model.add(Dense(200, input_shape=(dim,), activation='relu', kernel_regularizer=regularizers.L1L2(l1=0, l2=λ)))
    model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.L1L2(l2=λ)))
    model.add(Dense(200, activation='relu', kernel_regularizer=regularizers.L1L2(l2=λ)))
    model.add(Dense(1, activation='sigmoid'))
    
    early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=500,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True)
    

    optimizer = keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.AUC()])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stopping])


    y_pred, test_auc = model.evaluate(X_test, y_test)
    return test_auc
