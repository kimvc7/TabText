import pandas as pd
import numpy as np
from itertools import product
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
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
