import pandas as pd
import numpy as np
from itertools import product
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.callbacks import EarlyStopping
import json
import os

tf.config.run_functions_eagerly(True)

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../config.json') as config_file:
        config = json.load(config_file)

import sys

sys.path.insert(0, './../../src/modules')
sys.path.insert(1, './../../src/utils')

def load_embeddings(path, num_files=1, start = 0):
    df = None
    for i in range( start, num_files):
        if i == 0:
            df = pd.read_csv(path + str(i) + ".csv", index_col = 0)
        else:
            try:
                df_i = pd.read_csv(path + str(i) + ".csv", index_col=0)
                df = pd.concat([df, df_i])
            except Exception as e:
                print(e)
    return df


def train_xgb(train_df, test_df, target, n_est, max_param, lr, λ, num_classes=2):
    ######################################
    #X_train, X_test, y_train and y_test are pandas dataframes. The number of rows corresponds to
    #the data size, and the number of columns corresponds to the number of features per data point.
    ######################################

    X_train, X_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]
    
    y_test_bin = label_binarize(y_test, classes=np.arange(num_classes))
    
    loss_fn ='binary:logistic'
    if num_classes > 2:
        loss_fn = 'multi:softprob'
    
    try:
        if num_classes == 2:
            clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, objective=loss_fn, reg_lambda=λ, tree_method='gpu_hist', gpu_id=0)
        else:
            clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, objective=loss_fn, reg_lambda=λ, tree_method='gpu_hist', gpu_id=0, num_class=num_classes)
        clf.fit(X_train, y_train)
        
    except:
        if num_classes == 2:
            clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, objective=loss_fn, reg_lambda=λ)
        else:
            clf = XGBClassifier(n_estimators = n_est, max_depth = max_param, learning_rate = lr, objective=loss_fn, reg_lambda=λ, num_class=num_classes)
        clf.fit(X_train, y_train)
        
    # Get the predicted probabilities for each class
    y_pred_proba = clf.predict_proba(X_test)
    if num_classes == 2:
        y_pred_proba = np.array(clf.predict_proba(X_test)[:, 1])
    else:
        y_pred_proba = clf.predict_proba(X_test)
        
    # Get the predicted class labels
    y_pred = clf.predict(X_test)
    
    # Calculate AUC for each class using the one-vs-rest approach
    if num_classes ==2:
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = roc_auc_score(y_test_bin, y_pred_proba, multi_class='ovo')

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    return auc, accuracy, clf




def train_nn(train_df, test_df, target, dim, num_epochs, lr, batch_size, λ):
    tf.random.set_seed(0)
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


def train_cart(train_df, test_df, target, min_samp, max_param, max_features):
    ######################################
    #X_train, X_test, y_train and y_test are pandas dataframes. The number of rows corresponds to
    #the data size, and the number of columns corresponds to the number of features per data point.
    ######################################

    X_train, X_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]
    
    clf = DecisionTreeClassifier(min_samples_split = min_samp, max_depth = max_param, max_features = max_features, random_state =0)
    
    clf.fit(X_train, y_train)
    y_pred = np.array(clf.predict_proba(X_test)[:,1])

    auc = roc_auc_score(y_test, y_pred)
    return auc

def train_ridge(train_df, test_df, target, alpha):
    ######################################
    #X_train, X_test, y_train and y_test are pandas dataframes. The number of rows corresponds to
    #the data size, and the number of columns corresponds to the number of features per data point.
    ######################################

    X_train, X_test = train_df.drop(columns=[target]), test_df.drop(columns=[target])
    y_train, y_test = train_df[target], test_df[target]
    
  
    clf = RidgeClassifier(alpha = alpha, random_state =0)
    clf.fit(X_train, y_train)

    auc = roc_auc_score(y_test, clf.decision_function(X_test))
    return auc
