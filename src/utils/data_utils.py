import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder


# Load config file with static parameters
with open('../../config.json') as config_file:
        config = json.load(config_file)

ENC_COL = config["ENC_COL"]


def encode_numerical(column):
    new_col = []
    for i in range(column.shape[0]):
        if column.iloc[i] in ["none" , "nan", "missing", ""]:
            new_col.append(np.nan)
        else:
            new_col.append(float(column.iloc[i]))
    return new_col


def encode_categorical_train(column):
    le = LabelEncoder()
    le.fit_transform(column[column.notnull()])
    return le

def encode_categorical_unseen(column, le):
    dic = dict(zip(le.classes_, le.transform(le.classes_)))
    return column.map(dic).fillna(0).astype(int)
   

def impute_data(df, imputer_fn):
    imp_vectors = []
    for i in range(df.shape[0]):
        enc_vector = df.iloc[i][ENC_COL]
        imp_vector = imputer_fn(enc_vector)
        imp_vectors.append(imp_vector)

    return imp_vectors
