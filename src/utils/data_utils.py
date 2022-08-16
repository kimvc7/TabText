import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder

from julia import Julia
Julia(sysimage='/home/gridsan/groups/IAI/images/2.0.0/julia-1.5.2/sys.so', compiled_modules = False)
from interpretableai import iai


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


def get_label_encoder(column):
    le = LabelEncoder()
    le.fit(column[column.notnull()])
    return le

def encode_categorical(column, encoder):
    dic = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
    return column.map(dic).fillna(0).astype(int)

def get_iai_imputer(df):
    lnr = iai.ImputationLearner(method='opt_knn', random_seed=1,cluster=True)
    lnr.fit(df)
    return lnr

def iai_impute_data(df, imputer):
    return imputer.transform(df)

def impute_data(df, imputer_fn):
    imp_vectors = []
    for i in range(df.shape[0]):
        enc_vector = df.iloc[i][ENC_COL]
        imp_vector = imputer_fn(enc_vector)
        imp_vectors.append(imp_vector)

    return imp_vectors

def date_diff_hrs(t1, t0):
    delta_t = round((t1-t0).total_seconds()/3600) # Result in hrs
    return delta_t

def hourly_weight_fn(time_array):
    most_recent = max(time_array)
    delta_hrs = np.array([date_diff_hrs(most_recent, t) for t in time_array])
    flipped_delta = delta_hrs.max() - delta_hrs
    if flipped_delta.sum() == 0:
        weights = [1/len(flipped_delta) for i in range(len(flipped_delta))]
    else:
        weights = flipped_delta/flipped_delta.sum()
    return weights



