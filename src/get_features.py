import pandas as pd
import datetime
from datetime import timedelta
import numpy as np
import json 
import sys
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]
sys.path.insert(0, DIR_NAME + 'TabText/src/utils')

from data_utils import *


def get_timebounded_data(timed_data, time_col, start = None, end = None):
    timebounded_df = timed_data.copy()

    if start is not None:
        timebounded_df = timed_data[timed_data[time_col].dt.date>= start]
    if end is not None:
        timebounded_df = timed_data[timed_data[time_col].dt.date < end]

    return timebounded_df

def get_features(ids, time_col, id_col, patients_path, feature_type, disch_info, weight_fn = hourly_weight_fn ):
    features = None

    for pat_id in ids:
        try:
            timed_data = pd.read_pickle(patients_path + str(pat_id) + ".pkl")
            timed_data = timed_data.fillna(0)
            timed_data[time_col] = timed_data[time_col].dt.to_pydatetime()
            discharge_info_pat = disch_info[disch_info[id_col] == pat_id]
            if discharge_info_pat.shape[0] > 0:
                discharge_date = discharge_info_pat[time_col].dt.date.values[0]
                timebounded_df = get_timebounded_data(timed_data, time_col, end=discharge_date)
                features = add_patient_features(features, pat_id, timebounded_df, id_col, time_col, weight_fn)
        except:
            print(pat_id)
    return features

def add_patient_features(features, pat_id, timebounded_df, id_col, time_col, weight_fn):
    feature_cols = list(timebounded_df.columns)[:]
    feature_cols.remove(time_col)
    timed_df = timebounded_df#[[time_col] + feature_cols]
    all_dates = timed_df.sort_values(time_col)[time_col].dt.date.unique()

    features_df = pd.DataFrame(columns = [id_col, time_col] + feature_cols)

    index = 0
    for date in all_dates:
        sub_df =  timed_df[timed_df[time_col].dt.date <= date]
        sub_df["weights"] = weight_fn(sub_df[time_col].dt.to_pydatetime())
        new_row = [pat_id, date] +list(sub_df[feature_cols].multiply(sub_df["weights"], axis="index").sum().values)
        features_df.loc[index] = new_row
        index +=1

    if features is None:
        return features_df
    else:
        return pd.concat([features, features_df])
    
def get_and_save_features(ids, time_col, id_col, feature_type, disch_info, pat_set, path, sentence_name, job_id="", weight_fn = hourly_weight_fn):
    patient_path = path + pat_set + "/" + feature_type + "/" + sentence_name + "/Patients/"
    features = get_features(ids, time_col, id_col, patient_path, feature_type, disch_info, weight_fn = hourly_weight_fn)
    
    dir_name = path + pat_set + "/" + feature_type + "/" + sentence_name + "/Features"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    features.to_csv(dir_name + "/" + job_id + ".csv")
