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

sys.path.insert(0, './../../src/utils')

from data_utils import *


def get_timebounded_data(timed_data, time_col, start = None, end = None):
    timebounded_df = timed_data.copy()

    if start is not None:
        timebounded_df = timed_data[timed_data[time_col].dt.date >= start]
    if end is not None:
        timebounded_df = timed_data[timed_data[time_col].dt.date < end]

    return timebounded_df


def get_features(ids, time_col, id_col, patients_path, feature_type, disch_info, weight_fn = hourly_weight_fn):
    """
    Parameters::
        ids: List of ids of patients whose features will be created.
        time_col: Name of column with timestamps.
        id_col: Name of column with patient ids.
        patients_path: Path to the folder containing the pickle files for the patients.
        feature_type: String describing the type of features to be created (e.g. "joint_embeddings", "joint_imputations")
        disch_info: Dataframe in which time_col column contains discharge time for the corresponding patient in id_col 
        weight_fn: Function that takes as input a list of timestamps and returns the corresponding list of weights, indicating the
        importance of each timstamp. Larger weights are more valuable.
        
    returns::
        features: Dataframe containing all the features of type feature_type for patients in ids.
        
        If a patient has data at timestamps t_1<= t_n, then 'n' features are created: the ith feature contains the patient
        information from t_1 to t_i. Data from different timetamps is averaged using the weights returned by weight_fn.
    """
    features = None
    count = 0
    for pat_id in ids:
        
        try:
            timed_data = pd.read_pickle(patients_path + str(pat_id) + ".pkl")
            
            if (time_col is not None):
                timed_data[time_col] = pd.to_datetime(timed_data[time_col], infer_datetime_format=True).dt.to_pydatetime()
                
            if (time_col is not None) and (disch_info is not None):
                discharge_info_pat = disch_info[disch_info[id_col] == pat_id]
                if discharge_info_pat.shape[0] > 0:
                    discharge_date = discharge_info_pat[time_col].dt.date.values[0]
                    timebounded_df = get_timebounded_data(timed_data, time_col, end=discharge_date)
                    features = add_patient_features(features, pat_id, timebounded_df, id_col, time_col, weight_fn)
            else:
                features = add_patient_features(features, pat_id, timed_data, id_col, time_col, weight_fn)
        
        except Exception as e:
            print(e)
    return features

def add_patient_features(features, pat_id, timebounded_df, id_col, time_col, weight_fn, time_series=False):
    feature_cols = list(timebounded_df.columns)[:]
    if time_col is not None:
        feature_cols.remove(time_col)
        timed_df = timebounded_df
        all_dates = timed_df.sort_values(time_col)[time_col].dt.date.unique()
        features_df = pd.DataFrame(columns = [id_col, time_col] + feature_cols)

        index = 0
        for date in all_dates:
            sub_df =  timed_df[timed_df[time_col].dt.date <= date]
            if time_series:
                sub_df["weights"] = weight_fn(sub_df[time_col].dt.to_pydatetime())
                new_row = [pat_id, date] + list(sub_df[feature_cols].multiply(sub_df["weights"], axis="index").sum().values)
            else:
                new_row = [pat_id, date] + [sub_df[x][sub_df[x].last_valid_index()] for x in feature_cols]
            features_df.loc[index] = new_row
            index +=1
            
    else:
        features_df = pd.DataFrame(columns = [id_col] + feature_cols)
        sub_df =  timebounded_df
        new_row = [pat_id] + [sub_df[x][sub_df[x].last_valid_index()] for x in feature_cols]
        features_df.loc[0] = new_row

    if features is None:
        return features_df
    else:
        return pd.concat([features, features_df])

def get_and_save_features(ids, time_col, id_col, feature_types, disch_info, pat_set, path, sentence_name, job_id="", weight_fn = hourly_weight_fn):
    for feature_type in feature_types:
        patient_path = path + pat_set + "/" + feature_type + "/" + sentence_name + "/Patients/"
        features = get_features(ids, time_col, id_col, patient_path, feature_type, disch_info, weight_fn = hourly_weight_fn)

        dir_name = path + pat_set + "/" + feature_type + "/" + sentence_name + "/Features"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        features.to_csv(dir_name + "/" + job_id + ".csv")
