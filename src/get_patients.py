import pandas as pd
import numpy as np
import json

# Load config file with static parameters
with open('../../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]


import sys

sys.path.insert(0, DIR_NAME + 'TabText/src/modules')
sys.path.insert(1, DIR_NAME + 'TabText/src/utils')

from Patient import *
from data_utils import *

def get_attributes_info(df, info_file_path, encoders = None):
    attributes_info = {}
    info_file = pd.read_csv(info_file_path)
    for i in range(info_file.shape[0]):
        col_name, attribute, verb, neg_verb, col_type = info_file.iloc[i]
        sd, avg = None, None
        if col_type != "numerical":
            if encoders is None:
                le = encode_categorical_train(df[col_name])
                enc = lambda x: encode_categorical_unseen(x, le)
            if encoders is not None:
                enc = encoders[col_name]

        if col_type == "numerical":
            col_values = df[[col_name]].astype(np.float)
            col_values = col_values[col_name][pd.notnull(col_values[col_name])]
            avg = col_values.mean()
            sd = col_values.std()
            enc = lambda x: encode_numerical(x)

        attributes_info[col_name] = {"attribute": attribute,
                                    "column_verb": verb,
                                    "column_neg_verb": neg_verb,
                                    "column_type": col_type,
                                    "avg": avg,
                                    "sd": sd,
                                    "column_encoder": enc}
    return attributes_info

def get_unique_ids(tables_info, id_key):
    unique_ids = set()
    for i in range(len(tables_info)):
        table_df = tables_info[i]["df"]
        table_ids = table_df[id_key].unique()
        unique_ids.update(table_ids)
    return unique_ids

def create_columns(attributes_info):
    columns = []
    for col_name in attributes_info:
        col_attribute = attributes_info[col_name]["attribute"]
        col_verb = attributes_info[col_name]["column_verb"]
        col_type = attributes_info[col_name]["column_type"]
        col_enc = attributes_info[col_name]["column_encoder"]
        if col_type == "binary":
            col_neg_verb = attributes_info[col_name]["column_neg_verb"]
            column = Binary_Column(col_name, col_attribute, col_verb, col_neg_verb, col_enc)
        elif col_type == "categorical":
            column = Categorical_Column(col_name, col_attribute, col_verb, col_enc)
        else:
            avg = attributes_info[col_name]["avg"]
            sd = attributes_info[col_name]["sd"]
            column = Numerical_Column(col_name, col_attribute, col_verb, avg, sd, col_enc)
        columns.append(column) 
    return columns

def create_patient_tables(tables_info, pat_id, id_col, time_col):
    pat_tables = []
    for i in range(len(tables_info)):

        table_df = tables_info[i]["df"]
        table_name = tables_info[i]["name"]
        attributes_info = tables_info[i]["attributes_info"]
        columns = create_columns(attributes_info)
        metadata = tables_info[i]["metadata"]

        if pat_id in table_df[id_col].unique():
            pat_table_df = table_df[table_df[id_col] == pat_id]
            if time_col not in table_df.columns:
                table = Table(table_name, pat_table_df, columns, metadata, None)
            else:
                table = Table(table_name, pat_table_df, columns, metadata, time_col)
            pat_tables.append(table)
    return pat_tables

def get_patients(tables_info, id_col, time_col):
    unique_ids = get_unique_ids(tables_info, id_col)
    patients = []
    for pat_id in unique_ids:
        tables = create_patient_tables(tables_info, pat_id, id_col, time_col)
        patient = Patient(tables, pat_id, time_col)
        patients.append(patient)
    return patients
