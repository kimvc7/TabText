import pandas as pd
import numpy as np
import json
import pickle
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../config.json') as config_file:
        config = json.load(config_file)

ALL_TYPES = ["joint_embeddings", "sep_embeddings", "joint_imputations", "sep_imputations", "text"]

import sys

sys.path.insert(0, './../../src/modules')
sys.path.insert(1, './../../src/utils')

from Patient import *
from data_utils import *


def create_columns(attributes_info):
    """
    Creates Column objects for the attributes in attributes_info
    """
    columns = []
    for col_name in attributes_info.names:
        attribute_info = getattr(attributes_info, col_name)
        col_attribute = attribute_info.attribute
        col_verb = attribute_info.column_verb
        col_type = attribute_info.column_type
        col_enc = attribute_info.column_encoder
        
        if col_type == "binary":
            col_neg_verb = attribute_info.column_neg_verb
            column = Binary_Column(col_name, col_attribute, col_verb, col_neg_verb, col_enc)
            
        elif col_type == "categorical":
            column = Categorical_Column(col_name, col_attribute, col_verb, col_enc)
            
        else:
            avg = attribute_info.avg
            sd = attribute_info.sd
            column = Numerical_Column(col_name, col_attribute, col_verb, avg, sd, col_enc)
        columns.append(column) 
        
    return columns

def create_patient_tables(tables_info, pat_id, id_col, time_col, clinical, long, biogpt, finetuned_path):
    """
    Creates Table objects for patient with id pat_id using dataframes in tables_info
    """
    pat_tables = []
    for i in range(len(tables_info)):

        table_name = tables_info[i].name
        attributes_info = tables_info[i].attributes_info
        columns = create_columns(attributes_info)
        metadata = tables_info[i].metadata
        imputer = tables_info[i].imputer
        table_df = tables_info[i].df
        empty = False

        if pat_id in table_df[id_col].unique():
            pat_table_df = table_df[table_df[id_col] == pat_id]
            
        else:
            empty = True
            missing_row = np.array([np.nan for i in range(len(table_df.columns))]).reshape((1, len(table_df.columns)))
            pat_table_df = pd.DataFrame(missing_row, columns = table_df.columns)
        if (time_col not in table_df.columns) or empty:
            table = Table(table_name, pat_table_df, columns, metadata, None, imputer, clinical, long, biogpt, finetuned_path)
        else:
            table = Table(table_name, pat_table_df, columns, metadata, time_col, imputer, clinical, long, biogpt, finetuned_path)
        
        pat_tables.append(table)

    return pat_tables

def get_patients(tables_info, id_col, time_col, unique_ids):
    """
    Creates Patient objects for each patient in unique_ids using data from tables_info
    """
    patients = []
    for pat_id in unique_ids:
        tables = create_patient_tables(tables_info, pat_id, id_col, time_col)
        patient = Patient(tables, pat_id, time_col)
        patients.append(patient)
    return patients





def get_and_save_pickle_patients(tables_info, id_col, time_col, ids, prefix, missing, replace, descriptive, meta, global_imp, pat_set, path, data_set, clinical, long, biogpt, finetuned_path="", feature_types = ALL_TYPES):
    """
    Creates and saves patient objects as pickle files for each patient in ids.
    """
    patients = []
    for pat_id in ids:
        tables = create_patient_tables(tables_info, pat_id, id_col, time_col, clinical, long, biogpt, finetuned_path)
        patient = Patient(tables, pat_id, time_col)
        patient.create_timed_data(prefix, missing, replace, descriptive, meta, global_imp, feature_types)

        for feature_type in feature_types:
            sent_name = data_set+"_"+ str(prefix) +"_"+ str(missing) +"_"+ str(replace) +"_"+ str(descriptive) +"_"+ str(meta)
            dir_name = path + pat_set +"/"+ feature_type +"/"+ sent_name + "/Patients"

            if not os.path.exists(dir_name):
                os.makedirs(dir_name)

            getattr(patient, feature_type).to_pickle(dir_name + "/" + str(patient.id) + ".pkl") 

