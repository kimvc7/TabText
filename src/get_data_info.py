import datetime
from datetime import timedelta
import pandas as pd
import sys
import json
import pickle
import os
import pathlib

with open(os.path.dirname(__file__) + '/../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]
ATTRIBUTES_PATH = config["ATTRIBUTES_PATH"]
ATTRIBUTES_PATH = config["ATTRIBUTES_PATH"]
IMPUTERS_PATH = config["IMPUTERS_PATH"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')

import dill as pickle
from data_utils import *

class Attribute(object):
    """
    Attribute object containing the information for a column in a specific tabular data structure.
    
        attribute:  String corresponging to the text description of the column.
        verb: Verb used to conjugate the attribute.
        neg_verb: Negative form of the verb used to conjugate the attribute.
        col_type: The type of the column: binary, categorical or numerical.
        avg: The average of the values observed for this column (to be computed usign Training set)
        sd: The standard deviation of the values observed for this column (to be computed usign Training set)
        column_encoder: The function used to encode categorical values of this column.
    """
    def __init__(self, attribute, verb, neg_verb, col_type, avg, sd, encoder):
        self.attribute = attribute
        self.column_verb = verb
        self.column_neg_verb = neg_verb
        self.column_type = col_type
        self.avg = avg
        self.sd = sd
        self.column_encoder = encoder
        
class AttributesInfo(object):
    """
    Module whose attributes are Attribute objects for each column in a specific tabular data structure.
    
        df: Dataframe with the tabular data
        info_file_path: String with the path to the file with the column information for each relevant column in df
        create_encoder_fn: Function to be used for encoding values of categorical columns 
    """
    def __init__(self, df, info_file_path, create_encoder_fn=get_label_encoder):
        self.info_file = pd.read_csv(info_file_path, keep_default_na=False, na_values=[''])
        self.create_attributes(df, create_encoder_fn)
        self.names = self.find_attribute_names()

    def find_attribute_names(self):
        names = []
        for i in range(self.info_file.shape[0]):
            col_name, attribute, verb, neg_verb, col_type = self.info_file.iloc[i]
            names.append(col_name)
        return names

    def create_attributes(self, df, create_encoder_fn):
        for i in range(self.info_file.shape[0]):
            col_name, attribute, verb, neg_verb, col_type = self.info_file.iloc[i]
            sd, avg = None, None

            if col_type != "numerical":
                col_values = df[[col_name]].astype(str)
                col_values = col_values[col_name][pd.notnull(col_values[col_name])]
                col_encoder = create_encoder_fn(col_values)
                enc = lambda x: encode_categorical(x, col_encoder)

            if col_type == "numerical":
                col_values = df[[col_name]].astype(np.float)
                col_values = col_values[col_name][pd.notnull(col_values[col_name])]
                avg = col_values.mean()
                sd = col_values.std()
                enc = encode_numerical
            setattr(self, col_name,  Attribute(attribute, verb, neg_verb, col_type, avg, sd, enc))

class TableInfo(object):
    """
    Module with all the relevant information of a specific tabular data structure.
    
        name: String with the name to be used for the table
        metadata: String with the written description of the information in the table
        imputer: Function to be used for imputing missing values in the table
        df: DataFrame containing the tabular data
        attributes_info: AttributesInfo object with the information for all the columns in the table
    
    """
    def __init__(self, name, metadata, imputer, df, attributes_info):
        self.name = name
        self.metadata = metadata
        self.imputer = imputer
        self.df = df
        self.attributes_info = attributes_info

def encode_table_df(table_df, attributes_info):
    """
    Parameters::
        table_df: Dataframe with the tabular data
        attributes_info:AttributesInfo object with the information for all the columns in table_df
    
    Returns:: 
        encoded_df: Encoded version of table_df
    """
    encoded_df = table_df.copy()
    cols_to_encode = attributes_info.names
    
    for col in cols_to_encode:
        col_values = table_df[col]
        col_encoder = getattr(attributes_info, col).column_encoder
        labels = col_encoder(col_values[col_values.notnull()])
        encoded_df[col] = pd.Series(labels, index=col_values[col_values.notnull()].index)
    
    return encoded_df

def get_filtered_ids(admission_df, discharge_df, id_col, time_col, location_col, loc, start=None, end=None):
    """
    Parameters:
        admission_df: Dataframe in which time_col column contains admission time for the corresponding patient in id_col
        discharge_df: Dataframe in which time_col column contains discharge time for the corresponding patient in id_col
        id_col: Name of the column containing the patient ids
        time_col: Name of the column containing the timestamps
        location_col: Name of the column in admission_df and discharge_df containing the location of the patient
        loc: Desired location to filter patients by.
        start: minimum admission time (used to filter patients)
        end: maximum discharge time (used to filter patients)
        
    Returns::
        list of patient ids satisfying all filtering conditions.
    """
    
    admission_df = admission_df[admission_df[location_col] == loc]
    discharge_df = discharge_df[discharge_df[location_col] == loc]
    
    admission_df["ADMISSION_DATE"] = pd.to_datetime(admission_df[time_col], infer_datetime_format=True)
    discharge_df["DISCHARGE_DATE"] = pd.to_datetime(discharge_df[time_col], infer_datetime_format=True)
    
    if start is not None:
        admission_df = admission_df[admission_df["ADMISSION_DATE"].dt.date >= start]
    if end is not None:
        discharge_df = discharge_df[discharge_df["DISCHARGE_DATE"].dt.date < end]
        
    los_df = discharge_df.merge(admission_df, on=["PAT_ENC_CSN_ID"], how="inner")
    los_df["LOS"] = los_df["DISCHARGE_DATE"] - los_df["ADMISSION_DATE"]

    #Filter only to inpatients that stay in hospital for at least two different dates
    los_df = los_df[los_df["LOS"]>= datetime.timedelta(days=1)]
    filtered_ids = set(los_df["PAT_ENC_CSN_ID"].unique()) 

    return list(filtered_ids)
    


def save_model_info(paths, id_col, time_col, imputer, pat_ids):
    """
    Saves the AttributedInfo object and the imputer function for each table, as well as as the global imputer for imputating
    merged tables.
    
    Parameters::
        paths = list of four paths [example_path, data_path, tables_path, columns_path]
            example_path: Path to the folder for the specific application.
            data_path: Path to the folder withall the data.
            tables_path: Path to the csv file with the information about each table (relative to data_path)
            columns_path: Path to the folder with the csv files containing information for the columns in each table.
        id_col: Name of the column containing the patient ids
        time_col: Name of the column containing the timestamps
        imputer: String indicating the type of imputer to use; one of "zero_imp" or "iai_imp".
        pat_ids: List of patient ids whose data will be used to save the model information (training data)
    """
    example_path, data_path, tables_path, columns_path = paths
    
    merged_df = pd.DataFrame(columns = [id_col])
    merged_cols = []
    merged_time_col = None
    info_df = pd.read_csv(data_path + tables_path)
    
    for i in range(info_df.shape[0]):
        table_name =  info_df.iloc[i]["name"]
        table_df = pd.read_csv(data_path + info_df.iloc[i]["path"])
        
        table_df = table_df[table_df[id_col].isin(pat_ids)]
        table_time_col = info_df.iloc[i]["time_col"]
        is_static = info_df["time_col"].isnull().iloc[i]
        base_cols = [id_col]  
        
        if not is_static:
            table_df[time_col] = pd.to_datetime(table_df[table_time_col], infer_datetime_format=True)
            base_cols += [time_col]
        
        columns_file = info_df.iloc[i]["columns_file"]
        attributes_info = AttributesInfo(table_df, data_path + columns_path + columns_file)
        
        if not os.path.exists(example_path + ATTRIBUTES_PATH):
            os.makedirs(example_path + ATTRIBUTES_PATH)
        with open(example_path + ATTRIBUTES_PATH + "/" + table_name + ".pkl", 'wb') as files:
            pickle.dump(attributes_info, files)

        table_cols = attributes_info.names
        merged_cols += table_cols
        table_df_clean = table_df[base_cols + table_cols]
        table_encoded_df = encode_table_df(table_df_clean, attributes_info)

        if is_static:
            assert(merged_time_col is None)
            merged_df = merged_df.merge(table_encoded_df, on=[id_col], how = "outer")
        else:
            merging_cols = [id_col]
            if merged_time_col is not None:
                merging_cols += [time_col]
            merged_df = merged_df.merge(table_encoded_df, on=merging_cols, how = "outer")
            merged_time_col = time_col

        if imputer == "zero_imp":
            table_imputer = lambda x: x.fillna(0)
        elif imputer == "iai_imp":
            table_imp = get_iai_imputer(table_encoded_df[table_cols])
            table_imputer = lambda x: iai_impute_data(x, table_imp)
        
        if not os.path.exists(example_path + IMPUTERS_PATH +"_" + imputer):
            os.makedirs(example_path + IMPUTERS_PATH +"_" + imputer)
        with open(example_path + IMPUTERS_PATH + "_" + imputer + "/" + table_name + ".pkl", 'wb') as files:
            pickle.dump(table_imputer, files)

    if imputer == "zero_imp":
        global_imputer = lambda x: x.fillna(0)
    elif imputer == "iai_imp":
        imp = get_iai_imputer(merged_df[merged_cols])
        global_imputer = lambda x: iai_impute_data(x, imp)
    
    with open(example_path + IMPUTERS_PATH + "_" + imputer + "/global_imputer.pkl", 'wb') as files:
            pickle.dump(global_imputer, files)



def get_model_info(paths, id_col, time_col, imputer, pat_ids):
    """
    Loads the AttributedInfo object, the imputer function for each table, as well as as the global imputer for imputating
    merged tables.
    
    Parameters::
        paths = list of four paths [example_path, data_path, tables_path, columns_path]
            example_path: Path to the folder for the specific application.
            data_path: Path to the folder withall the data.
            tables_path: Path to the csv file with the information about each table (relative to data_path)
            columns_path: Path to the folder with the csv files containing information for the columns in each table.
        id_col: Name of the column containing the patient ids
        time_col: Name of the column containing the timestamps
        imputer: String indicating the type of imputer to use; one of "zero_imp" or "iai_imp".
        pat_ids: List of patient ids whose data will be used to save the model information (training data)
    """
    example_path, data_path, tables_path, columns_path = paths
    
    tables_info = []
    info_df = pd.read_csv(data_path + tables_path)
    
    for i in range(info_df.shape[0]):
        table = {}
        name = info_df.iloc[i]["name"]
        metadata = info_df.iloc[i]["metadata"]
        table_df = pd.read_csv(data_path + info_df.iloc[i]["path"])

        table_df = table_df[table_df[id_col].isin(pat_ids)]
        table_time_col = info_df.iloc[i]["time_col"]
        is_static = info_df["time_col"].isnull().iloc[i]
        base_cols = [id_col]
        
        if not is_static:
            table_df[time_col] = pd.to_datetime(table_df[table_time_col], infer_datetime_format=True)
            base_cols += [time_col]
        
        attributes_info = pd.read_pickle(example_path + ATTRIBUTES_PATH + "/" + name + ".pkl")
        table_imputer = pd.read_pickle(example_path + IMPUTERS_PATH + "_" + imputer + "/" + name + ".pkl")
        table_cols = attributes_info.names
        df = table_df[base_cols + table_cols]
        info = TableInfo(name, metadata, table_imputer, df, attributes_info)
        tables_info.append(info)
    
    global_imputer = pd.read_pickle(example_path + IMPUTERS_PATH + "_" + imputer + "/global_imputer.pkl")
    return tables_info, global_imputer
