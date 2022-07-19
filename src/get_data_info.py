import datetime
from datetime import timedelta
import pandas as pd
import sys
import json
import pickle
import os

with open('../../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]
ATTRIBUTES_PATH = config["ATTRIBUTES_PATH"]
IMPUTERS_PATH = config["IMPUTERS_PATH"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')

import dill as pickle
from data_utils import *

class Attribute(object):
     def __init__(self, attribute, verb, neg_verb, col_type, avg, sd, encoder):
         self.attribute = attribute
         self.column_verb = verb
         self.column_neg_verb = neg_verb
         self.column_type = col_type
         self.avg = avg
         self.sd = sd
         self.column_encoder = encoder
        
class AttributesInfo(object):
    def __init__(self, df, info_file_path, create_encoder_fn=get_label_encoder):
        self.info_file = pd.read_csv(info_file_path)
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
                col_encoder = create_encoder_fn(df[col_name])
                enc = lambda x: encode_categorical(x, col_encoder)

            if col_type == "numerical":
                col_values = df[[col_name]].astype(np.float)
                col_values = col_values[col_name][pd.notnull(col_values[col_name])]
                avg = col_values.mean()
                sd = col_values.std()
                enc = encode_numerical
            setattr(self, col_name,  Attribute(attribute, verb, neg_verb, col_type, avg, sd, enc))

class TableInfo(object):
    def __init__(self, name, metadata, imputer, df, attributes_info):
        self.name = name
        self.metadata = metadata
        self.imputer = imputer
        self.df = df
        self.attributes_info = attributes_info

def encode_table_df(table_df, attributes_info):
    encoded_df = table_df.copy()
    cols_to_encode = attributes_info.names
    
    for col in cols_to_encode:
        col_values = table_df[col]
        col_encoder = getattr(attributes_info, col).column_encoder
        labels = col_encoder(col_values[col_values.notnull()])
        encoded_df[col] = pd.Series(labels, index=col_values[col_values.notnull()].index)
    
    return encoded_df

def get_filtered_ids(filter_map, df, id_col):
    new_df = df.copy()
    for col in filter_map:
        new_df = new_df[new_df[col].isin(filter_map[col])]
    return new_df[id_col].unique()


def save_model_info(tables_path, col_path, id_col, time_col, imputer, train_start = None, train_end = None, filtered_ids = None):
    merged_df = pd.DataFrame(columns = [id_col])
    merged_cols = []
    merged_time_col = None
    info_df = pd.read_csv(tables_path)
    
    for i in range(info_df.shape[0]):
        table_name =  info_df.iloc[i]["name"]
        table_df = pd.read_csv(info_df.iloc[i]["path"])
        
        if filtered_ids is not None:
            table_df = table_df[table_df[id_col].isin(filtered_ids)]
        table_time_col = info_df.iloc[i]["time_col"]
        is_static = info_df["time_col"].isnull().iloc[i]
        base_cols = [id_col]

        if not is_static:
            table_df[time_col] = pd.to_datetime(table_df[table_time_col], infer_datetime_format=True)
            base_cols += [time_col]
            
            if train_start is not None:
                table_df = table_df[table_df[time_col].dt.date >= train_start]
            if train_end is not None:
                table_df = table_df[table_df[time_col].dt.date < train_end]
            
        
        columns_file = info_df.iloc[i]["columns_file"]
        attributes_info = AttributesInfo(table_df, col_path + columns_file)
        
        if not os.path.exists(ATTRIBUTES_PATH):
            os.makedirs(ATTRIBUTES_PATH)
        with open(ATTRIBUTES_PATH + "/" + table_name + ".pkl", 'wb') as files:
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

        if imputer == "zero":
            table_imputer = lambda x: x.fillna(0)
        elif imputer == "iai":
            table_imp = get_iai_imputer(table_encoded_df[table_cols])
            table_imputer = lambda x: iai_impute_data(x, table_imp)
        
        if not os.path.exists(IMPUTERS_PATH):
            os.makedirs(IMPUTERS_PATH)
        with open(IMPUTERS_PATH + "/" + table_name + ".pkl", 'wb') as files:
            pickle.dump(table_imputer, files)

    if imputer == "zero":
        global_imputer = lambda x: x.fillna(0)
    elif imputer == "iai":
        imp = get_iai_imputer(merged_df[merged_cols])
        global_imputer = lambda x: iai_impute_data(x, imp)
    
    with open(IMPUTERS_PATH + "/global_imputer.pkl", 'wb') as files:
            pickle.dump(global_imputer, files)



def get_model_info(tables_path, id_col, time_col, start_date=None, end_date=None, filtered_ids=None):
    pat_ids = set()
    tables_info = []
    info_df = pd.read_csv(tables_path)
    
    for i in range(info_df.shape[0]):
        table = {}
        name = info_df.iloc[i]["name"]
        metadata = info_df.iloc[i]["metadata"]
        table_df = pd.read_csv(info_df.iloc[i]["path"])
        
        if filtered_ids is not None:
            table_df = table_df[table_df[id_col].isin(filtered_ids)]
        
        table_time_col = info_df.iloc[i]["time_col"]
        is_static = info_df["time_col"].isnull().iloc[i]
        base_cols = [id_col]
        
        if not is_static:
            table_df[time_col] = pd.to_datetime(table_df[table_time_col], infer_datetime_format=True)
            base_cols += [time_col]
            
            if start_date is not None:
                table_df = table_df[table_df[time_col].dt.date >= start_date]
            if end_date is not None:
                table_df = table_df[table_df[time_col].dt.date < end_date]
            
            pat_ids = pat_ids.union(set(table_df[id_col].unique()))
        
        attributes_info = pd.read_pickle(ATTRIBUTES_PATH + "/" + name + ".pkl")
        imputer = pd.read_pickle(IMPUTERS_PATH + "/" + name + ".pkl")
        table_cols = attributes_info.names
        df = table_df[base_cols + table_cols]
        info = TableInfo(name, metadata, imputer, df, attributes_info)
        tables_info.append(info)
    
    global_imputer = pd.read_pickle(IMPUTERS_PATH + "/global_imputer.pkl")
    return tables_info, pat_ids, global_imputer








