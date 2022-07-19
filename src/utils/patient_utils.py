import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '../modules')
from Table import *

# Load config file with static parameters
with open('../../config.json') as config_file:
    config = json.load(config_file)
    
TEXT_COL = config["TEXT_COL"]

def merge_text(table1, table2, time_col):
    merged_df = pd.DataFrame()
    merged_time_col = time_col
    
    df1 = table1.text.copy()
    df2 = table2.text.copy()

    if table1.is_static() and table2.is_static():
        df = table1.df.copy()
        df1[TEXT_COL] = df1[TEXT_COL] + df2[TEXT_COL]
        merged_df = df1
        merged_time_col = None

    elif table1.is_static():
        df2[TEXT_COL] = df1.iloc[0][TEXT_COL] + df2[TEXT_COL]
        merged_df = df2
    
    elif table2.is_static():
        df1[TEXT_COL] = df1[TEXT_COL] + df2.iloc[0][TEXT_COL]
        merged_df = df1
    else:
        df = df1.merge(df2, how="outer", on=time_col)
        df = df.fillna("")
        df[TEXT_COL] = df[TEXT_COL + "_x"] + df[TEXT_COL + "_y"]
        merged_df = df[[time_col, TEXT_COL]]
    
    table = Table(time_col = merged_time_col)
    table.text =  merged_df

    return table



def merge_tables(table1, table2, time_col, table_attribute):
    merged_df = pd.DataFrame()
    merged_time_col = time_col
    
    df1 = getattr(table1, table_attribute).copy()
    df2 = getattr(table2, table_attribute).copy()


    if table1.is_static() and table2.is_static():
        merged_df = pd.concat([df1.reset_index(drop=True),df2.reset_index(drop=True)], axis=1)
        merged_time_col = None

    elif table1.is_static():
        repeated_df1 = df1.loc[df1.index.repeat(df2.shape[0])].reset_index()
        df2[df1.columns] = repeated_df1[df1.columns]
        merged_df = df2

    elif table2.is_static():
        repeated_df2 = df2.loc[df2.index.repeat(df1.shape[0])].reset_index()
        df1[df2.columns] = repeated_df2[df2.columns]
        merged_df = df1
    
    else:
        merged_df = df1.merge(df2, on=[time_col], how="outer")

    table = Table(time_col = merged_time_col)
    setattr(table, table_attribute, merged_df)

    return table




