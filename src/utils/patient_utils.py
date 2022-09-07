import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + '/../modules')
from Table import *

def merge_text(table1, table2, time_col):
    """
    Parameters::
        table1, table2: Table objects to be merged (in that order)
        time_col: Name of the column containing the timestamps for each table.
    
    Returns::
        table: Table object whose .text dataframe is the result of merging table1.text and table2.text.
        1) If one of the tables is static, its static text is concatenated to the text in each row of the other table.
        2) if none of the tables are static, texts with the same timestamps in table1 and table2 are merged into a single string. 
    """
    merged_df = pd.DataFrame()
    merged_time_col = time_col
    
    df1 = table1.text.copy()
    df2 = table2.text.copy()

    if table1.is_static() and table2.is_static():
        df = table1.df.copy()
        df1["text"] = df1["text"] + df2["text"]
        merged_df = df1
        merged_time_col = None

    elif table1.is_static():
        df2["text"] = df1.iloc[0]["text"] + df2["text"]
        merged_df = df2
    
    elif table2.is_static():
        df1["text"] = df1["text"] + df2.iloc[0]["text"]
        merged_df = df1
    else:
        df = df1.merge(df2, how="outer", on=time_col)
        df = df.fillna("")
        df["text"] = df["text" + "_x"] + df["text" + "_y"]
        merged_df = df[[time_col, "text"]]
    
    table = Table(time_col = merged_time_col)
    table.text =  merged_df

    return table



def merge_tables(table1, table2, time_col, table_attribute):
    """
    Parameters::
        table1, table2: Table objects to be merged (in that order)
        time_col: Name of the column containing the timestamps for each table.
        table_attribute: String with the name of a table object attribute (e.g. "encodings", "imputations" or "embeddings")
    
    Returns::
        table: Table object whose dataframe attribute 'table_attribute' is the result of merging the corresponding attribute from
        table1 and table2
        1) If one of the tables is static, its single-row table_attribute is concatenated to each row in the table_attribute of
        the other table.
        2) If none of the tables are static, table_attributes with the same timestamps in table1 and table2 are concatenated. 
    """
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




