import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '../modules')
from Table import *

def merge_text(table1, table2, time_col):
    new_df = pd.DataFrame()
    new_time_col = time_col
    if table1.is_static() and table2.is_static():
        df = table1.df.copy()
        df[TEXT_COL] = table1.df[TEXT_COL] + table2.df[TEXT_COL]
        new_df = df[TEXT_COL]
        new_time_col = None
    elif table1.is_static():
        df = table2.df.copy()
        df[TEXT_COL] = table1.df.iloc[0][TEXT_COL] + table2.df[TEXT_COL]
        new_df = df[[time_col, TEXT_COL]]
    elif table2.is_static():
        df = table1.df.copy()
        df[TEXT_COL] = table1.df[TEXT_COL] + table2.df.iloc[0][TEXT_COL]
        new_df = df[[time_col, TEXT_COL]]
    else:
        df = table1.df.copy()
        df = df.merge(table2.df, how="outer", on=time_col)
        print()
        df = df.fillna("")
        df[TEXT_COL] = df[TEXT_COL + "_x"] + df[TEXT_COL + "_y"]
        new_df = df[[time_col, TEXT_COL]]
    table = Table("Merged Table", new_df, table1.columns + table2.columns, "", new_time_col)
    return table


def merge_emb(table1, table2, time_col):
    new_df = pd.DataFrame()
    new_time_col = time_col
    if table1.is_static() and table2.is_static():
        df = table1.df.copy()
        df[EMB_COL] = np.concatenate((df1[EMB_COL][0] ,  df2[EMB_COL][0]))
        new_df = df[EMB_COL]
        new_time_col = None
    elif table1.is_static():
        df = table2.df.copy()
        df[EMB_COL] = df[EMB_COL].apply(lambda x: np.concatenate((table1.df.iloc[0][EMB_COL], x)))
        new_df = df[[time_col, EMB_COL]]
    elif table2.is_static():
        df = table1.df.copy()
        df[EMB_COL] = df[EMB_COL].apply(lambda x: np.concatenate((x, table2.df.iloc[0][EMB_COL])))
        new_df = df[[time_col, EMB_COL]]
    else:
        df = table1.df.copy()
        df = df.merge(table2.df, how="outer", on=time_col)
        df[EMB_COL + "_x"] = df[EMB_COL + "_x"].apply(lambda d: d if isinstance(d, np.ndarray) else [])
        df[EMB_COL + "_y"] = df[EMB_COL + "_y"].apply(lambda d: d if isinstance(d, np.ndarray) else [])
        df[EMB_COL] = [np.concatenate((df[EMB_COL + "_x"][i], df[EMB_COL + "_y"][i])) for i in range(len(df))]
        new_df = df[[time_col, EMB_COL]]
    table = Table("Merged Table", new_df, table1.columns + table2.columns, "", new_time_col)
    return table
