import pandas as pd
import numpy as np
import sys

sys.path.insert(0, '../modules')
from Table import *


def merge_text(table1, table2, time_col):
    new_df = pd.DataFrame()
    new_time_col = time_col
    new_cols = [Column(TEXT_COL)]
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
        df = df.fillna("")
        df[TEXT_COL] = df[TEXT_COL + "_x"] + df[TEXT_COL + "_y"]
        new_df = df[[time_col, TEXT_COL]]
    table = Table("Merged Table", new_df, new_cols , "", new_time_col)
    return table


def merge_arrays(table1, table2, time_col, array_col):
    new_df = pd.DataFrame()
    new_time_col = time_col
    new_cols = [Column(array_col)]
    if table1.is_static() and table2.is_static():
        df = table1.df.copy()
        df[array_col] = np.concatenate((df1[array_col][0] ,  df2[array_col][0]))
        new_df = df[array_col]
        new_time_col = None
    elif table1.is_static():
        df = table2.df.copy()
        df[array_col] = df[array_col].apply(lambda x: np.concatenate((table1.df.iloc[0][array_col], x)))
        new_df = df[[time_col, array_col]]
    elif table2.is_static():
        df = table1.df.copy()
        df[array_col] = df[array_col].apply(lambda x: np.concatenate((x, table2.df.iloc[0][array_col])))
        new_df = df[[time_col, array_col]]
    else:
        df = table1.df.copy()
        df = df.merge(table2.df, how="outer", on=time_col)
        df[array_col + "_x"] = df[array_col + "_x"].apply(lambda d: d if isinstance(d, np.ndarray) else [])
        df[array_col + "_y"] = df[array_col + "_y"].apply(lambda d: d if isinstance(d, np.ndarray) else [])
        df[array_col] = [np.concatenate((df[array_col + "_x"][i], df[array_col + "_y"][i])) for i in range(len(df))]
        new_df = df[[time_col, array_col]]
    table = Table("Merged Table", new_df, new_cols, "", new_time_col)
    return table


