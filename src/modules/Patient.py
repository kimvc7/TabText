from Column import *
from Table import *
import functools
from functools import reduce
import json
import sys

# Load config file with static parameters
with open('../../config.json') as config_file:
        config = json.load(config_file)

TEXT_COL = config["TEXT_COL"]
EMB_COL = config["EMB_COL"]
WEIGHT_COL = config["WEIGHT_COL"]
DIR_NAME = config["DIRECTORY_NAME"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')
from patient_utils import *
from biobert_utils import *

class Patient(object):
    def __init__(self, tables, pat_id, time_col):
        self.id = pat_id
        self.time_col = time_col
        self.tables = tables

    def get_tables_name(self):
        table_names = []
        for table in self.tables:
            table_names.append(table.name)
        return table_names

    def create_timed_data(self, prefix, missing_word, replace_numbers, descriptive, merge_tables_text=True):
        for table in self.tables:
            table.create_text(prefix, missing_word, replace_numbers, descriptive)
            
        timed_data = reduce(lambda t1, t2: merge_text(t1, t2, self.time_col), self.tables).df
        
        if merge_tables_text:
            timed_data[EMB_COL] = create_embeddings(timed_data)
        
        else:
            for table in self.tables:
                table.create_embeddings()

            emb_data = reduce(lambda t1, t2: merge_emb(t1, t2, self.time_col), self.tables).df
            timed_data[EMB_COL + "_per_table"] = emb_data[EMB_COL]
        self.timed_data = timed_data
        return timed_data
    
    def get_timebounded_embeddings(self, weight_fn, start_hr = None, end_hr = None, merge_tables_text=True):
        timebounded_df = self.timed_data.copy()

        if merge_tables_text:
            timebounded_df = timebounded_df[[EMB_COL, self.time_col]]
        else:
            timebounded_df = timebounded_df[[EMB_COL + "_per_table", self.time_col]]

        if start_hr is not None:
            timebounded_df = timebounded_df[timebounded_df[self.time_col]>= start_hr]
        if end_hr is not None:
            timebounded_df = timebounded_df[timebounded_df[self.time_col]<= end_hr]

        timebounded_df[WEIGHT_COL] = weight_fn(timebounded_df[self.time_col])
        return timebounded_df
