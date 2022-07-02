from Column import *
from Table import *
import functools
from functools import reduce
import json
import sys
import numpy as np

# Load config file with static parameters
with open('../../config.json') as config_file:
        config = json.load(config_file)

TEXT_COL = config["TEXT_COL"]
EMB_COL = config["EMB_COL"]
ENC_COL = config["ENC_COL"]
IMP_COL = config["IMP_COL"]
DIR_NAME = config["DIRECTORY_NAME"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')
from patient_utils import *
from biobert_utils import *
from data_utils import *

class Patient(object):
    def __init__(self, tables, pat_id, time_col):
        self.id = pat_id
        self.time_col = time_col
        self.tables = tables
        self.timed_data = pd.DataFrame()

    def get_tables_name(self):
        table_names = []
        for table in self.tables:
            table_names.append(table.name)
        return table_names

    def create_timed_data(self, prefix, missing_word, replace_numbers, descriptive, separate_tables=True, join_tables=True, include_vec=True, include_emb=True, imputer_fn=np.nan_to_num, imputer_map = None):
        
        for table in self.tables:
            if include_emb:
                table.create_text(prefix, missing_word, replace_numbers, descriptive)
            if include_vec and (imputer_fn is not None):
                if separate_tables:
                     table.create_encoded_imputed_vectors(imputer_fn)
                     #table.create_encoded_imputed_vectors(imputer_map[table.name])
                else:
                    table.create_encoded_imputed_vectors()


        text_data = reduce(lambda t1, t2: merge_text(t1, t2, self.time_col), self.tables).df
        self.timed_data = text_data

        enc_data = reduce(lambda t1, t2: merge_arrays(t1, t2, self.time_col, ENC_COL), self.tables).df
        self.timed_data[ENC_COL] = enc_data[ENC_COL]
        
        if include_emb:
            if join_tables:
                self.timed_data[EMB_COL + "_joint_tables"] = create_embeddings(self.timed_data)
            if separate_tables:
                for table in self.tables:
                    table.create_embeddings()

                emb_data = reduce(lambda t1, t2: merge_arrays(t1, t2, self.time_col, EMB_COL), self.tables).df
                self.timed_data[EMB_COL + "_separate_tables"] = emb_data[EMB_COL]
        
        if include_vec:
            if join_tables:
                self.timed_data[IMP_COL + "_joint_tables"] = impute_data(self.timed_data, imputer_fn)
            if separate_tables:
                for table in self.tables:
                    table.create_embeddings()

                imp_data = reduce(lambda t1, t2: merge_arrays(t1, t2, self.time_col, IMP_COL), self.tables).df
                self.timed_data[IMP_COL + "_separate_tables"] = imp_data[IMP_COL]

    
    def get_timebounded_embeddings(self, start_hr = None, end_hr = None):
        timebounded_df = self.timed_data.copy()

        if start_hr is not None:
            timebounded_df = self.timed_data[self.timed_data[self.time_col]>= start_hr]
        if end_hr is not None:
            timebounded_df = self.timed_data[self.timed_data[self.time_col]<= end_hr]

        return timebounded_df
