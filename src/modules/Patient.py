from Column import *
from Table import *
import functools
from functools import reduce
import json
import sys
import numpy as np
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../../config.json') as config_file:
        config = json.load(config_file)

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

    def create_timed_data(self, prefix, missing, replace_numbers, descriptive, global_imp, sep_tables=True, join_tables=True):
        
        for table in self.tables:
            table.create_text(prefix, missing, replace_numbers, descriptive)
            table.create_encoded_imputed_vectors()


        text_table = reduce(lambda t1, t2: merge_text(t1, t2, self.time_col), self.tables)
        self.text = text_table.text
        enc_table = reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "encodings"), self.tables)
        self.encodings = enc_table.encodings
        

        if join_tables:
            
            self.joint_embeddings = create_embeddings(self.text)
            
            joint_imputations = global_imp(self.encodings.drop([self.time_col], axis = 1))
            joint_imputations[self.time_col] = self.encodings[self.time_col]
            self.joint_imputations = joint_imputations
                
        if sep_tables:
            
            for table in self.tables:
                table.create_embeddings()
                    
            emb_table= reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "embeddings"), self.tables)
            self.sep_embeddings = emb_table.embeddings
  
            imp_table = reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "imputations"), self.tables)
            self.sep_imputations = imp_table.imputations



