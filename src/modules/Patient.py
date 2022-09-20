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
    """
    Patient module containing original, encoded, imputed, and text-embedded information for a specific id.
    
    pat_id: The patient id.
    tables: List of table objects with the tabular information for the given patient.
    time_col: Name of the column in df containing the timestamp for each observation.
    
    """
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

    def create_timed_data(self, prefix, missing, replace_nums, descriptive, meta, global_imp, feature_types):
        """
        Creates original, encoded, imputed, and text-embedded (timestamped) dataframes with the information of the given patient.
        
        Parameters::
            prefix: String containing the desired prefix to add at the beginning of each sentence ("", "the Patient", etc.)
            missing: String describing how to handle missing values (e.g. "", "is missing" "imp_replace") 
            replace_nums: Boolean indicating weather or not to replace numerical values with text (e.g. very low, high, normal)
            descriptive: Boolean indicating weather or not each sentence should be descriptive.
            meta: Boolean indicating weather or not to include meta information in the paragraphs.
            global_imp: Function used to impute missing values of the merged tables.
            sep_tables: Boolean indicating weather or not to impute numbers and embed text before merging the table objects.
            joint_tables: Boolean indicating weather or not to impute numbers and embed text after merging the table objects.
            
        """
        for table in self.tables:
            table.create_encoded_imputed_vectors()
            table.create_text(prefix, missing, replace_nums, descriptive, meta)

        reversed_tables = list(reversed(self.tables))
        text_table = reduce(lambda t1, t2: merge_text(t1, t2, self.time_col), reversed_tables)
        self.text = text_table.text.sort_values(by=[self.time_col])
        enc_table = reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "encodings"), reversed_tables)
        self.encodings = enc_table.encodings.sort_values(by=[self.time_col])
        

        if "joint_embeddings" in feature_types:
            
            self.joint_embeddings = create_embeddings(self.text).sort_values(by=[self.time_col])
            
        if "joint_imputations" in feature_types:
            joint_imputations = global_imp(self.encodings.drop([self.time_col], axis = 1))
            joint_imputations[self.time_col] = self.encodings[self.time_col]
            self.joint_imputations = joint_imputations.sort_values(by=[self.time_col])
                
        if "sep_embeddings" in feature_types:
            
            for table in self.tables:
                table.create_embeddings()
                    
            emb_table= reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "embeddings"), reversed_tables)
            self.sep_embeddings = emb_table.embeddings.sort_values(by=[self.time_col])
            
        if "sep_imputations" in feature_types:
  
            imp_table = reduce(lambda t1, t2: merge_tables(t1, t2, self.time_col, "imputations"), reversed_tables)
            self.sep_imputations = imp_table.imputations.sort_values(by=[self.time_col])



