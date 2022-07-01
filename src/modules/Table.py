from Column import *
import json
import sys

# Load config file with static parameters
with open('../../config.json') as config_file:
	config = json.load(config_file)

TEXT_COL = config["TEXT_COL"]
EMB_COL = config["EMB_COL"]
DIR_NAME = config["DIRECTORY_NAME"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')
from biobert_utils import *

#One table per patient per tabular data structure
class Table(object):
    def __init__(self, name, df, columns, metadata, time_col):
        self.name = name
        self.headers = df.columns
        self.columns = columns
        self.metadata = metadata
        self.df = df
        self.time_col = time_col

    def is_temporal(self):
        return self.time_col is not None
    
    def is_static(self):
        return self.time_col is None

        
    def create_text(self, prefix, missing_word, replace_numbers, descriptive):
        text = []
        for t_i in range(self.df.shape[0]):
            text_i = self.metadata
            
            for column in self.columns:
                print
                value = self.df.iloc[t_i][column.name]
                col_text = column.create_sentence(value, prefix, missing_word, replace_numbers, descriptive)
                if len(col_text) >0:
                    col_text += ", "
                text_i += col_text
            text_i = text_i[:-2]+ ". "    
            text.append(text_i)
    
        self.df[TEXT_COL] = text 

    
    def create_embeddings(self):
        embeddings = []

        for i in range(self.df.shape[0]):
            text = self.df.iloc[i][TEXT_COL]
            full_embedding = get_biobert_embeddings(text)[0]
            embeddings.append(full_embedding.reshape(-1))

        self.df[EMB_COL] = embeddings
        
    def get_timebounded_df(self, start_hr, end_hr):
        
        if self.time_col is None:
            return self.df
        
        else:
            timebounded_df = self.df.copy()

            if start_hr is not None:
                timebounded_df = timebounded_df[timebounded_df[time_col]>= start_hr]
            if end_hr is not None:
                timebounded_df = timebounded_df[timebounded_df[time_col]<= end_hr]

            return timebound_df
    
