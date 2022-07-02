from Column import *
import json
import sys

# Load config file with static parameters
with open('../../config.json') as config_file:
	config = json.load(config_file)

TEXT_COL = config["TEXT_COL"]
EMB_COL = config["EMB_COL"]
DIR_NAME = config["DIRECTORY_NAME"]
ENC_COL = config["ENC_COL"]
IMP_COL = config["IMP_COL"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')
from biobert_utils import *

#One table per patient per tabular data structure
class Table(object):
    def __init__(self, name, df, columns, metadata, time_col):
        self.name = name
        self.columns = columns
        self.metadata = metadata
        self.time_col = time_col
        self.df = df[self.get_column_names()]

    def is_temporal(self):
        return self.time_col is not None
    
    def is_static(self):
        return self.time_col is None

    def get_column_names(self):
        col_names = []
        for column in self.columns:
            col_names.append(column.name)
        if self.time_col is not None:
            col_names.append(self.time_col)
        return col_names


    def create_encoded_imputed_vectors(self, impute_fn = None):
        encoded_df =  pd.DataFrame()
  
        for column in self.columns:
            col_values = self.df[column.name]
            col_encoder = column.encode_fn
            labels = col_encoder(col_values[col_values.notnull()])
            encoded_df[column.name] = col_values
            encoded_df[column.name] = pd.Series(labels, index=col_values[col_values.notnull()].index)

        self.df[ENC_COL] = [np.array(encoded_df.iloc[i]) for i in range(self.df.shape[0])]
        if impute_fn is not None:
            imputed_df = impute_fn(encoded_df)
            self.df[IMP_COL] = [np.array(imputed_df[i]) for i in range(self.df.shape[0])]
                

        
    def create_text(self, prefix, missing_word, replace_numbers, descriptive):
        text = []
        for t_i in range(self.df.shape[0]):
            text_i = self.metadata
            
            for column in self.columns:
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
    
