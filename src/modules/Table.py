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
    def __init__(self, name="", df=pd.DataFrame(), columns=None, metadata=None, time_col=None, imputer=None):
        self.name = name
        self.columns = columns
        self.metadata = metadata
        self.time_col = time_col
        self.df = df
        self.imputer = imputer
        self.is_empty = pd.isna(df).all().all()
        
    def is_temporal(self):
        return self.time_col is not None
    
    def is_static(self):
        return self.time_col is None      
    
        
    def create_text(self, prefix, missing_word, replace_numbers, descriptive, omit_empty = True):
        self.text = pd.DataFrame()
        if self.is_temporal():
            self.text[self.time_col] = self.df[self.time_col]
        text = []
        
        if self.is_empty & omit_empty:
            text.append("")
            
        else:
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

        self.text[TEXT_COL] =  text
        
    def create_encoded_imputed_vectors(self):
        encoded_df =  pd.DataFrame()
        
        for column in self.columns:
            col_values = self.df[column.name]
            col_encoder = column.encode_fn
            labels = col_encoder(col_values[col_values.notnull()])
            encoded_df[column.name] = col_values
            encoded_df[column.name] = pd.Series(labels, index=col_values[col_values.notnull()].index)
        
        if self.imputer is not None:
            self.imputations = self.imputer(encoded_df)
            if self.is_temporal():
                self.imputations[self.time_col] = self.df[self.time_col]
       
        self.encodings = encoded_df
        if self.is_temporal():
            self.encodings[self.time_col] = self.df[self.time_col]
        
    
    def create_embeddings(self):
        embeddings = []

        for i in range(self.text.shape[0]):
            text = self.text.iloc[i][TEXT_COL]
            full_embedding = get_biobert_embeddings(text)[0]
            embeddings.append(full_embedding.reshape(-1))
        
        emb_df =  pd.DataFrame(np.array(embeddings))
        emb_df = emb_df.set_index(self.text.index)
        merged_df = pd.concat([self.text, emb_df], axis=1)
        merged_df = merged_df.rename({i: self.name + "_" + str(i) for i in range(len(embeddings[0]))}, axis='columns')

        self.embeddings = merged_df.drop([TEXT_COL], axis = 1)
        
