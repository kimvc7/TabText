from Column import *
import json
import sys
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]

sys.path.insert(0, DIR_NAME + 'TabText/src/utils')
from biobert_utils import *


class Table(object):
    """
    Table module containing tabular information for a specific id.
    
    name: String corresponding to the name of the table.
    df: Dataframe containing the tabular data for a specific id.
    columns: List of Column objects corresponing to the columns in df.
    metadata: String containing metadata information about this table structure.
    time_col: Name of the column in df containing the timestamp for each observation.
    imputer: Function used to impute the missing values in df. 
    
    """
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
    
        
    def create_encoded_imputed_vectors(self):
        """
        Creates encoded and imputed versions of the table contents.
        """
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
        

    def create_text(self, prefix, missing_word, replace_numbers, descriptive, meta, omit_empty = True, sep = "</s>"):
        """
        Creates a timestamped dataframe; each row contains a String (paragraph) with all the tabular information for the
        corresponding timestamp.
        
        Paramteres::
            prefix: String containing the desired prefix to add at the beginning of each sentence ("", "the Patient", etc.)
            missing_word: String describing how to handle missing values (e.g. "", "is missing" "imp_replace") 
            replace_numbers: Boolean indicating weather or not to replace numerical values with text (e.g. very low, high, normal)
            descriptive: Boolean indicating weather or not each sentence should be descriptive.
            meta: Boolean indicating weather or not to include meta information in the paragraphs.
            omit_empty: Boolean indicating weather or not to create a paragraph for empty tables
            sep: String indicating what symbol to use at the end of the paragraph as a separator between tables.
        """
        self.text = pd.DataFrame()
        if self.is_temporal():
            self.text[self.time_col] = self.df[self.time_col]
        text = []
        
        if self.is_empty & omit_empty:
            text.append("")
            
        else:
            for t_i in range(self.df.shape[0]):
                text_i = ""
                if meta & (len(str(self.metadata)) >1):
                    text_i = self.metadata

                for column in self.columns:
                    value = self.df.iloc[t_i][column.name]
                    imp_value = self.imputations.iloc[t_i][column.name]
                    col_text = column.create_sentence(value, imp_value, prefix, missing_word, replace_numbers, descriptive)
                    if len(col_text) >0:
                        col_text += ", "
                    text_i += col_text
                text_i = text_i[:-2]+ ". " + sep    
                text.append(text_i)

        self.text["text"] =  text
    
    def create_embeddings(self):
        """
        Creates a timestamped dataframe; each row contains NLP embeddings for the paragraph of the corresponding timestamp.
        """
        embeddings = []

        for i in range(self.text.shape[0]):
            text = self.text.iloc[i]["text"]
            full_embedding = get_biobert_embeddings(text)[0]
            embeddings.append(full_embedding.reshape(-1))
        
        emb_df =  pd.DataFrame(np.array(embeddings))
        emb_df = emb_df.set_index(self.text.index)
        merged_df = pd.concat([self.text, emb_df], axis=1)
        merged_df = merged_df.rename({i: self.name + "_" + str(i) for i in range(len(embeddings[0]))}, axis='columns')

        self.embeddings = merged_df.drop(["text"], axis = 1)
        
