import pandas as pd
from math import isnan
from transformers import AutoTokenizer, AutoModel, logging
import torch
import numpy as np
import json

# Load config file with static parameters
with open('../../config.json') as config_file:
        config = json.load(config_file)

biobert_path = config["BIOBERT_PATH"]
TEXT_COL = config["TEXT_COL"]
EMB_COL = config["EMB_COL"]

biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
biobert_model = AutoModel.from_pretrained(biobert_path)

def get_biobert_embeddings(text):
    # Inputs:
    #   text -> Input text (str)
    #
    # Outputs:
    #   embeddings -> Final Biobert embeddings with vector dimensionality = (1,768)
    #   hidden_embeddings -> Last hidden layer in Biobert model with vector dimensionality = (token_size,768)

    # %% EXAMPLE OF USE
    # embeddings, hidden_embeddings = get_biobert_embeddings(text)

    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    outputs = biobert_model(**tokens_pt)
    last_hidden_state = outputs.last_hidden_state
    pooler_output = outputs.pooler_output
    hidden_embeddings = last_hidden_state.detach().numpy()
    embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings

def create_embeddings(df):
    embeddings = []

    for i in range(df.shape[0]):
        text = df.iloc[i][TEXT_COL]
        full_embedding = get_biobert_embeddings(text)[0]
        embeddings.append(full_embedding.reshape(-1))
        
    emb_df =  pd.DataFrame(np.array(embeddings))
    emb_df = emb_df.set_index(df.index)
    merged_df = pd.concat([df, emb_df], axis=1)

    return merged_df.drop(columns= TEXT_COL, axis=1)
