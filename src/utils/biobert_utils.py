import pandas as pd
from math import isnan
from transformers import AutoTokenizer, AutoModel, logging, LongformerModel, LongformerTokenizer, AutoModelForMaskedLM
import torch
import numpy as np
import json
import os

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../../config.json') as config_file:
        config = json.load(config_file)

biobert_path = config["BIOBERT_PATH"]
long_biobert_path = config["LONG_BIOBERT_PATH"]


def get_biobert_embeddings(text, long_input=True):
    # Inputs:
    #   text -> Input text (str)
    #
    # Outputs:
    #   embeddings -> Final Biobert embeddings with vector dimensionality = (1,768)
    #   hidden_embeddings -> Last hidden layer in Biobert model with vector dimensionality = (token_size,768)

    # %% EXAMPLE OF USE
    # embeddings, hidden_embeddings = get_biobert_embeddings(text)
    
    biobert_tokenizer = AutoTokenizer.from_pretrained(long_biobert_path + "tokenizer/")
    biobert_model = AutoModelForMaskedLM.from_pretrained(long_biobert_path + 'model', output_hidden_states=True)
    tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    
    if not long_input:
        biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
        biobert_model = AutoModel.from_pretrained(biobert_path)
        tokens_pt = biobert_tokenizer(text, return_tensors="pt")

    outputs = biobert_model(**tokens_pt)
    
    if long_input:
        hidden_embeddings = outputs.hidden_states[-1].detach().numpy()
        last_hidden_shape = hidden_embeddings.shape
        pooling = torch.nn.AvgPool2d([last_hidden_shape[1], 1])
        embeddings = pooling(outputs.hidden_states[-1])
        embeddings = torch.reshape(embeddings, (1, 768)).detach().numpy()
    else:
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_embeddings = last_hidden_state.detach().numpy()
        embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings

def create_embeddings(df):
    embeddings = []

    for i in range(df.shape[0]):
        text = df.iloc[i]["text"]
        full_embedding = get_biobert_embeddings(text)[0]
        embeddings.append(full_embedding.reshape(-1))
        
    emb_df =  pd.DataFrame(np.array(embeddings))
    emb_df = emb_df.set_index(df.index)
    merged_df = pd.concat([df, emb_df], axis=1)

    return merged_df.drop(columns= "text", axis=1)
