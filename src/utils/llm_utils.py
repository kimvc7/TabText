import pandas as pd
from math import isnan
from transformers import AutoTokenizer, AutoModel, logging, LongformerModel, LongformerTokenizer, AutoModelForMaskedLM, BioGptTokenizer, BioGptForCausalLM, LongformerTokenizer
import torch
import numpy as np
import json
import os
import datetime
from datetime import timedelta
import sys
from sklearn.model_selection import train_test_split
import os.path
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, logging, LongformerModel, LongformerTokenizer, AutoModelForMaskedLM
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import math

# Load config file with static parameters
with open(os.path.dirname(__file__) + '/../../config.json') as config_file:
        config = json.load(config_file)

biogpt_path = config["BIOGPT_PATH"]
clinical_long_path = config["CLINICAL_LONGFORMER_PATH"]
longformer_path = config["LONGFORMER_PATH"]

def get_llm_embeddings(text, long_input=True, finetuned_path="", biogpt=False, clinical=True):
    """
    Parameters::
        text: String with input text
        long_input: Boolean indicating weather to use Clinical LongFormer or Clinical Bert
        finetuned_path: Path for finetuned model
        biogpt: Boolean indicating if embeddings should be computed using BioGPT
        clinical: Boolean indicating if embeddings should be computed using model pretrained on clinical notes
        
    Returns::
        embeddings: Final Biobert embeddings with vector dimensionality = (1,768)
        hidden_embeddings: Last hidden layer in Biobert model with vector dimensionality = (token_size, 768)
    """

    if long_input:
        if clinical:
            long_biobert_tokenizer = AutoTokenizer.from_pretrained(clinical_long_path + "tokenizer/")
            if finetuned_path != "":
                model = AutoModelForMaskedLM.from_pretrained(finetuned_path, output_hidden_states=True)
                assert(biogpt == False)
                assert(long_input == True)
                assert(clinical == True)
            else:
                model = AutoModelForMaskedLM.from_pretrained(clinical_long_path + 'model', output_hidden_states=True)
            tokens_pt = long_biobert_tokenizer(text, return_tensors="pt", truncation=True)
        if not clinical:
            assert(not fine_tuned)
            longformer_tokenizer = LongformerTokenizer.from_pretrained(longformer_path)
            model = AutoModelForMaskedLM.from_pretrained(longformer_path, output_hidden_states=True)
            tokens_pt = longformer_tokenizer(text, return_tensors="pt", truncation=True)
    
    if not long_input and not biogpt:
        biobert_tokenizer = AutoTokenizer.from_pretrained(biobert_path)
        model = AutoModel.from_pretrained(biobert_path)
        tokens_pt = biobert_tokenizer(text, return_tensors="pt")
    
    if biogpt:
        biogpt_tokenizer = BioGptTokenizer.from_pretrained(biogpt_path)
        model = BioGptForCausalLM.from_pretrained(biogpt_path, output_hidden_states=True)
        tokens_pt = biogpt_tokenizer(text, return_tensors="pt")

    outputs = model(**tokens_pt)
    
    if long_input or biogpt:
        hidden_embeddings = outputs.hidden_states[-1].detach().numpy()
        last_hidden_shape = hidden_embeddings.shape
        pooling = torch.nn.AvgPool2d([last_hidden_shape[1], 1])
        embeddings = pooling(outputs.hidden_states[-1])
        dimension = last_hidden_shape[2]
        embeddings = torch.reshape(embeddings, (1, dimension)).detach().numpy()

    else:
        last_hidden_state = outputs.last_hidden_state
        pooler_output = outputs.pooler_output
        hidden_embeddings = last_hidden_state.detach().numpy()
        embeddings = pooler_output.detach().numpy()

    return embeddings, hidden_embeddings

def create_embeddings(df):
    """
    Parameters::
        df: DataFrame with a column named "text"
        
    Returns::
        merged_df: DataFrame with 768 columns; each row contains the embeddings for the text in the corresponding row of df.
    """
    embeddings = []

    for i in range(df.shape[0]):
        text = df.iloc[i]["text"]
        full_embedding = get_llm_embeddings(text)[0]
        embeddings.append(full_embedding.reshape(-1))
        
    emb_df =  pd.DataFrame(np.array(embeddings))
    emb_df = emb_df.set_index(df.index)
    merged_df = pd.concat([df, emb_df], axis=1)

    return merged_df.drop(columns= "text", axis=1)


def fine_tune(X_train, original_llm_path, data_set, batch_size):
    """
    Parameters::
        X_train: Dataframe with 'id' and 'text' columns for finetuning
        original_llm_path: The path of the original llm model that is going to be finetuned
        data_set: name of the data set where the text data is coming from
        batch_size: batch size to be used for finetuning
        
    Returns::
        merged_df: DataFrame with 768 columns; each row contains the embeddings for the text in the corresponding row of df.
    """
    X_train = X_train.reset_index()
    X_train['id'] = X_train.index
    X_train = X_train[['id', 'text']]


    dataset = Dataset.from_pandas(X_train, split='train')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForMaskedLM.from_pretrained(original_llm_path + 'model').to(device)
    tokenizer = AutoTokenizer.from_pretrained(original_llm_path + "tokenizer/")
    
    def tokenize_function(batched_data):
        result = tokenizer(batched_data['text'], padding='max_length', truncation=True, max_length=1024)
        if tokenizer.is_fast:
            result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
        return result

    def group_texts(batched_data):
        concatenated_examples = {k: sum(batched_data[k], []) for k in batched_data.keys()}
        total_length = len(concatenated_examples[list(batched_data.keys())[0]])
        total_length = (total_length // chunk_size) * chunk_size
        result = {k : [t[i: i+chunk_size] for i in range(0, total_length, chunk_size)] for k, t in concatenated_examples.items()}
        result['labels'] = result['input_ids'].copy()
        return result
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['text', 'id'])

    chunk_size = 1024
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    dataset_split = lm_datasets.train_test_split(test_size=0.1)

    batch_size = batch_size
    
    # Show the training loss with every epoch
    logging_steps = len(dataset_split["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir= data_set + "_finetuned",
        overwrite_output_dir=False,
        evaluation_strategy="epoch",
        num_train_epochs=7,
        learning_rate=2e-5,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        push_to_hub=False,
        fp16=True,
        logging_steps=logging_steps,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_split["train"],
        eval_dataset=dataset_split["test"],
        data_collator=data_collator,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    trainer.save_model()
