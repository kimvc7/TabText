import datetime
from datetime import timedelta
import json
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import os.path
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, logging, LongformerModel, LongformerTokenizer, AutoModelForMaskedLM
import torch
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
import math

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="Wine", help="which uci dataset")
parser.add_argument("--job_num", type=int, default=0, help='which job to run')
parser.add_argument("--clinical", action='store_true', help="clinical llm or not")
parser.add_argument("--long", action='store_true', help='long input or not')
parser.add_argument("--biogpt", action='store_true', help='biogpt or not')
parser.add_argument("--imputer", type=str, default="zero_imp", help="which imputer to use")
parser.add_argument("--prefix", type=str, default="", help="which prefix to add")
parser.add_argument("--missing", type=str, default="", help="how to handle missing values")
parser.add_argument("--replace", action='store_true', help='replace numbers or not')
parser.add_argument("--descriptive", action='store_true', help='use descriptive language or not')
parser.add_argument("--meta", action='store_true', help='use metadata or not')

args = parser.parse_args()



with open('configs/config_' + args.data_set+ '.json') as config_file:
        UCI_config = json.load(config_file)

""
           #Load data and parameters
""
EXAMPLE_PATH = UCI_config["EXAMPLE_PATH"]
TABLES_FILE = UCI_config["TABLES_FILE"]
COLUMNS_PATH = UCI_config["COLUMNS_PATH"]
ID_COL = UCI_config["ID_COL"]
TIME_COL = None

sys.path.insert(0, './../../src')
from get_data_info import *
from get_patients import *
from get_features import *
        
                    
JOB_NUM = args.job_num

DATA_PATH = EXAMPLE_PATH + UCI_config["RAW_DATA_PATH"] 
paths = [EXAMPLE_PATH, DATA_PATH, TABLES_FILE, COLUMNS_PATH]


""
                #Choose settings
""
imputer = args.imputer
prefix = args.prefix
missing = args.missing
replace = args.replace
descriptive = args.descriptive
meta = args.meta
clinical = args.clinical
long = args.long
biogpt = args.biogpt



""
           #Get relevant patient ids
""
tables_info, global_imputer, all_ids = get_model_info(paths, ID_COL, TIME_COL, imputer, "Training", None, model_name=args.data_set)

##############################################
# Save patients info for desired feature types
# #############################################
llm_name = ""
original_llm_path = ""

if long:
    if clinical:
        original_llm_path = "./../../LLMs/ClinicalLongformer/"
        llm_name = "ClinicalLongformer"
    else:
        original_llm_path = "./../../LLMs/Longformer/"
        llm_name = "Longformer"
if biogpt:
    assert(long==False)
    assert(clinical==False)
    assert(finetuned==False)
    llm_name = "BioGPT"


""
           #Get relevant patient ids
""
##############################################
# Save patients info for desired feature types
# #############################################
folder_name = "Training/" + llm_name + "/" +args.data_set + "/"
get_and_save_pickle_patients(tables_info, ID_COL, TIME_COL, all_ids, prefix, missing, replace, descriptive, meta, global_imputer, folder_name, EXAMPLE_PATH, "RAW_DATA", clinical, long, biogpt, finetuned_path, ["text"])

sent_name = "RAW_DATA_" + str(prefix) +"_"+ str(missing) +"_"+ str(replace) +"_"+ str(descriptive) +"_"+ str(meta)

get_and_save_features(all_ids, TIME_COL, ID_COL, ["text"], None, folder_name, EXAMPLE_PATH, sent_name, job_id=(str(0)))


X_text = pd.read_csv(folder_name + "text/" + sent_name + "/Features/0.csv", index_col=0)
X_train = X_text[[ID_COL, "text"]]
fine_tune(X_train, original_llm_path, data_set, batch_size=4)
