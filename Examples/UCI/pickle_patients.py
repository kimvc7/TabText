import datetime
from datetime import timedelta
import json
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")
import os.path
from pathlib import Path

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="Wine", help="which uci dataset")
parser.add_argument("--job_num", type=int, default=0, help='which job to run')
parser.add_argument("--clinical", action='store_true', help="clinical llm or not")
parser.add_argument("--long", action='store_true', help='long input or not')
parser.add_argument("--biogpt", action='store_true', help='biogpt or not')
parser.add_argument("--finetuned", action='store_true', help='used finetuned model or not')
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
TARGET_FILE = UCI_config["TARGET_INFO_FILE"]
TARGET_COL = UCI_config["TARGET_COL"]
split_seed = UCI_config["TARGET_SPLIT_SEED"]
split_ratio = UCI_config["TEST_SPLIT_RATIO"]
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
finetuned = args.finetuned

if finetuned:
    finetuned_path = EXAMPLE_PATH + args.data_set + "_finetuned/"
else:
    finetuned_path = ""
    
feature_types = ["text_per_col", "sep_embeddings", "sep_imputations"]
llm_name = ""

if long:
    if clinical:
        if finetuned:
            llm_name = "ClinicalLongformerFinetuned"
            finetuned_path = args.data_set + "_finetuned"
        else:
            llm_name = "ClinicalLongformer"
    else:
        if finetuned:
            llm_name = "LongformerFinetuned"
            finetuned_path = args.data_set + "_finetuned"
        else:
            llm_name = "Longformer"
if biogpt:
    assert(long==False)
    assert(clinical==False)
    assert(finetuned==False)
    llm_name = "BioGPT"


for JOB_SET in ["Training", "Testing"]:
    tables_info, global_imputer, all_ids = get_model_info(paths, ID_COL, TIME_COL, imputer, JOB_SET, None, model_name=args.data_set)
    JOB_LENGTH = int(len(all_ids)/5)+1
    pats = [all_ids[i] for i in range(JOB_NUM*JOB_LENGTH, min((JOB_NUM+1)*JOB_LENGTH, len(all_ids)))]

    folder_name = JOB_SET + "/" + llm_name + "/" +args.data_set + "/"
    get_and_save_pickle_patients(tables_info, ID_COL, TIME_COL, pats, prefix, missing, replace, descriptive, meta, global_imputer, folder_name, EXAMPLE_PATH, "RAW_DATA", clinical, long, biogpt, finetuned_path, feature_types)

    sent_name = "RAW_DATA_" + str(prefix) +"_"+ str(missing) +"_"+ str(replace) +"_"+ str(descriptive) +"_"+ str(meta)
    get_and_save_features(pats, TIME_COL, ID_COL, feature_types, None, folder_name, EXAMPLE_PATH, sent_name,   job_id=str(JOB_NUM))
