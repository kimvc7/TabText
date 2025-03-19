import json
import sys
import warnings
warnings.filterwarnings("ignore")
import argparse
import os.path
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn import preprocessing
import os
from sklearn.preprocessing import LabelEncoder


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="Wine", help="which uci dataset")
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


data_set = args.data_set
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


with open('configs/config_' +data_set+ '.json') as config_file:
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
from train_models import *

llm_name = ""
if long:
    if clinical:
        if finetuned:
            llm_name = "ClinicalLongformerFinetuned"
        else:
            llm_name = "ClinicalLongformer"
    else:
        if finetuned:
            llm_name = "LongformerFinetuned"
        else:
            llm_name = "Longformer"
if biogpt:
    assert(long==False)
    assert(clinical==False)
    assert(finetuned==False)
    llm_name = "BioGPT"

sent_name = "RAW_DATA_" + str(prefix) +"_"+ str(missing) +"_"+ str(replace) +"_"+ str(descriptive) +"_"+ str(meta)

X_emb_train = load_embeddings("Training/" + llm_name + "/" + data_set + "/sep_embeddings/" + sent_name + "/Features/", start=0, num_files=1)

X_tab_train = load_embeddings("Training/" + llm_name + "/" + data_set + "/sep_imputations/" + sent_name+"/Features/", start=0, num_files=1)

X_emb_test = load_embeddings("Testing/" + llm_name + "/" + data_set + "/sep_embeddings/" + sent_name + "/Features/", start=0, num_files=1)

X_tab_test = load_embeddings("Testing/" + llm_name + "/" + data_set + "/sep_imputations/" + sent_name + "/Features/", start=0, num_files=1)

targets_df = pd.read_csv(TARGET_FILE)[[ID_COL, TARGET_COL]]
le = LabelEncoder()
target_encoded = le.fit_transform(targets_df[TARGET_COL])
targets_df[TARGET_COL] = target_encoded
num_classes = len(le.classes_)

merged_train = X_tab_train.merge(X_emb_train, on=[ID_COL], how="inner").merge(targets_df, on=[ID_COL], how="inner")
merged_train = merged_train.drop(columns=[ID_COL])

merged_valtest = X_tab_test.merge(X_emb_test, on=[ID_COL], how="inner").merge(targets_df, on=[ID_COL], how="inner")
merged_valtest = merged_valtest.drop(columns=[ID_COL])

def get_valid_cols(method, merged_columns, tab_columns):
    valid_cols = []
    if method == 'tabular':
        valid_cols = [c for c in merged_columns if ((c in tab_columns) or (c==TARGET_COL))]
    elif method == 'merged':
        valid_cols = list(merged_columns)
    elif method == 'language':
        valid_cols = [c for c in merged_columns if (c not in tab_columns)]
    return valid_cols


merged_test, merged_val = train_test_split(merged_valtest, test_size=0.5, random_state=split_seed, stratify=merged_valtest[TARGET_COL])

for method in ["tabular", "merged", "language"]:
    folder_name = data_set  + "_" + method
    valid_cols = get_valid_cols(method, merged_train.columns, X_tab_train.columns)
    df_train, df_val, df_test = merged_train[valid_cols], merged_val[valid_cols], merged_test[valid_cols]

    for n_est in [100, 200, 300]:
        for max_param in [3, 5, 7]:
            for lr in [0.05, 0.1, 0.3]:
                for λ in [0.01, 0.001, 1e-4, 1e-5, 0]:

                    val_auc, val_acc, _ = train_xgb(df_train, df_val, TARGET_COL, n_est, max_param, lr, λ, num_classes)
                    test_auc, test_acc, _ = train_xgb(pd.concat([df_train, df_val], axis=0), df_test, TARGET_COL, n_est, max_param, lr, λ, num_classes)
                    target = TARGET_COL

                    results = [target, val_auc, test_auc, str(df_train.shape), str(df_val.shape), str(df_test.shape),
                               val_acc, test_acc, n_est, max_param, lr, λ, split_seed]

                    column_list = ["target", "val_auc",  "test_auc",  "train_size", "val_size", "test_size", 
                                   "val_acc", "test_acc", "n_est", "max_param", "lr", "lambda", "seed"]

                    df_results = pd.DataFrame(np.array([results])) 

                    if not os.path.exists(EXAMPLE_PATH + 'Results/'+llm_name + '/'+ sent_name + "/" ):
                        os.makedirs(EXAMPLE_PATH + 'Results/'+ llm_name  +'/' + sent_name + "/" )

                    # if file does not exist write header   
                    if not os.path.isfile(EXAMPLE_PATH + 'Results/'+ llm_name  +'/' + sent_name + "/" + folder_name + ".csv"):
                        pd.DataFrame([column_list]).to_csv(EXAMPLE_PATH + 'Results/'+ llm_name +'/' + sent_name + "/" + folder_name + ".csv", header=False)

                    # else it exists so append without writing the header
                    df_results.to_csv(EXAMPLE_PATH + 'Results/'+ llm_name  +'/'+ sent_name + "/" + folder_name + ".csv",
                                      mode='a', header=False)
