import datetime
from datetime import timedelta
import json
import sys
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="Wine", help="which uci dataset")
parser.add_argument("--imputer", type=str, default="zero_imp", help="which imputer to use")
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


DATA_PATH = EXAMPLE_PATH + UCI_config["RAW_DATA_PATH"] 
paths = [EXAMPLE_PATH, DATA_PATH, TABLES_FILE, COLUMNS_PATH]
targets_df = pd.read_csv(TARGET_FILE)
imputer = args.imputer


train_df, test_df = train_test_split(targets_df, test_size=split_ratio, random_state=split_seed, stratify=targets_df[TARGET_COL])
training_ids = train_df[ID_COL].unique()
testing_ids = test_df[ID_COL].unique()

save_model_info(paths, ID_COL, TIME_COL, imputer, training_ids, testing_ids, model_name=args.data_set)
