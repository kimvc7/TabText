import datetime
from datetime import timedelta
import json
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

with open('config.json') as config_file:
        HH_config = json.load(config_file)
        
##############################################
           #Load data and parameters
##############################################

DIR_NAME = HH_config["TABTEXT_PATH"]
EXAMPLE_PATH = HH_config["EXAMPLE_PATH"]
TABLES_FILE = HH_config["TABLES_FILE"]
COLUMNS_PATH = HH_config["COLUMNS_PATH"]
ID_COL = HH_config["ID_COL"]
TIME_COL = HH_config["TIME_COL"]
TRAINING_START = HH_config["TRAINING_START"]
TRAINING_END = HH_config["TRAINING_END"]
VAL_END = HH_config["VAL_END"]
TESTING_END = HH_config["TESTING_END"]
DISCHARGE_INFO_FILE = HH_config["DISCHARGE_INFO_FILE"]
ADMISSION_INFO_FILE = HH_config["ADMISSION_INFO_FILE"]
LOCATION_COL = HH_config["LOCATION_COL"]
HOSPITAL = HH_config["HOSPITAL"]


sys.path.insert(0, DIR_NAME + 'TabText/src')
from get_data_info import *
from get_patients import *
from get_features import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_set", type=str, default="RAW_DATA", help="which data set from 'RAW_DATA' or 'PROCESSED_DATA'")
args = parser.parse_args()
                    
DATA_PATH = EXAMPLE_PATH + HH_config[args.data_set + "_PATH"] 
paths = [EXAMPLE_PATH, DATA_PATH, TABLES_FILE, COLUMNS_PATH]

train_start = datetime.date(TRAINING_START["YEAR"], TRAINING_START["MONTH"], TRAINING_START["DAY"])
train_end = datetime.date(TRAINING_END["YEAR"], TRAINING_END["MONTH"], TRAINING_END["DAY"])
val_end = datetime.date(VAL_END["YEAR"], VAL_END["MONTH"], VAL_END["DAY"])
test_end = datetime.date(TESTING_END["YEAR"], TESTING_END["MONTH"], TESTING_END["DAY"])

admission_df = pd.read_csv(DATA_PATH + ADMISSION_INFO_FILE)
discharge_df = pd.read_csv(DATA_PATH + DISCHARGE_INFO_FILE)


##############################################
      #Save Model info w.r.t Training Set
##############################################
imputer = "zero_imp"
training_ids = get_filtered_ids(admission_df, discharge_df, ID_COL, TIME_COL, LOCATION_COL, HOSPITAL, train_start, train_end)
save_model_info(paths, ID_COL, TIME_COL, imputer, training_ids)
