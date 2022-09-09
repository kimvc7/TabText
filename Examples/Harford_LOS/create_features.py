import datetime
from datetime import timedelta
import json
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

with open('config.json') as config_file:
        HH_config = json.load(config_file)
        

###############################################
           #Load data and parameters
###############################################
        
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
parser.add_argument("--job_num", type=int, default=0, help='which job to run')
parser.add_argument("--data_set", type=str, default="RAW_DATA", help="which data set from 'RAW_DATA' or 'PROCESSED_DATA'")
parser.add_argument("--job_set", type=str, default="Training", help="indicate which set from 'Training' or 'Testing'")
args = parser.parse_args()
                    
JOB_NUM = args.job_num
JOB_SET = args.job_set
JOB_LENGTH = 10000
DATA_PATH = EXAMPLE_PATH + HH_config[args.data_set + "_PATH"] 

paths = [EXAMPLE_PATH, DATA_PATH, TABLES_FILE, COLUMNS_PATH]
train_start = datetime.date(TRAINING_START["YEAR"], TRAINING_START["MONTH"], TRAINING_START["DAY"])
train_end = datetime.date(TRAINING_END["YEAR"], TRAINING_END["MONTH"], TRAINING_END["DAY"])
val_end = datetime.date(VAL_END["YEAR"], VAL_END["MONTH"], VAL_END["DAY"])
test_end = datetime.date(TESTING_END["YEAR"], TESTING_END["MONTH"], TESTING_END["DAY"])

admission_df = pd.read_csv(DATA_PATH + ADMISSION_INFO_FILE)
discharge_df = pd.read_csv(DATA_PATH + DISCHARGE_INFO_FILE)
discharge_df[TIME_COL] = pd.to_datetime(discharge_df[TIME_COL], infer_datetime_format=True)

###############################################
                #Choose settings
###############################################
imputer = "zero_imp"
prefix = ""
missing = "is missing"
replace = False
descriptive = False
meta = False


###############################################
           #Get relevant patient ids
###############################################

if JOB_SET == "Training":
    training_ids = get_filtered_ids(admission_df, discharge_df, ID_COL, TIME_COL, LOCATION_COL, HOSPITAL, train_start, train_end)
    train_tables_info, global_imputer = get_model_info(paths, ID_COL, TIME_COL, imputer, training_ids)
    all_ids = training_ids
    tables_info = train_tables_info
    
elif JOB_SET == "Validation":
    validation_ids = get_filtered_ids(admission_df, discharge_df, ID_COL, TIME_COL, LOCATION_COL, HOSPITAL, train_end, val_end)
    val_tables_info, global_imputer = get_model_info(paths, ID_COL, TIME_COL, imputer, validation_ids)
    all_ids = validation_ids
    tables_info = val_tables_info
    
elif JOB_SET == "Testing":
    testing_ids = get_filtered_ids(admission_df, discharge_df, ID_COL, TIME_COL, LOCATION_COL, HOSPITAL, val_end, test_end)
    test_tables_info, global_imputer = get_model_info(paths, ID_COL, TIME_COL, imputer, testing_ids)
    all_ids = testing_ids
    tables_info = test_tables_info

pats = [all_ids[i] for i in range(JOB_NUM*JOB_LENGTH, min((JOB_NUM+1)*JOB_LENGTH, len(all_ids)))]


###############################################
#Create/Save features for desired feature types
###############################################

for feature_type in ["joint_imputations", "joint_embeddings", "sep_embeddings", "sep_imputations"]:
    sent_name = args.data_set + "_" + str(prefix) +"_"+ str(missing)+"_"+str(replace)+"_"+str(descriptive)+"_"+str(meta)
    get_and_save_features(pats, TIME_COL, ID_COL, feature_type, discharge_df, JOB_SET + "/", EXAMPLE_PATH, sent_name, job_id=str(JOB_NUM))
