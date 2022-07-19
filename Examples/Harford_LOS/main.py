import datetime
from datetime import timedelta
import json
import sys

import warnings
warnings.filterwarnings("ignore")

with open('../../config.json') as config_file:
        config = json.load(config_file)

DIR_NAME = config["DIRECTORY_NAME"]
PATIENTS_PATH = config["PATIENTS_PATH"]

with open('config.json') as config_file:
        HH_config = json.load(config_file)

TABLES_PATH = HH_config["TABLES_PATH"]
COL_PATH = HH_config["COLUMNS_PATH"]
ID_COL = HH_config["ID_COL"]
TIME_COL = HH_config["TIME_COL"]
FILTERING_DF_PATH = HH_config["FILTERING_DF"]
FILTER_MAP = HH_config["FILTER_MAP"]
TRAINING_START = HH_config["TRAINING_START"]
TRAINING_END = HH_config["TRAINING_END"]
TESTING_START = HH_config["TESTING_START"]
TESTING_END = HH_config["TESTING_END"]
DISCHARGE_INFO_PATH = HH_config["DISCHARGE_INFO_PATH"]


sys.path.insert(0, DIR_NAME + 'TabText/src')
from get_data_info import *
from get_patients import *
from get_features import *


train_start = datetime.date(TRAINING_START["YEAR"], TRAINING_START["MONTH"], TRAINING_START["DAY"])
train_end = datetime.date(TRAINING_END["YEAR"], TRAINING_END["MONTH"], TRAINING_END["DAY"])
test_start = datetime.date(TESTING_START["YEAR"], TESTING_START["MONTH"], TESTING_START["DAY"])
test_end = datetime.date(TESTING_END["YEAR"], TESTING_END["MONTH"], TESTING_END["DAY"])
filtering_df = pd.read_csv(FILTERING_DF_PATH)
filtered_ids = get_filtered_ids(FILTER_MAP, filtering_df, ID_COL)


#Save all the model info (classes, imputers, encoders) with respect to the training set
#save_model_info(TABLES_PATH, COL_PATH,ID_COL, TIME_COL, "zero", train_start, train_end, filtered_ids)

#Get info for training set
train_tables_info, training_ids, global_imputer = get_model_info(TABLES_PATH, ID_COL, TIME_COL, train_start, train_end, filtered_ids)
#Get info for testing set
test_tables_info, testing_ids, global_imputer = get_model_info(TABLES_PATH, ID_COL, TIME_COL, test_start, test_end, filtered_ids)


#Choose sentence settings
prefix = "the patient"
missing = "missing"
replace = False
descriptive = True

#Save training data
get_and_save_pickle_patients(train_tables_info, ID_COL, TIME_COL, list(training_ids)[1:5], prefix, missing, replace, descriptive, global_imputer)

#Save testing data
testing_ids = testing_ids - training_ids
get_and_save_pickle_patients(test_tables_info, ID_COL, TIME_COL, list(testing_ids)[1:5], prefix, missing, replace, descriptive, global_imputer)

#Get discharge times
disch_info = pd.read_csv(DISCHARGE_INFO_PATH)
disch_info[TIME_COL] = pd.to_datetime(disch_info[TIME_COL], infer_datetime_format=True)

#Create training_features
for feature_type in ["joint_embeddings", "sep_embeddings", "joint_imputations", "sep_imputations"]:
    train_features = create_features(list(training_ids)[1:5], TIME_COL, ID_COL,  PATIENTS_PATH, feature_type, disch_info, train_start, train_end)
    test_features = create_features(list(testing_ids)[1:5], TIME_COL, ID_COL, PATIENTS_PATH, feature_type, disch_info, test_start, test_end)
    print("Features: ", str(feature_type))
    print("============Train Features============")
    print(train_features)
    print("============Test Features============")
    print(test_features)

