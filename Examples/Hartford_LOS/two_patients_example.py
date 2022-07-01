import datetime
from datetime import timedelta
import pandas as pd
import sys

sys.path.insert(0, '../../src')
from get_patients import *

import warnings
warnings.filterwarnings("ignore")

def fix_RASS_LAST(rass_col):
    new_rass = []
    for i in range(len(rass_col)):
        if str(rass_col[i]).lower() in ["nan", "none", ""]:
            new_rass.append(str(rass_col[i]))
        else:
            new_rass.append(rass_col[i].split(">")[1])
    return new_rass

def fix_ipa_ambulating(ambulating_col):
    new_amb = []
    amb_map = {0:"not difficult", 1:"moderately difficult", 2:"cannot ambulate independentely"}
    for i in range(len(ambulating_col)):
        if str(ambulating_col[i]).lower() in ["nan", "none", ""]:
            new_amb.append(str(ambulating_col[i]))
        else:
            new_amb.append(amb_map[float(ambulating_col[i])])
    return new_amb

def create_time_weights(timestamps):
    #TO-DO
    n = len(timestamps)
    return [1/n for i in range(n)]

event_df = pd.read_csv("../../../allData_downloaded/adtEventData.csv", "|")
clin_df = pd.read_csv("../../../allData_downloaded/clindocData.csv", "|")
enc_df = pd.read_csv("../../../allData_downloaded/encounterData.csv", "|")

event_df = event_df[(event_df['PAT_ENC_CSN_ID'] == 100087860068) | (event_df['PAT_ENC_CSN_ID'] ==100083064488)]
clin_df = clin_df[(clin_df['PAT_ENC_CSN_ID'] == 100087860068) | (clin_df['PAT_ENC_CSN_ID'] ==100083064488)]
enc_df = enc_df[(enc_df['PAT_ENC_CSN_ID'] == 100087860068) | (enc_df['PAT_ENC_CSN_ID'] ==100083064488)]

event_df['time'] = pd.to_datetime(event_df['EFFECTIVE_DTTM'], infer_datetime_format=True)#.dt.date 
clin_df['time'] = pd.to_datetime(clin_df['CALENDAR_DT'], infer_datetime_format=True)#.dt.date

clin_df["RASS_LAST"] = fix_RASS_LAST(clin_df["RASS_LAST"].values)
clin_df["IPA_DIFFICULTY_AMBULATING"] = fix_ipa_ambulating(clin_df["IPA_DIFFICULTY_AMBULATING"].values)

events = {}
events["df"] = event_df
events["name"] = "ADT_Events"
events["attributes_info"] = get_attributes_info(event_df, "./DataInfo/adtEventDataColumnsInfo.csv")
events["metadata"] = "The following is the information for admission, discharge and transfer events: "

clinical = {}
clinical["df"] = clin_df
clinical["name"] = "Clinical_Documents"
clinical["attributes_info"] = get_attributes_info(clin_df, "./DataInfo/clinDocDataColumnsInfo.csv")
clinical["metadata"] = "The following is the clinical information: "

encounter = {}
encounter["df"] = enc_df
encounter["name"] = "Encounter_Information"
encounter["attributes_info"] = get_attributes_info(enc_df, "./DataInfo/encounterDataColumnsInfo.csv")
encounter["metadata"] = "The following is the encounter information: "


tables_info = [encounter, events, clinical]

patients = get_patients(tables_info, 'PAT_ENC_CSN_ID', 'time')
print("=================================")
print("FIRST PATIENT TIMED DATA:")
print("=================================")
print(patients[0].create_timed_data("the patient ", "missing", replace_numbers=False, descriptive=True, merge_tables_text=True))

print("=================================")
print("FIRST PATIENT TIME_BOUNDED DATA:")
print("=================================")
t = patients[0].create_timed_data("the patient ", "missing", replace_numbers=False, descriptive=True, merge_tables_text=True) 
print(patients[0].get_timebounded_embeddings(create_time_weights, start_hr = t.iloc[2]["time"], end_hr = t.iloc[5]["time"], merge_tables_text=True))


print("=================================")
print("SECOND PATIENT TIMED DATA")
print("=================================")
print(patients[1].create_timed_data("the patient ", "missing", replace_numbers=False, descriptive=True, merge_tables_text=True))
for patient in patients:
    print("=================================")
    print("PATIENT: ", patient.id)
    print("=================================")
    timed_data = patient.create_timed_data("the patient ", "missing", replace_numbers=False, descriptive=True, merge_tables_text=True)
    print("first row is: ")
    print("time: ", str(timed_data.iloc[0]["time"]))
    print("text: ", timed_data.iloc[0]["text"])
    #print("embeddings", str(timed_data.iloc[0]["embeddings"]))

