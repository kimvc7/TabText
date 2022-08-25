import pandas as pd
import numpy as np
import json
import datetime

with open('config.json') as config_file:
        HH_config = json.load(config_file)

ID_COL = HH_config["ID_COL"]
TIME_COL = HH_config["TIME_COL"]
EXAMPLE_PATH = HH_config["EXAMPLE_PATH"]
RAW_DATA_PATH = HH_config["RAW_DATA_PATH"]
PROCESSED_DATA_PATH = HH_config["PROCESSED_DATA_PATH"]


#Load all data frames
event_df = pd.read_csv("../../../allData_unique/adtEventData.csv", "|")
clin_df = pd.read_csv("../../../allData_unique/clindocData.csv", "|")
enc_df = pd.read_csv("../../../allData_unique/encounterDataNew.csv", "|")
orders_df = pd.read_csv("../../../allData_unique/ordersData.csv", "|", encoding='latin-1')
c19_df = pd.read_csv("../../../allData_unique/C19_addon.csv", "|", encoding='latin-1')
notes_df = pd.read_csv("../../../allData_unique/notesData.csv", "|", encoding='latin-1')
dnr_df = pd.read_csv("../../../allData_unique/dnrData.csv", "|", encoding='latin-1')
dis_df = pd.read_csv("../../../allData_unique/discharge_addon.csv", "|", encoding='latin-1')
adtOr_df = pd.read_csv("../../../allData_unique/adtOrdersData.csv", "|", encoding='latin-1')

#Find ids of patients who eventually have class "inpatient"
inpatient_ids = clin_df["PAT_ENC_CSN_ID"].unique()


#Create admission and discharge files
event_df = event_df[event_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]

admission_df = event_df[event_df["EVENT_TYPE"] == "Admission"]
admission_df[TIME_COL] = pd.to_datetime(admission_df["EFFECTIVE_DTTM"], infer_datetime_format=True).dt.date
admission_df = admission_df[admission_df[TIME_COL].notnull()]
admission_df = admission_df.sort_values(TIME_COL).groupby(["PAT_ENC_CSN_ID", "LOCATION_NAME"]).tail(1)
admission_df = admission_df[["PAT_ENC_CSN_ID", TIME_COL, "LOCATION_NAME"]]

discharge_df = event_df[event_df["EVENT_TYPE"] == "Discharge"]
discharge_df[TIME_COL] = pd.to_datetime(discharge_df["EFFECTIVE_DTTM"], infer_datetime_format=True).dt.date
discharge_df = discharge_df[discharge_df[TIME_COL].notnull()]
discharge_df = discharge_df.sort_values(TIME_COL).groupby(["PAT_ENC_CSN_ID", "LOCATION_NAME"]).tail(1)
discharge_df = discharge_df[["PAT_ENC_CSN_ID", TIME_COL, "LOCATION_NAME"]]

#Create Discharge and Admission Files
admission_df = admission_df[admission_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
discharge_df = discharge_df[discharge_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
admission_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "admissionInfo.csv")
discharge_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "dischargeInfo.csv")

#Filter all tables based on valid patient ids
event_df = event_df[event_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
enc_df = enc_df[enc_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
orders_df = orders_df[orders_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
c19_df = c19_df[c19_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
notes_df = notes_df[notes_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
dnr_df = dnr_df[dnr_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
dis_df = dis_df[dis_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]
adtOr_df = adtOr_df[adtOr_df["PAT_ENC_CSN_ID"].isin(inpatient_ids)]

#Select most recent info for static data
enc_df[TIME_COL] = pd.to_datetime(enc_df['EXTRACTED_DATE'], infer_datetime_format=True)
enc_df[TIME_COL] = enc_df[TIME_COL].fillna(min(enc_df[TIME_COL]))
enc_df = enc_df.sort_values(TIME_COL).groupby(ID_COL).tail(1)
enc_df = enc_df.drop([TIME_COL], axis = 1)


#Save Raw Tables
event_df.to_csv(EXAMPLE_PATH +  RAW_DATA_PATH + "adtEventData.csv")
clin_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "clindocData.csv")
enc_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "encounterData.csv")
orders_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "ordersData.csv")
c19_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "C19_addon.csv")
notes_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "notesData.csv")
dnr_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "dnrData.csv")
dis_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "discharge_addon.csv")
adtOr_df.to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "adtOrdersData.csv")



#Find and save moratlity targets

home_wo_service = ["Home or Self Care"]
expired = ["Expired", "Expired in Medical Facility", "Expired in Medical Facility/ HHC Home Care only"]
hospice = ["Hospice/Medical Facility", "Hospice/Home"]
home_w_service = ["Home-Health Care Svc", "Home with Health Care Services"]

other_facilities_hospitals = ["Skilled Nursing Facility", "Admitted as an Inpatient", "Short Term-Acute Care Hospital", "Short Term Hospital",  "Inpatient Psychiatric Hospital", "Another Health Care Institution Not Defined", "Inpatient Rehab Facility", "Nursing Facility", "Critical Access Hospital", "Custodial Care Facility", "LTACH (Long Term Acute Care Hospital)", "Intermediate Care Facility", "Cancer Center/Children's Hospital","Skilled Nursing Facility","Inpatient Rehab Facility","Psychiatric Hospital",
"Short Term Hospital","Short Term-Acute Care Hospital","Inpatient Psychiatric Hospital",
"LTACH (Long Term Acute Care Hospital)","Critical Access Hospital","Nursing Facility","Intermediate Care Facility","Skilled Nursing Facility","Nursing Facility","Inpatient Rehab Facility",
        "Intermediate Care Facility","Short Term-Acute Care Hospital","Inpatient Psychiatric Hospital",
        "Cancer Center/Children's Hospital","Critical Access Hospital","LTACH (Long Term Acute Care Hospital)","Another Health Care Institution Not Defined"]

ama_other = ["Left Against Medical Advice/ AMA", "Court/Law Enforcement", "ED Dismiss - Never Arrived","Another Health Care Institution Not Defined","Left Against Medical Advice/ AMA",
        "Admitted as an Inpatient","Still a Patient"]

miss = ["missing"]


def filterExpiredOnly(row):
    if row.DISCHARGE_DISPO is not None:
        if row.DISCHARGE_DISPO in (expired + hospice):
            return 1
        else:
            return 0
    return 0

target_data = enc_df[["PAT_ENC_CSN_ID", "DISCHARGE_DISPO"]]
target_data['target_mortality'] = target_data.apply(lambda row: filterExpiredOnly(row), axis=1)
target_data[["PAT_ENC_CSN_ID", 'target_mortality']].to_csv(EXAMPLE_PATH + RAW_DATA_PATH + "target_mortality.csv")


# ranges = ["PAO2", "PH", "AST", "CALCIUM", "HGB", "MCV", "METHGB", "CRP", "LACTIC_ACID", "PACO2", "ALT", 
#           "AMYLASE", "CREATININE", "SODIUM", "POTASSIUM", "CARBOXYHGB", "PTT", "D_DIMER", "LYMPHS_ABS",
#           "LDH", "HIGH_SENSITIVITY_D_DIMER", "PLATELET", "INTERLEUKIN_6", "HGBA1C"]


# def fix_RASS_LAST(rass_col):
#     new_rass = []
#     for i in range(len(rass_col)):
#         if str(rass_col[i]).lower() in ["nan", "none", ""]:
#             new_rass.append(str(rass_col[i]))
#         else:
#             new_rass.append(rass_col[i].split(">")[1])
#     return new_rass

# def fix_ipa_ambulating(ambulating_col):
#     new_amb = []
#     amb_map = {0:"not difficult", 1:"moderately difficult", 2:"cannot ambulate independentely"}
#     for i in range(len(ambulating_col)):
#         if str(ambulating_col[i]).lower() in ["nan", "none", ""]:
#             new_amb.append(str(ambulating_col[i]))
#         else:
#             new_amb.append(amb_map[float(ambulating_col[i])])
#     return new_amb


# def fix_LAST_SCORE(last_score_col):
#     map_dict = {'no pain': 0, 'mild pain':2, 'moderate pain': 4, 'severe pain': 8}
#     new_col = []
#     for i in range(len(last_score_col)):
#         if str(last_score_col[i]).lower() in ["nan", "none", ""]:
#             new_col.append(str(last_score_col[i]))
#         elif last_score_col[i] in map_dict.keys():
#             new_col.append(map_dict[last_score_col[i]])
#         else:
#             new_col.append(last_score_col[i].split("-")[0])
#     return new_col

# def fix_lab_value(lab_col):
#     new_col = []
#     for i in range(len(lab_col)):
#         try:
#             new_col.append(float(lab_col[i]))
#         except:
#             new_col.append(None)
#     return new_col

# def EXPAND_RANGE(col):    
#     new_col_low = []
#     new_col_high = []
    
#     for i in range(len(col)):
#         try:
#             value = col[i]
#             if "-" in value:
#                 assert(len(value.split("-")) == 2)
#                 new_col_low.append(value.split("-")[0])
#                 new_col_high.append(value.split("-")[1])
#             elif ">" in value:
#                 assert(len(value.split(">")) == 2)
#                 new_col_low.append(value.split(">")[1])
#                 new_col_high.append(None)
#             elif "<" in value:
#                 assert(len(value.split("<")) == 2)
#                 new_col_low.append(None)
#                 new_col_high.append(value.split("<")[1])
#         except:
#             new_col_low.append(None)
#             new_col_high.append(None)
            
#     return new_col_low, new_col_high

# clin_df["RASS_LAST"] = fix_RASS_LAST(clin_df["RASS_LAST"].values)
# clin_df["IPA_DIFFICULTY_AMBULATING"] = fix_ipa_ambulating(clin_df["IPA_DIFFICULTY_AMBULATING"].values)
# clin_df['LAST_PAIN_SCORE'] = fix_LAST_SCORE(clin_df['LAST_PAIN_SCORE'].values)

# for col in orders_df.columns:
#     if ("_VALUE" in col) or (col == "WHITE_BLOOD_CELL"):
#         orders_df[col] = fix_lab_value(orders_df[col].values)
        
# for col in c19_df:
#     if "_RANGE" in col:
#         c19_df[col.split("_RANGE")[0]] = fix_lab_value(c19_df[col.split("_RANGE")[0]].values)
        
# for col in ranges:
#     c19_df[col + "_LOW"] = EXPAND_RANGE(c19_df[col + "_RANGE"].values)[0]
#     c19_df[col + "_HIGH"] = EXPAND_RANGE(c19_df[col + "_RANGE"].values)[1]

# event_df.to_csv("./Data/ProcessedData/adtEventData.csv")
# clin_df.to_csv("./Data/ProcessedData/clindocData.csv")
# enc_df.to_csv("./Data/ProcessedData/encounterData.csv")
# orders_df.to_csv("./Data/ProcessedData/ordersData.csv")
# c19_df.to_csv("./Data/ProcessedData/c19Data.csv")


