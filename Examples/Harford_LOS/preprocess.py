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

#Select most recent info for adt events
event_df[TIME_COL] = pd.to_datetime(event_df["EFFECTIVE_DTTM"], infer_datetime_format=True).dt.date
event_df = event_df.sort_values("EFFECTIVE_DTTM").groupby([ID_COL, TIME_COL]).tail(1)
adtOr_df[TIME_COL] = pd.to_datetime(adtOr_df["ORDER_DTTM"], infer_datetime_format=True).dt.date
adtOr_df = adtOr_df.sort_values("ORDER_DTTM").groupby([ID_COL, TIME_COL]).tail(1)


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
