import pandas as pd
import numpy as np
import json

with open('config.json') as config_file:
        HH_config = json.load(config_file)

ID_COL = HH_config["ID_COL"]
TIME_COL = HH_config["TIME_COL"]



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


def fix_LAST_SCORE(last_score_col):
    map_dict = {'no pain': 0, 'mild pain':2, 'moderate pain': 4, 'severe pain': 8}
    new_col = []
    for i in range(len(last_score_col)):
        if str(last_score_col[i]).lower() in ["nan", "none", ""]:
            new_col.append(str(last_score_col[i]))
        elif last_score_col[i] in map_dict.keys():
            new_col.append(map_dict[last_score_col[i]])
        else:
            new_col.append(last_score_col[i].split("-")[0])
    return new_col

event_df = pd.read_csv("../../../allData_unique/adtEventData.csv", "|")
clin_df = pd.read_csv("../../../allData_unique/clindocData.csv", "|")
enc_df = pd.read_csv("../../../allData_unique/encounterData.csv", "|")


clin_df["RASS_LAST"] = fix_RASS_LAST(clin_df["RASS_LAST"].values)
clin_df["IPA_DIFFICULTY_AMBULATING"] = fix_ipa_ambulating(clin_df["IPA_DIFFICULTY_AMBULATING"].values)
clin_df['LAST_PAIN_SCORE'] = fix_LAST_SCORE(clin_df['LAST_PAIN_SCORE'].values)


enc_df[TIME_COL] = pd.to_datetime(enc_df['EXTRACTED_DATE'], infer_datetime_format=True)
enc_df[TIME_COL] = enc_df[TIME_COL].fillna(min(enc_df[TIME_COL]))
enc_df = enc_df.sort_values(TIME_COL).groupby(ID_COL).tail(1)
enc_df = enc_df.drop([TIME_COL], axis = 1)

event_df.to_csv("./Data/adtEventData.csv")
clin_df.to_csv("./Data/clindocData.csv")
enc_df.to_csv("./Data/encounterData.csv")


