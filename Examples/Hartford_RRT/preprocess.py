import pandas as pd
import numpy as np
import json

with open('config.json') as config_file:
        HH_config = json.load(config_file)

ID_COL = HH_config["ID_COL"]
TIME_COL = HH_config["TIME_COL"]

def clean_vitals_meas(x):
    if('/' in x):
        two_nums = x.split('/')
        try:
            return int(two_nums[0])/int(two_nums[1])
        except:
            # Not sure why some are of this format, they have 0/0, 90/0 etc
            print(x)
            return 0.0
    else:
        return float(x)
    
def change_height_from_str(x):
    try:
        converted = round(float(x.split()[1][:-1]) * 2.54 + float(x.split()[0][:-1]) * 30.48, 3)
        return converted
    except:
        return x
    
def str_to_float(x):
    try:
        return float(x.strip())
    except:
        return np.nan
    
def format_alcohol(x):
    x = x.strip()
    if('-' in x):
        two_nums = x.split('-')
        return (float(two_nums[0]) + float(two_nums[1]))/2
    elif(x == ''):
        return np.nan
    else:
        return float(x)
    
demo_df = pd.read_csv("../../../../0HVI/data/LDS_Demo.txt", "|")
demo_df['HEIGHT'] = demo_df['HEIGHT'].apply(change_height_from_str)

enc_df = pd.read_csv("../../../../0HVI/data/LDS_Encounter.txt", "|")
enc_df['HEIGHT'] = enc_df['HEIGHT'].apply(change_height_from_str)

med_df = pd.read_csv("../../../../0HVI/data/LDS_Meds.txt", "|")
med_df['DOSE'] = med_df['DOSE'].apply(str_to_float)

prob_df = pd.read_csv("../../../../0HVI/data/LDS_ProblemList.txt", "|")

social_df = pd.read_csv("../../../../0HVI/data/LDS_SocialHx.txt", "|")
social_df['TOBACCO_PAK_PER_DY'] = social_df['TOBACCO_PAK_PER_DY'].apply(str_to_float)
social_df['ALCOHOL_OZ_PER_WK'] = social_df['ALCOHOL_OZ_PER_WK'].apply(format_alcohol)
social_df['ILLICIT_DRUG_FREQ'] = social_df['ILLICIT_DRUG_FREQ'].apply(str_to_float)

signs_df = pd.read_csv("../../../../0HVI/data/LDS_SignsSymptoms.txt", "|", encoding= 'unicode_escape')

vitals_df = pd.read_csv("../../../../0HVI/data/LDS_Vitals.txt", "|")
vitals_df['MEAS_VALUE'] = vitals_df['MEAS_VALUE'].str.strip().apply(clean_vitals_meas)

redcap_df = pd.read_csv("../../../../0HVI/data/LDS_RedCap.txt", "|")

demo_df = demo_df.rename(columns={'WEIGHT': "WEIGHT_demo", "HEIGHT": "HEIGHT_demo"})
enc_df = enc_df.rename(columns={'WEIGHT': "WEIGHT_enc", "HEIGHT": "HEIGHT_enc"})
signs_df = signs_df.rename(columns={'MEAS_VALUE': "MEAS_VALUE_signs", 'FLO_MEAS_NAME': 'FLO_MEAS_NAME_signs'})
vitals_df = vitals_df.rename(columns={'MEAS_VALUE': "MEAS_VALUE_vitals", 'FLO_MEAS_NAME': 'FLO_MEAS_NAME_vitals'})

prob_df.loc[prob_df['PRINCIPAL_PROB_YN'] == ' ', 'PRINCIPAL_PROB_YN'] = 'N'

df_lis = {'med': [med_df, 'PERFORMEDFROMDTM'], 
          'prob': [prob_df, 'NOTED_DATE'], 
          'social': [social_df, 'CONTACT_DATE'], 
          'signs': [signs_df, 'dtDateRec'], 
          'vitals': [vitals_df, 'dtDateRec']}
for name in df_lis.keys():
    df = df_lis[name][0]
    time = df_lis[name][1]
    df[time] = pd.to_datetime(df[time], infer_datetime_format=True)#.dt.date
    df[time] = df[time].dt.round("H")
    print(name)
    
enc_df[TIME_COL] = enc_df['HOSP_DISCHRG_TIME']

# Make sure you remove any trailing spaces or new lines because it will crash BERT
for df in [demo_df, enc_df, med_df, prob_df, social_df, signs_df, redcap_df]:
    for col in df.columns:
        try:
            df[col] = df[col].str.strip()
        except:
            print(col, 'prob not string')


redcap_df = redcap_df[redcap_df['Complete'].str.strip() == 'Complete']

def change_date(x):
    try:
        return pd.to_datetime(x)
    except:
        return None
    
redcap_df['time'] = redcap_df['Dateofevent'].str.strip() + ' ' + redcap_df['Timeofevent'].str.strip()
redcap_df['time'] = redcap_df['time'].apply(change_date)
redcap_df = redcap_df[~redcap_df['time'].isna()]

med_df_new = pd.DataFrame()
prob_df_new = pd.DataFrame()
social_df_new = pd.DataFrame()
signs_df_new = pd.DataFrame()
vitals_df_new = pd.DataFrame()

for i in redcap_df['PAT_ENC_CSN_GUID'].unique():
    pat_redcap = redcap_df[redcap_df['PAT_ENC_CSN_GUID'] == i]
    pat_redcap_time = pat_redcap['time'].values[0]

    new_med_df = med_df[(med_df['PAT_ENC_CSN_GUID'] == i) & (med_df['PERFORMEDFROMDTM'] < pat_redcap_time)]
    new_prob_df = prob_df[(prob_df['PAT_ENC_CSN_GUID'] == i) & (prob_df['NOTED_DATE'] < pat_redcap_time)]
    new_social_df = social_df[(social_df['PAT_ENC_CSN_GUID'] == i) & (social_df['CONTACT_DATE'] < pat_redcap_time)]
    new_signs_df = signs_df[(signs_df['PAT_ENC_CSN_GUID'] == i) & (signs_df['dtDateRec'] < pat_redcap_time)]
    new_vitals_df = vitals_df[(vitals_df['PAT_ENC_CSN_GUID'] == i) & (vitals_df['dtDateRec'] < pat_redcap_time)]
    
    med_df_new = pd.concat([med_df_new, new_med_df], axis = 0)
    prob_df_new = pd.concat([prob_df_new, new_prob_df], axis = 0)
    social_df_new = pd.concat([social_df_new, new_social_df], axis = 0)
    signs_df_new = pd.concat([signs_df_new, new_signs_df], axis = 0)
    vitals_df_new = pd.concat([vitals_df_new, new_vitals_df], axis = 0)
    
    
demo_df.to_csv('./Data/demo.csv')
enc_df.to_csv('./Data/enc.csv')
med_df_new.to_csv('./Data/med.csv')
prob_df_new.to_csv('./Data/problem.csv')
social_df_new.to_csv('./Data/social.csv')
signs_df_new.to_csv('./Data/signs.csv')
redcap_df.to_csv('./Data/redcap.csv')
vitals_df_new.to_csv('./Data/vitals.csv')