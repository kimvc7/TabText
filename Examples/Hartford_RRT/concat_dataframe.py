from tqdm import tqdm
from glob import glob
import pandas as pd

lis_files = glob('Processed_csv_train/*')
info_csv = lis_files[0].split('/')[1].split('.')[0].split('-')

redcap = pd.read_csv('../../../../0HVI/data/LDS_RedCap.txt', sep = '|')
redcap['time'] = pd.to_datetime(redcap['Dateofevent'].str.strip(), 
                                infer_datetime_format=True)#.dt.date
redcap['target'] = (redcap['Status'].str.strip() == 'Expired').astype(int)

for feature_type in ["joint_embeddings", "sep_embeddings", "joint_imputations", "sep_imputations"]:
#     feature_type = "joint_embeddings"
    init_df = pd.DataFrame()
    for file in tqdm(lis_files):
        info_csv = file.split('/')[1].split('.')[0].split('-')
        feature_type_file = info_csv[1]
        pat_id = '-'.join(file.split('/')[1].split('-')[2:-4])
        if(feature_type_file == feature_type):
            temp_df = pd.read_csv(file)
            try:
                red_pat = redcap[redcap['PAT_ENC_CSN_GUID'] == temp_df['PAT_ENC_CSN_GUID'].values[0]]
            except:
                print(file)
            time_event = red_pat['time'].values[0]
            temp_df['CurrentDate'] = pd.to_datetime(temp_df['CurrentDate'].str.strip(), 
                                infer_datetime_format=True)#.dt.date

            temp_df = temp_df[temp_df['CurrentDate'] < time_event]
            temp_df['target'] = red_pat['target'].values[0]
            append = temp_df.tail(1)
            if(len(init_df) == 0):
                init_df = append
            else:
                init_df = pd.concat([init_df, append], axis = 0)

    init_df.to_csv(f'train-{feature_type}.csv')