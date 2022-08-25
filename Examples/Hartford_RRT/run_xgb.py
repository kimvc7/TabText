import pandas as pd
import xgboost as xgb
import sys

feature_type_lis = ["joint_embeddings", "sep_embeddings", "joint_imputations", "sep_imputations"]
ind = int(sys.argv[1])
feature_type = feature_type_lis[ind]

train = pd.read_csv(f'train-{feature_type}.csv')
x_train = train.drop(['Unnamed: 0.1', 'Unnamed: 0', 'PAT_ENC_CSN_GUID', 'CurrentDate', 'target'], axis = 1)
y_train = train['target']

x_train = x_train.drop_duplicates()
y_train = y_train[x_train.drop_duplicates().index]

print(feature_type)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_train, model.predict_proba(x_train)[:, 1]))

# test = pd.read_csv('test-joint_embeddings--missing-True-False.csv')
test = pd.read_csv(f'test-{feature_type}.csv')
x_test = test.drop(['Unnamed: 0.1', 'Unnamed: 0', 'PAT_ENC_CSN_GUID', 'CurrentDate', 'target'], axis = 1)
y_test = test['target']

from sklearn.metrics import roc_auc_score
print(roc_auc_score(y_test, model.predict_proba(x_test)[:, 1]))

