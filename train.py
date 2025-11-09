
import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


C = 1.0
n_splits = 5

output_file = f'finpay_C={C}.bin'


df = pd.read_csv('merchant_data_project.csv')


df.columns = df.columns.str.lower().str.replace(' ', '_')

for col in df.columns:
    if 'date' in col.lower():
        df[col] = pd.to_datetime(df[col], errors='coerce')



df.drop(columns=['onboard_date', 'first_trans_date', 'last_trans_date', 'state', 'wallet_id'], inplace=True)


cols_to_convert = [
    'days_to_first_txn',
    'days_since_last_txn',
    'tenure_days',
    'total_txn_count',
    'total_tpv',
    'avg_tpv',
    'txn_count_30d',
    'tpv_30d',
    'avg_txn_value_30d',
    'txn_count_90d',
    'tpv_90d',
    'avg_txn_value_90d',
    'total_commission'
]

# Clean and convert all columns safely
for col in cols_to_convert:
    df[col] = (
        df[col]
        .astype(str)                        
        .str.replace(',', '', regex=False)  
        .str.replace('₦', '', regex=False)  
        .str.strip()                        
    )
    df[col] = pd.to_numeric(df[col], errors='coerce')  

df[cols_to_convert] = df[cols_to_convert].fillna(0)




df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


categorical = list(df.select_dtypes(include=['object']).columns)
numerical = list(df.select_dtypes(exclude=['object']).columns)


numerical = [col for col in numerical if col != 'churn_flag']



def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model



def predict(df, dv, model):
     dicts = df[categorical + numerical].to_dict(orient='records')

     X = dv.transform(dicts)
     y_pred = model.predict_proba(X)[:,1]

     return y_pred

print(f'Doing validation with C={C}')


kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)  

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn_flag.values
    y_val = df_val.churn_flag.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('Validation result')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


print('Training the final model')
dv, model = train(df_full_train, df_full_train.churn_flag.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn_flag.values

auc = roc_auc_score(y_test, y_pred)

print(f'auc = {auc}')


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to {output_file}')




