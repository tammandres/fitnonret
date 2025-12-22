import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re


# Paths
data_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\results')
in_path = out_path


# Helper functions
def summarise_cat(df, col, name, digits=2, sort=True):
    """Summarise categorical data"""
    s = df[col]

    # Designate missing values with 'NULL'
    if 'NULL' in s:
        raise ValueError("NULL is among values")
    else:
        s = s.fillna('NULL')

    counts = s.value_counts(sort=sort)
    if (counts < 10).any():
        raise ValueError('Some counts less than 10')
    perc = counts / df.shape[0] * 100
    perc = perc.round(digits)

    out = pd.concat(objs=[counts, perc], axis=1)
    out['reformat'] = counts.astype(str) + ' (' + perc.astype(str) + ')'
    out = out.reset_index()
    out.columns = ['Category', 'Value', 'Percent', 'Value (percent)']
    out['Characteristic'] = name
    out = out[['Characteristic', 'Category', 'Value', 'Percent', 'Value (percent)']]
    return out


def describe(df, df_all, desc_return=False):

    desc = pd.DataFrame()

    # Add number of patients
    num_patient = df.patient_id.nunique()
    row = pd.DataFrame({'Characteristic': ['Number of patients'], 'Category': [''], 'Value': [num_patient],
                        'Percent': ['100'],
                        'Value (percent)': [str(num_patient) + ' (100)']})
    desc = pd.concat(objs=[desc, row], axis=0)

    # Add number of tests per patint
    test_count = df_all.groupby('patient_id').size()
    test_count[test_count > 3] = '4+'
    test_count = test_count.astype(str)
    test_count = test_count.rename('num_fit').reset_index()
    row = summarise_cat(test_count, 'num_fit', 'Number of tests per patient')
    desc = pd.concat(objs=[desc, row], axis=0)

    if not desc_return:
        cat_cols = {'gender_male': 'Gender', 
                    'age_group': 'Age group',
                    'ethnicity': 'Ethnicity', 
                    'imd_quintile': 'IMD quintile',
                    }
    else:
        cat_cols = {'gender_male': 'Gender', 
                    'age_group': 'Age group',
                    'ethnicity': 'Ethnicity', 
                    'imd_quintile': 'IMD quintile',
                    'ret1_days14': 'Test return within 14 days (type 1)',
                    'ret2_days14': 'Test return within 14 days (type 2)',
                    'ret1_days28': 'Test return within 28 days (type 1)',
                    'ret2_days28': 'Test return within 28 days (type 2)',
                    'ret1_days70': 'Test return within 70 days (type 1)',
                    'ret2_days70': 'Test return within 70 days (type 2)',
                    }

    for c, name in cat_cols.items():
        print(c)
        row = summarise_cat(df, c, name)
        row.Category = row.Category.astype(str)
        row = row.sort_values(by='Category')
        if 'NULL' in row.Category.tolist():
            row0 = row.loc[row.Category != 'NULL']
            row1 = row.loc[row.Category == 'NULL']
            row = pd.concat(objs=[row0, row1], axis=0)
        desc = pd.concat(objs=[desc, row], axis=0)

    desc = desc.reset_index(drop=True)

    test = desc.Value < 10
    assert not any(test)

    return desc


# Read data
df = pd.read_csv(data_path / 'first_fit_nonret.csv')
df_all = pd.read_csv(data_path / 'fit_nonret.csv')

# Subset to 70 day follow-up
test = df.nonret1_days70.isna() == df.nonret2_days70.isna()
test.all()

print(df.shape, df_all.shape)
df = df.loc[~df.nonret1_days70.isna()]
df_all = df_all.loc[~df_all.nonret1_days70.isna()]
print(df.shape, df_all.shape)

assert df.fit_request_date_fu.min() >= 70
assert df_all.fit_request_date_fu.min() >= 70

# Remove patients who died before 70 day follow-up without returning a test
print(df.shape, df_all.shape)
df = df.loc[~((df.days_to_death <= 70) & (df.censored == 1))]
df_all = df_all.loc[~((df_all.days_to_death <= 70) & (df_all.censored == 1))]
print(df.shape, df_all.shape)


df.gender_male = df.gender_male.replace({0: 'Not male', 1: 'Male'})
df.imd_quintile = df.imd_quintile.fillna('Not known')

outcomes = ['nonret1_days14', 'nonret1_days28', 'nonret1_days70', 
            'nonret2_days14', 'nonret2_days28', 'nonret2_days70']

for o in outcomes:
    new_name = o[3:]
    df[new_name] = 1 - df[o]
    df[new_name] = df[new_name].replace({0: 'No return', 1: 'Return'})
    
    #days = int(re.findall('days(\d+)', o)[0])
    #test = (df.censored == 1) & (df.days_to_death <= days)
    #print(test.sum())
    #df.loc[test, new_name] = 'Not known'


desc = describe(df, df_all, desc_return=True)
desc.Category = desc.Category.replace({'NULL': 'Not known'})

assert (desc.Value > 10).all()

desc.to_csv(out_path / 'descriptives_all.csv', index=False)

desc_reformat = pd.DataFrame()
for cat in desc.Characteristic.drop_duplicates():
    d = desc.loc[desc.Characteristic == cat].drop(labels=['Characteristic'], axis=1)
    row = pd.DataFrame({'Category': [cat], 'Value': [''], 'Percent': [''], 'Value (percent)': ['']})
    d = pd.concat(objs=[row, d], axis=0)
    desc_reformat = pd.concat(objs=[desc_reformat, d], axis=0)
desc_reformat = desc_reformat.rename(columns={'Category': 'Characteristic'})
desc_reformat.to_csv(out_path / 'descriptives_all_reformat.csv', index=False)



## ---- Descriptives by return ----
outcomes = ['nonret1_days14', 'nonret1_days28', 'nonret1_days70', 
            'nonret2_days14', 'nonret2_days28', 'nonret2_days70']

import re

npat = df.shape[0]

for i, o in enumerate(outcomes):
    print(o)
    days = int(re.findall('days(\d+)', o)[0])
    test = (df.censored == 1) & (df.days_to_death <= days)

    df0 = df.loc[df[o] == 0]
    df1 = df.loc[df[o] == 1]

    assert df0.shape[0] + df1.shape[0] == npat
    df_all0 = df_all.loc[df_all[o] == 0]
    df_all1 = df_all.loc[df_all[o] == 1]

    desc0 = describe(df0, df_all0)
    desc1 = describe(df1, df_all1)

    assert (desc0.Value > 10).all()
    assert (desc1.Value > 10).all()

    days = re.findall('days(\d+)', o)[0]
    return_type = re.findall('nonret(\d+)', o)[0]

    desc1 = desc1.drop(labels=['Value', 'Percent'], axis=1).rename(columns={'Value (percent)': 'No return within ' + days + ' days (type ' + return_type + ')'})
    desc0 = desc0.drop(labels=['Value', 'Percent'], axis=1).rename(columns={'Value (percent)': 'Test return within ' + days + ' days (type ' + return_type + ')'})

    desc01 = desc0.merge(desc1, how='outer')

    if i == 0:
        desc = desc01
    else:
        desc = desc.merge(desc01, how='outer', on=['Characteristic', 'Category'])

out_name = 'descriptives_by-return.csv'
desc.to_csv(out_path / out_name, index=False)

desc_reformat = pd.DataFrame()
for cat in desc.Characteristic.drop_duplicates():
    d = desc.loc[desc.Characteristic == cat].drop(labels=['Characteristic'], axis=1)
    row = pd.DataFrame({'Category': [cat], 'Value': [''], 'Percent': [''], 'Value (percent)': ['']})
    d = pd.concat(objs=[row, d], axis=0)
    desc_reformat = pd.concat(objs=[desc_reformat, d], axis=0)
desc_reformat = desc_reformat.rename(columns={'Category': 'Characteristic'})
desc_reformat.to_csv(out_path / 'descriptives_by-return_reformat.csv', index=False)

    
