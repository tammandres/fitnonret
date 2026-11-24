import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import re


# Path
data_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\results_fixed')
in_path = out_path


# Helper functions
def summarise_cat(df, col, name, digits=2, sort=True):
    """Summarise categorical data"""
    s = df[col]

    # Designate missing values with 'NULL'
    if (s == 'NULL').any():
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


# ---- Descriptives table
#region

# Read data
df = pd.read_csv(data_path / 'first_fit_nonret_fixed.csv')
df_all = pd.read_csv(data_path / 'fit_nonret_fixed.csv')

# Subset to 70 day follow-up
test = df.nonret1_days70.isna() == df.nonret2_days70.isna()
assert test.all()

test = df_all.nonret1_days70.isna() == df_all.nonret2_days70.isna()
assert test.all()

print(df.shape, df_all.shape)
df = df.loc[~df.nonret1_days70.isna()]
df_all = df_all.loc[~df_all.nonret1_days70.isna()]
print(df.shape, df_all.shape)

assert df.fit_request_date_fu.min() >= 70
assert df_all.fit_request_date_fu.min() >= 70

df.gender_male = df.gender_male.replace({0: 'Not male', 1: 'Male'})
df.imd_quintile = df.imd_quintile.fillna('Not known')

outcomes = ['nonret1_days14', 'nonret1_days28', 'nonret1_days70', 
            'nonret2_days14', 'nonret2_days28', 'nonret2_days70']

for o in outcomes:
    new_name = o[3:]
    df[new_name] = 1 - df[o]
    df[new_name] = df[new_name].replace({0: 'No return', 1: 'Return'})
    
desc = describe(df, df_all, desc_return=True)
desc.Category = desc.Category.replace({'NULL': 'Not known'})

assert (desc.Value >= 10).all()

desc.to_csv(out_path / 'descriptives_all.csv', index=False)

desc_reformat = pd.DataFrame()
for cat in desc.Characteristic.drop_duplicates():
    d = desc.loc[desc.Characteristic == cat].drop(labels=['Characteristic'], axis=1)
    row = pd.DataFrame({'Category': [cat], 'Value': [''], 'Percent': [''], 'Value (percent)': ['']})
    d = pd.concat(objs=[row, d], axis=0)
    desc_reformat = pd.concat(objs=[desc_reformat, d], axis=0)
desc_reformat = desc_reformat.rename(columns={'Category': 'Characteristic'})
desc_reformat.to_csv(out_path / 'descriptives_all_reformat.csv', index=False)


## Descriptives by return 
outcomes = ['nonret1_days14', 'nonret1_days28', 'nonret1_days70', 
            'nonret2_days14', 'nonret2_days28', 'nonret2_days70']

import re

npat = df.shape[0]
desc = pd.DataFrame()

for i, o in enumerate(outcomes):
    print(o)
    days = int(re.findall(r'days(\d+)', o)[0])

    df0 = df.loc[df[o] == 0]
    df1 = df.loc[df[o] == 1]

    assert df0.shape[0] + df1.shape[0] == npat  # if someone had 70-day fu, they also had 28 and 14 day fu, hence okay to use common npat
    df_all0 = df_all[df_all.patient_id.isin(df0.patient_id)]
    df_all1 = df_all[df_all.patient_id.isin(df1.patient_id)]

    desc0 = describe(df0, df_all0)
    desc1 = describe(df1, df_all1)

    assert (desc0.Value >= 10).all()
    assert (desc1.Value >= 10).all()

    desc0['nonret'] = 0
    desc1['nonret'] = 1

    days = re.findall(r'days(\d+)', o)[0]
    return_type = re.findall(r'nonret(\d+)', o)[0]

    desc01 = pd.concat(objs=[desc0, desc1], axis=0)
    desc01['return_type'] = return_type
    desc01['days'] = days

    desc = pd.concat(objs=[desc, desc01], axis=0)

assert desc.Value.min() >= 10

out_name = 'descriptives_by-return.csv'
desc.to_csv(out_path / out_name, index=False)
#endregion


# ---- Return probabilities from KM curve
#region
df = pd.read_csv(r'Z:\fit_nonreturn_paper_20250417\results_fixed\km_curve_data-full.csv')
df = df.loc[df.return_type == 2]
df['p_return'] = 1 - df.surv
s = df.loc[df.time.isin([10, 14, 28, 70, 365]), 
           ['time', 'p_return', 'p14', 'p28', 'p70']]
cols = ['p_return','p14', 'p28', 'p70']
for c in cols:
    s[c] = s[c].round(3)*100
print(s)
s['type'] = 'km'
s.to_csv(out_path / 'return_probabilities_km.csv', index=False)
#endregion


# ---- Return probabilities for patients with 70-day fu
#region
df2 = pd.read_csv(r'Z:\fit_nonreturn_paper_20250417\data\first_fit_nonret_fixed.csv')
df2 = df2.loc[df2.fit_request_date_fu >= 70]

s = pd.DataFrame()
for days in [10, 14, 28, 70]:
    out = (df2.days_to_return <= days).agg(['sum', 'mean'])
    out = pd.DataFrame(out).transpose()
    out.columns= ['n', 'p_return']
    out.p_return *= 100
    out.index = [days]
    s = pd.concat(objs=[s, out], axis=0)
s.index.name = 'time'
s.n = s.n.astype(int)
s = s.reset_index()
s.p_return = s.p_return.round(1)
print(s)
s['type'] = '70-day fu'

s.to_csv(out_path / 'return_probabilities_fu-70.csv', index=False)
#endregion


# ---- Return rate over time
#region
df2 = pd.read_csv(r'Z:\fit_nonreturn_paper_20250417\data\first_fit_nonret_fixed.csv')
df2 = df2.loc[df2.fit_request_date_fu >= 70]
df2.shape

df2.fit_request_date_corrected = pd.to_datetime(df2.fit_request_date_corrected)
df2['fit_request_quarter'] = df2.fit_request_date_corrected.dt.quarter
df2['fit_request_year'] = df2.fit_request_date_corrected.dt.year

yearly_return = df2.groupby('fit_request_year').nonret2_days70.mean() * 100
yearly_return = 100 - yearly_return
yearly_volume = df2.groupby('fit_request_year').size()

data = pd.concat(objs=[yearly_return, yearly_volume], axis=1)
data.columns = ['return_rate', 'testing_volume']
data = data.loc[data.index < 2024]
data = data.reset_index()

data.to_csv(out_path / 'request_volume_over_time.csv', index=False)

import matplotlib.pyplot as plt
plt.scatter(data.testing_volume, data.return_rate)
plt.show()
#endregion