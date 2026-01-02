import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


data_path = Path('Z:/fit_dataproduct_20240626')


# ---- 1. Read data ----
#region

# Read FIT requests
gpreqs = pd.read_parquet(data_path / 'gpreqs')
print(gpreqs.shape[0], gpreqs.patient_id.nunique())  # 95418 rows, 71971 patient IDs

# Read FIT values
lims = pd.read_parquet(data_path / 'fit_values')
lims['gp_from_icen_and_loc'] = lims['gp_from_icen_and_loc'].astype(int)
lims.fit_val_clean = lims.fit_val_clean.astype(float)
gplims = lims.loc[lims.gp_from_icen_and_loc==1].copy()  # Retain GP FITs

# Dbl check number of patients and data types
print(gpreqs.patient_id.nunique())
print(gplims.patient_id.nunique())
gplims.dtypes
gpreqs.dtypes

# Dates to datetime? Not necessary atm because data is in parquet format
dates_to_datetime = False
if dates_to_datetime:
    DATE_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
    gpreqs.fit_request_date = pd.to_datetime(gpreqs.fit_request_date, format=DATE_FORMAT)
    gpreqs.fit_request_or_sample_date = pd.to_datetime(gpreqs.fit_request_or_sample_date, format=DATE_FORMAT)
    gplims.fit_date = pd.to_datetime(gplims.fit_date, format=DATE_FORMAT)
    gplims.fit_date_received = pd.to_datetime(gplims.fit_date_received, format=DATE_FORMAT)
print(gpreqs.fit_request_date.max(), gplims.fit_date_authorised.max())
print(gpreqs.fit_request_date.min(), gplims.fit_date_authorised.min())

# Check: icen in gplims are almost always in gpreqs, with 103 expections
gplims.dropna(subset='icen').icen.isin(gpreqs.icen).mean()
(~gplims.dropna(subset='icen').icen.isin(gpreqs.icen)).sum()

# Check - each icen number always associated with one patient
test = pd.concat(objs=[gplims[['patient_id', 'icen']], gpreqs[['patient_id', 'icen']]], axis=0)
test = test.dropna(subset='icen').drop_duplicates()
assert test.groupby('icen').patient_id.nunique().max() == 1

# Check - one row per ICEN number in gpreqs
assert gpreqs.icen.nunique() == gpreqs.shape[0]

#endregion


# ---- 2. Double check time deltas ----
#region

# fit_request_or_sample_date vs fit_request_date
#  When fit_request_or_sample date and fit_request_date are both available
#  then fit_request_date is smaller in about 71% of cases, equal in about 28% of cases, and greater in about 0.7% of cases
delta = gpreqs.fit_request_or_sample_date - gpreqs.fit_request_date 
delta = delta[~gpreqs.fit_request_date.isna()]
delta.describe(percentiles=[0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 0.99, 0.999, 0.9999])

test = gpreqs.fit_request_or_sample_date > gpreqs.fit_request_date 
test = test[~gpreqs.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date is smaller than fit_request_or_sample_date 71% of times

test = gpreqs.fit_request_or_sample_date == gpreqs.fit_request_date 
test = test[~gpreqs.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date is equal to fit_request_or_sample_date 28% of times

test = gpreqs.fit_request_or_sample_date < gpreqs.fit_request_date 
test = test[~gpreqs.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date is greater than fit_request_or_sample_date 0.7% of times (580 cases)

# fit_request_or_sample_date vs fit_date
tmp = gplims[['patient_id', 'icen', 'fit_date', 'fit_date_received']].\
    merge(gpreqs[['patient_id', 'icen', 'fit_request_date', 'fit_request_or_sample_date']], how='left')
tmp.isna().sum()

test = tmp.fit_request_or_sample_date == tmp.fit_date
test = test.loc[~tmp.fit_request_or_sample_date.isna()]
print(test.mean(), test.sum())  # fit_request_or_sample_date equal to fit_date 99% of times when both available

test = tmp.fit_request_date < tmp.fit_date
test = test.loc[~tmp.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date smaller than fit_date 71% of times when both available

test = tmp.fit_request_date == tmp.fit_date
test = test.loc[~tmp.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date equal to fit_date 28% of times when both available

test = tmp.fit_request_date < tmp.fit_date_received
test = test.loc[~tmp.fit_request_date.isna()]
print(test.mean(), test.sum())  # fit_request_date smaller than fit_date_received 99% of times when both available

# fit_date_received vs fit_date_authorised: date authorised always greater
delta = lims.fit_date_authorised - lims.fit_date_received
delta.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.9, 0.95, 0.99])

# fit_date_received vs fit_date: equal about 30% of times, date_received greater about 70% of times
delta = lims.fit_date_received - lims.fit_date
delta.describe(percentiles=[0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3, 0.9, 0.95, 0.99])

#endregion


# ---- 3. Correct fit request and sample dates ----
#region

# Correct the FIT request date
#  Note that fit_request_or_sample_date is by default the request date. 
#   when the sample is received in the lab, 
#   it is changed to sample date (or to receipt date if sample date was not known)
#   and in that case the 'fit_request_date' column is separately given as the original date
# 
#  This is confirmed by Brian Shine, and also the following:
#  (1) "fit_request_or_sample_date" is 99% of times equal to "fit_date" in gplims when both dates are available
#  (2) when "fit_request_date" is not available, then 97% of times there's no ICEN number in lims table,
#      and when it is available, then 99% of times there is ICEN number in lims table.
#
#  Therefore, "fit_request_date_corrected" is set equal to "fit_request_date" (dtc2, dtcorig) when that is available
#  and otherwise set equal to "fit_request_or_sample_date" (dtc)

# .... 3.1. Correct the FIT request date ....
mask = ~gpreqs.fit_request_date.isna()
gpreqs['fit_request_date_corrected'] = gpreqs.fit_request_or_sample_date.copy()
gpreqs.loc[mask, 'fit_request_date_corrected'] = gpreqs.loc[mask].fit_request_date
gpreqs.fit_request_date_corrected.isna().sum()
delta = gpreqs.fit_request_or_sample_date - gpreqs.fit_request_date_corrected
delta.describe(percentiles=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.9, 0.99, 0.999, 0.9999])
delta.loc[mask].describe(percentiles=[0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.9, 0.99, 0.999, 0.9999])

# Check: Does the absence of fit_request_date indicate that test was not received in the lab with its request number?
# It seems to be so with high probability: when there's no dtc2, there's no ICEN in lims table 97% of times
# When there is dtc2, there is icen in lims 98% of times
# Not sure why it is not 100% -- even when applying lower date cutoff to gpreqs this pattern persists.
gpreqs['icen_in_lims'] = gpreqs.icen.isin(gplims.icen).astype(int)
gpreqs['no_dtc2'] = gpreqs.fit_request_date.isna().astype(int)

test = gpreqs.merge(gplims[['patient_id', 'icen', 'fit_date', 'fit_val', 'fit_val_clean']], how='left')
test.groupby('no_dtc2').icen_in_lims.value_counts(sort=False)
test.groupby('no_dtc2').icen_in_lims.value_counts(sort=False, normalize=True)

# Compute length of follow-up from FIT request date to datacut of lab results 
# Note: the lab results were extracted in February 2024, but date shifts were applied
# the max date in the processed lab data is thus different than datacut
# Solution: use max date observed in the data
# atm, do not use datacut, but max fit date received - as the latter is slightly larger
max_date = gplims.fit_date_received.max()
gpreqs['fit_request_date_fu'] = max_date - gpreqs.fit_request_date_corrected
gpreqs['fit_request_date_fu'] = gpreqs['fit_request_date_fu'].dt.days + gpreqs['fit_request_date_fu'].dt.seconds / (60 * 60 * 24)
gpreqs['fit_request_date_fu'].describe()
gpreqs.fit_request_date_fu.min()

# .... 3.2. Correct the FIT date ....
#  Sometimes FIT sample date is equal to FIT request date.
#  In these cases, set it equal to FIT date received instead.
#  It looks like these are examples where the sample date was not known
#  and was set equal to request date (rather than date received)
lims['fit_date_corrected'] = lims.fit_date.copy()
print(lims.shape)
lims = lims.merge(gpreqs[['patient_id', 'icen', 'fit_request_date', 'fit_request_date_corrected', 'fit_request_or_sample_date']], how='left')
print(lims.shape)
mask = lims.fit_date == lims.fit_request_date_corrected
print(mask.sum())
lims.loc[mask, 'fit_date_corrected'] = lims.loc[mask, 'fit_date_received']
gplims = lims.loc[lims.gp_from_icen_and_loc == 1]  # extract gplims again

# Double check time deltas again
gplims.isna().sum()  # in gplims, fit_request_date not known about 6477 times, fit_Request_date_corrected not known 5997 times

test = gplims.fit_date_corrected == gplims.fit_date_received
test = test.loc[~gplims.fit_date_corrected.isna()]
print(test.sum(), test.mean())  # In about 55% of times, fit sample date is not known and equal to date received

test = gplims.fit_date_corrected == gplims.fit_request_date_corrected
test = test.loc[~gplims.fit_request_date_corrected.isna()]
print(test.sum(), test.mean())  # In 69 times, fit_date_corrected equal to fit_request_date

test = gplims.fit_date_corrected > gplims.fit_request_date_corrected
test = test.loc[~gplims.fit_request_date_corrected.isna()]
print(test.sum(), test.mean())  # In 99% of times, fit_date_corrected is greater than fit_request_date_corrected

# Review time from request, to sample, to received
delta = gplims.fit_date_corrected - gplims.fit_request_date_corrected
delta2 = gplims.fit_date_received - gplims.fit_request_date_corrected
print(delta.describe(percentiles=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]))
print(delta2.describe(percentiles=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]))

# Another check 
gplims.isna().sum()
mask = gplims.fit_date > gplims.fit_date_received
print(mask.sum(), mask.mean())  ## FIT date is never greater than date received

mask = gplims.fit_date_corrected > gplims.fit_date_received
print(mask.sum(), mask.mean())  ## FIT date corrected is never greater than date received

mask = gplims.fit_request_date_corrected > gplims.fit_date_received
print(mask.sum(), mask.mean())  ## FIT request date corrected is greater than date received in 325 cases

mask = gplims.fit_request_or_sample_date > gplims.fit_date_received
print(mask.sum(), mask.mean())  ## FIT request date corrected is greater than date received in 325 cases

delta = gplims.fit_date_received - gplims.fit_date_corrected  #
delta.describe(percentiles=[0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99])

# Indicator for sample date not known
gplims['fit_sample_date_not_known'] = (gplims.fit_date_corrected == gplims.fit_date_received).astype(int)
print(gplims.fit_sample_date_not_known.mean())

#endregion


# ---- 4. Get time to return and non-return indicators ----
#region

# Dbl check date ranges
print(gpreqs.fit_request_date_corrected.min(), gpreqs.fit_request_date_corrected.max())
print(gplims.fit_request_date_corrected.min(), gplims.fit_request_date_corrected.max())

# Retain relevant columns
gpreqs_sub = gpreqs[['patient_id', 'icen', 'gp', 'fit_request_date_corrected', 'icen_in_lims', 'fit_request_date_fu']]
gplims_sub = gplims[['patient_id', 'icen', 'fit_date_corrected', 'fit_date_received', 'fit_date_authorised', 
                     'fit_request_or_sample_date', 'fit_sample_date_not_known',
                     'fit_val', 'fit_val_clean']]

# Container for requests to be removed
icen_rm = pd.Series()

#--------
# 4.1. FIT requests, where the request number appears in the lab results
#--------

# Get all FIT results that have icen number
gplims_with_icen = gplims_sub.dropna(subset=['icen']).drop_duplicates()

# In a few cases, there are multiple results for the same request number
# Select the first returned result
#gplims_with_icen.loc[gplims_with_icen.icen.duplicated(keep=False)]  
gplims_with_icen = gplims_with_icen.sort_values(by=['patient_id', 'icen', 'fit_date_received'])
gplims_with_icen = gplims_with_icen.groupby(['icen']).first().reset_index()
assert gplims_with_icen.shape[0] == gplims_with_icen.icen.nunique()

# Then add results to requests
g0 = gpreqs_sub.merge(gplims_with_icen, how='inner', on=['patient_id', 'icen'])
assert g0.shape[0] == g0.icen.nunique()
g0['return_type'] = 'icen_in_lims'
print(g0.shape[0])

# Remove requests where request date after FIT date received (323 requests where request appeared in lab under same num
# but request date was after FIT date; 323 patients
i = g0.loc[g0.fit_date_received < g0.fit_request_date_corrected].icen
print(i.nunique())
g0.loc[g0.icen.isin(i)].patient_id.nunique()
g0 = g0.loc[~g0.icen.isin(i)]
icen_rm = pd.concat(objs=[icen_rm, i])
print(g0.shape[0])
gpreqs_sub = gpreqs_sub.loc[~gpreqs_sub.icen.isin(i)]

#--------
# 4.2. FIT requests, where the patient does not appear in the lab table at all
#--------
g1 = gpreqs_sub.loc[~gpreqs_sub.patient_id.isin(gplims.patient_id)]
g1['return_type'] = 'no_lims'
print(g1.shape[0])
assert g1.icen.nunique() == g1.shape[0]

#--------
# 4.3. FIT requests, where the request number does not appear in lab table, 
# but there is a test result with no request number in the lab table
# and no other test request occurred in between
#--------

# FIT results without request number
gplims_unlabelled = gplims_sub.loc[gplims_sub.icen.isna()].rename(columns={'icen': 'icen_lims'})

# Merge to requests on patient ID
g2 = gpreqs_sub.loc[gpreqs_sub.icen_in_lims==0].merge(gplims_unlabelled, on=['patient_id'], how='inner')
assert not g2.fit_date_received.isna().any()

# Remove where lab result before request (439 requests)
g2.shape[0]
g2 = g2.loc[g2.fit_date_received >= g2.fit_request_date_corrected]
g2.shape[0]

# Remove results where another request number appeared before the unlabelled FIT was received
print(g2.shape[0])
r = gpreqs_sub[['patient_id', 'fit_request_date_corrected', 'icen']].dropna(subset=['icen'])
r = r.rename(columns={'fit_request_date_corrected':'request_date2', 'icen': 'icen2'})
tmp = g2.merge(r, on='patient_id', how='left')
mask = (tmp.request_date2 <= tmp.fit_date_received) & \
       (tmp.request_date2 >= tmp.fit_request_date_corrected) & \
       (tmp.icen != tmp.icen2)
tmp = tmp.loc[mask]
tmp.loc[mask, ['patient_id', 'fit_request_date_corrected', 'request_date2', 'icen', 'icen2', 'fit_date_received', 'fit_val']]
g2 = g2.loc[~g2.icen.isin(tmp.icen)]
print(g2.shape[0])

# Get the most recent unlabelled result
idxmin = g2.groupby('icen')['fit_date_received'].idxmin()
g2 = g2.loc[idxmin]
print(g2.shape[0])
print(g2.icen.isna().sum(), g2.icen_lims.isna().sum())
g2['return_type'] = 'unlabelled_kit'
g2 = g2.drop(labels=['icen_lims'], axis=1)
print(g2.shape[0])
assert g2.icen.nunique() == g2.shape[0]

#--------
# 4.4. FIT requests, where result from another request number occurs nearby
#--------

gplims_labelled = gplims_sub.loc[~gplims_sub.icen.isna()].rename(columns={'icen': 'icen_lims'})

# Merge to requests on patient ID
g3 = gpreqs_sub.loc[gpreqs_sub.icen_in_lims==0].merge(gplims_labelled, on=['patient_id'], how='inner')
assert not g3.fit_date_received.isna().any()

# Remove where lab result before request (3966 -- these are not errors as merging data with another request num)
g3.shape[0]
g3 = g3.loc[g3.fit_date_received >= g3.fit_request_date_corrected]
g3.shape[0]

# Remove where there was already an unlabelled result before
g3 = g3.loc[~g3.icen.isin(g2.icen)]

# Get the most recent labelled result
idxmin = g3.groupby('icen')['fit_date_received'].idxmin()
g3 = g3.loc[idxmin]
print(g3.shape[0])
print(g3.icen.isna().sum(), g3.icen_lims.isna().sum())
assert g3.icen.nunique() == g3.shape[0]
assert (~g3.icen.isin(g2.icen)).all()
g3['return_type'] = 'kit_from_another_icen'
g3 = g3.drop(labels=['icen_lims'], axis=1)
print(g3.shape[0])

# --------
# 4.5. Combine
# --------

# Concatenate
g = pd.concat(objs=[g0, g1, g2, g3], axis=0)
assert g.shape[0] == g.icen.nunique()
assert g.patient_id.isna().sum() == 0

# Add remaining requests
g_add = gpreqs_sub.loc[~gpreqs_sub.icen.isin(g.icen)]
print(g_add.shape[0])
g_add['return_type'] = 'nonreturn_other'
g = pd.concat(objs=[g, g_add], axis=0)

assert g.shape[0] == g.icen.nunique()
assert g.icen.isin(gpreqs.icen).all()
assert gpreqs_sub.icen.isin(g.icen).all()
assert g.shape[0] == g.icen.nunique()
assert not g.return_type.isna().any()
assert g.patient_id.isna().sum() == 0
print(g.shape[0])

g.return_type.value_counts()

## Note: at this stage, g includes all requests, except 323 icen where lab date before request date for requests 
## where icen in lims 
gpreqs.shape[0] - gpreqs_sub.shape[0]

# Days to return
g['days_to_return'] = (g.fit_date_received - g.fit_request_date_corrected).dt.days
assert (g.days_to_return.dropna() >= 0).all()

# Drop patients with type 1 return after 365 days (36 icen, 36 patients)
# No need to drop type 2 return
t = g.loc[(~g.days_to_return.isna()) & (g.return_type != 'kit_from_another_icen')]
t.return_type.unique()
t.days_to_return.describe(percentiles=[0.9, 0.95, 0.99])
(t.days_to_return > 180).sum()
(t.days_to_return > 365).sum()
icen_rm2 = t.loc[t.days_to_return > 365].icen
print(icen_rm2.nunique())
print(g.shape)
g = g.loc[~g.icen.isin(icen_rm2)]
print(g.shape)

tsub = t.loc[t.days_to_return > 365]
tsub.patient_id.nunique()
assert g.shape[0] + 36 + 323 == gpreqs.shape[0]

# For type 2 returns where return time is after 365 days, 
# set the time to 366 days (so that indicator for 365 day nonreturn is not affected)
# but the KM-curve, if calculated, wouldn't go past 365 days.
mask = g.days_to_return > 365  # mask.sum() == 817
assert (g.loc[mask, 'return_type'] == 'kit_from_another_icen').all()
g['type2_return_after_365days'] = 0
g.loc[mask, 'type2_return_after_365days'] = 1
g.loc[mask, 'days_to_return'] = 366

# Add death date
demo = pd.read_parquet(data_path / 'demographics')
death = demo[['patient_id', 'death_date']].drop_duplicates()
shape0 = g.shape[0]
g = g.merge(death, how='left').merge(death, how='left')
assert g.shape[0] == shape0

# Get overall censoring indicator and return indicator
g.days_to_return.isna().sum()
g.days_to_return.isna().mean()
g['censored'] = g.days_to_return.isna().astype(int)
g['fit_return'] = (1 - g.censored).astype(int)

print(g.loc[g.return_type != 'kit_rom_another_icen'].days_to_return.max())

# For censored tests, set days_to_return as days to datacut (if no death), or days to death (if death)
print(g.days_to_return.max())
g['days_to_death'] = None
mask = ~g.death_date.isna()
g.loc[mask, 'days_to_death'] = (pd.to_datetime(g.loc[mask, 'death_date']) - g.loc[mask, 'fit_request_date_corrected']).dt.days

mask = (g.censored == 1) & (~g.death_date.isna())
print(mask.sum())  # 922
g.loc[mask, 'days_to_return'] = g.loc[mask, 'days_to_death']

mask = (g.censored == 1) & (g.death_date.isna())
print(mask.sum())  #9164
g.loc[mask, 'days_to_return'] = g.loc[mask, 'fit_request_date_fu']

assert (g.days_to_return >= 0).all()

# Given that we consider max 365 day fu (longer returns excluded), set max censored days to return to 365
g.loc[(g.censored == 1) & (g.days_to_return > 365), 'days_to_return'] = 365

# Get type 1 return, where tests under another request number are not considered
g['days_to_return_type1'] = (g.fit_date_received - g.fit_request_date_corrected).dt.days
assert (g.days_to_return_type1.dropna() >= 0).all()
mask = (g.return_type == 'kit_from_another_icen')
assert mask.sum() > 0
g.loc[mask, 'days_to_return_type1'] = np.nan
g['censored_type1'] = g.days_to_return_type1.isna().astype(int)
g['fit_return_type1'] = (1 - g.censored_type1).astype(int)

mask = (g.censored_type1 == 1) & (~g.death_date.isna())
print(mask.sum())  # 1223
g.loc[mask, 'days_to_return_type1'] = g.loc[mask, 'days_to_death']

mask = (g.censored_type1 == 1) & (g.death_date.isna())
print(mask.sum())  #12823
g.loc[mask, 'days_to_return_type1'] = g.loc[mask, 'fit_request_date_fu']

g.loc[(g.censored_type1 == 1) & (g.days_to_return > 365), 'days_to_return_type1'] = 365
assert (g.days_to_return_type1 >= 0).all()
print(g.fit_return.mean(), g.fit_return_type1.mean())

assert g.patient_id.isna().sum() == 0

# Get nonreturn indicators. E.g. if d = 70, then 
# "A test is not returned, if there is no evidence of return within 70 days"
#  Patients without 70-day follow-up are excluded (indicator set to nan)
#  Patients who returned a test after 70 days are considered nonreturns.
days = [7, 14, 28, 70, 180, 365]
for d in days:
    colname = 'nonret2_days' + str(d)
    g[colname] = 1
    g.loc[(g.days_to_return <= d) & (g.fit_return == 1), colname] = 0
    g.loc[g.fit_request_date_fu < d, colname] = np.nan
    print(d, g.loc[~g[colname].isna(), colname].mean())

days = [7, 14, 28, 70, 180, 365]
for d in days:
    colname = 'nonret1_days' + str(d)
    g[colname] = 1
    g.loc[(g.days_to_return_type1 <= d) & (g.fit_return_type1 == 1), colname] = 0
    g.loc[g.fit_request_date_fu < d, colname] = np.nan
    print(d, g.loc[~g[colname].isna(), colname].mean())


# Explore: how many requests were made where a patient died before 70 day return?
test = (g.censored == 1) & (g.days_to_death < 70)
print(test.sum())

test = (g.censored_type1 == 1) & (g.days_to_death < 70)
print(test.sum())

# Dbl check multiple requests occuring on the same day
#  For 72 patients, requests occur on the same day with dif request number
#  However, their return status is always the same (not returned or returned), and days_to_return as well
#  It is therefore fine to just randomly select one among those
mask = g[['patient_id', 'fit_request_date_corrected']].duplicated(keep=False)
mask.sum()  # 144 rows
g.loc[mask].patient_id.nunique()  # 72 patients
gsub = g.loc[mask]
test = gsub.groupby(['patient_id', 'fit_request_date_corrected'])['fit_return'].nunique()
assert test.min() == 1
test = gsub.groupby(['patient_id', 'fit_request_date_corrected'])['days_to_return'].nunique()
assert test.min() == 1

assert gsub.icen.isna().sum() == 0
gsub_select = gsub.groupby(['patient_id', 'fit_request_date_corrected']).first() # 72 rows
gsub_select = gsub_select.reset_index()

print(g.shape)
g = pd.concat(objs=[g.loc[~mask], gsub_select], axis=0)
g = g.sort_values(by=['fit_request_date_corrected', 'patient_id'])
print(g.shape)

assert g.patient_id.isna().sum() == 0
assert g.icen.nunique() == g.shape[0]
assert g.shape[0] + 323 + 36 + (144 - 72) == gpreqs.shape[0]

#endregion


# ---- 5. Add demographics ----
#region

# Add demographics
demo = pd.read_parquet(data_path / 'demographics')
imd = demo.groupby('patient_id').index_of_multiple_deprivation.max().reset_index()
dsub = demo[['patient_id', 'gender_code', 'ethnic_group_code', 'year_of_birth', 'month_of_birth']].drop_duplicates()
shape0 = g.shape[0]
g = g.merge(imd, how='left').merge(dsub, how='left')
assert g.shape[0] == shape0

# Compute depriation quintile (instead of decile)
g.index_of_multiple_deprivation.value_counts()
imd_map = {1: [1, 2], 
           2: [3, 4],
           3: [5, 6],
           4: [7, 8],
           5: [9, 10]
           }
imd_map_inverse = {}
for key, val in imd_map.items():
    for v in val:
        imd_map_inverse[v] = key
g['imd_quintile'] = g.index_of_multiple_deprivation.replace(imd_map_inverse)
assert all(g.imd_quintile.isna() == g.index_of_multiple_deprivation.isna())
g.groupby('imd_quintile').index_of_multiple_deprivation.unique()

# Compute age with monthly margin of error (i.e. max error 30 days), remove age < 18
g['birth_date_approx'] = pd.to_datetime(g.year_of_birth.astype(str) + '-' + g.month_of_birth.astype(str) + '-01')
g['age_at_request'] = (g.fit_request_date_corrected - g.birth_date_approx).dt.days / 365
assert not g.age_at_request.isna().any()
i = g.loc[g.age_at_request < 18].icen  # 214 requests
g.loc[g.icen.isin(i)].patient_id.nunique()  # 203 patients
g = g.loc[~g.icen.isin(i)]
icen_rm = pd.concat(objs=[icen_rm, i], axis=1).drop_duplicates()
assert g.shape[0] + 323 + 36 + (144 - 72) + 214 == gpreqs.shape[0]

##  Total num removed: 645 (0.68%)
num_rm = gpreqs.shape[0] - g.shape[0]
num_rm / gpreqs.shape[0] * 100

# Compute year and month of request
g['request_year'] = g.fit_request_date_corrected.dt.year
g['request_month'] = g.fit_request_date_corrected.dt.month

# Simplify ethnicity
# https://www.datadictionary.nhs.uk/data_elements/ethnic_category.html
g.ethnic_group_code.value_counts()
ethnic_map = {'White': ['A', 'B', 'C'], 
              'Mixed': ['D', 'E', 'F', 'G'],
              'Asian': ['H', 'J', 'K', 'L'],
              'Black': ['M', 'N', 'P'],
              'Other': ['R', 'S'],
              'Not stated': ['Z'],
              'Not known': ['99']}
vals = list(ethnic_map.values())
vals = [a for b in vals for a in b]
codes = g.ethnic_group_code.unique()
codes = [c for c in codes if c is not None]
assert all(v in vals for v in codes)

g['ethnicity'] = np.nan
for key, val in ethnic_map.items():
    mask = g.ethnic_group_code.isin(val)
    g.loc[mask, 'ethnicity'] = key
g.ethnicity.value_counts()
assert all(g.ethnicity.isna() == g.ethnic_group_code.isna())

g.loc[g.ethnicity.isna(), 'ethnicity'] = 'Not known'
g.ethnicity.value_counts()
assert (~g.ethnicity.isna().any())

# Male indicator for gender 
g['gender_male'] = (g.gender_code == 'M').astype(int)
g.gender_male.mean()

# Ethnicity to dummies (drop White, so it is the reference)
eth_ind = pd.get_dummies(g.ethnicity, drop_first = False)
eth_ind = eth_ind.astype(int)
eth_ind.columns = eth_ind.columns.str.lower()
eth_ind = eth_ind.loc[:, eth_ind.columns != 'white']
eth_ind.columns = ['ethnicity_' + c for c in eth_ind.columns]
print(g.shape)
g = pd.concat(objs=[g, eth_ind], axis=1)
print(g.shape)

# Age group
age = g.age_at_request
print(age.isna().sum())
age_max = np.round(age.max() + 1, 1)
age_group = pd.cut(age, bins=[18, 40, 50, 60, 70, 80, 90, age_max], right=False)
age_group.value_counts(sort=False)

g['age_group'] = age_group.astype(str)
g.age_group = g.age_group.replace({'[18.0, 40.0)': '18-39',
                                   '[40.0, 50.0)': '40-49',
                                   '[50.0, 60.0)': '50-59',
                                   '[60.0, 70.0)': '60-69',
                                   '[70.0, 80.0)': '70-79',
                                   '[80.0, 90.0)': '80-89'})
g.age_group  = g.age_group .replace({'[90.0, ' + str(age_max) + ')': '90+'})
print(g.age_group.unique())


#endregion


# ---- 6. Get test set and save ----
#region

# Get first request per patient
g = g.sort_values(by=['patient_id', 'fit_request_date_corrected'])
gsub = g.groupby('patient_id').first().reset_index()
assert gsub.patient_id.nunique() == gsub.shape[0]
gsub.nonret1_days70.mean()
gsub.nonret2_days70.mean()

# Dbl check how many patients have multiple requests at the same date
mask = g[['patient_id', 'fit_request_date_corrected']].duplicated(keep=False)
mask.sum()
s = g.loc[mask]

# Dbl check return rate over time
gsubsub = gsub.dropna(subset=['nonret2_days70'])
gsubsub = gsubsub.loc[gsubsub.request_year < 2024]
s = gsubsub.groupby('request_year').nonret2_days70.mean()
n = gsubsub.groupby('request_year').size().rename('n')
sn = pd.concat(objs=[s, n], axis=1).reset_index()
sn
#plt.scatter(sn.n, sn.nonret1_days70)
#plt.show()

# Reset index
g = g.reset_index(drop=True)
gsub = gsub.reset_index(drop=True)

# Assign most recent 10% of patients with 70-day fu as the test set 
# Note 2025-10-09: the code currently assigns 10% of all patients (not just those with 70-day fu): gsub.shape[0] not gsubsub.shape[0]
gsub = gsub.sort_values(by=['fit_request_date_corrected'])
assert (gsub.loc[gsub.fit_request_date_fu < 70].index == gsub.loc[gsub.nonret1_days70.isna()].index).all()
gsubsub = gsub.loc[gsub.fit_request_date_fu >= 70]
gsubsub['test_set'] = 0
ntest = int(np.round(0.1 * gsub.shape[0]))
gsubsub.loc[gsubsub.iloc[-ntest:].index, 'test_set'] = 1
assert gsubsub.test_set.sum() == ntest
gsubsub.groupby('test_set').fit_request_date_corrected.describe()
gsubsub.groupby('test_set').nonret1_days70.sum()
gsubsub.groupby('test_set').size()
gsubsub.test_set.mean()  # 10.3%
gsubsub.shape[0] * 0.1
gsub.shape[0] * 0.1
gsubsub.test_set.sum()
(gsubsub.test_set == 0).mean()  # 89.7%

## Note - not considering deaths before fu atm, but if doing so, only 116 rows, won't affect the first decimal place
mask_death = (gsubsub.censored == 1) & (~gsubsub.days_to_death.isna()) & (gsubsub.days_to_death <= 70)
mask_death.sum()
t = gsubsub.loc[~mask_death]
t.shape[0]
t.test_set.mean()
(t.test_set==0).mean()  # 89.7%


# (And also treat most patients with less than 70 day fu as test set, even though
# they won't appear in the logistic analysis)
gsub['test_set'] = 0
gsub.loc[gsub.patient_id.isin(gsubsub.loc[gsubsub.test_set==1].patient_id), 'test_set'] = 1
assert gsub.test_set.sum() == gsubsub.test_set.sum()
gsub.loc[gsub.fit_request_date_fu < 70, 'test_set'] = 1
gsub.test_set.sum()
gsubsub.test_set.sum()
gsub.test_set.mean()
gsub.loc[gsub.fit_request_date_fu >= 70].test_set.mean()

# Propagate the test set indicator the full requests table as well
g['test_set'] = 0
g.loc[g.patient_id.isin(gsub.loc[gsub.test_set == 1].patient_id), 'test_set'] = 1
g.test_set.mean()

# Save
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
out_path.mkdir(exist_ok=True)
g.to_csv(out_path / 'fit_nonret.csv', index=False)
gsub.to_csv(out_path / 'first_fit_nonret.csv', index=False)


# Check number of requests per patient
s = g.groupby('patient_id').icen.nunique().value_counts(normalize=True) * 100
s

g2 = g.loc[g.patient_id.duplicated(keep=False)].sort_values(by=['patient_id', 'fit_request_date_corrected'])
g2['dshift'] = g2.fit_request_date_corrected.shift(-1)
g2 = g2[['patient_id', 'icen', 'fit_request_date_corrected', 'dshift']]
g2['delta'] = (g2.dshift - g2.fit_request_date_corrected).dt.days
s2 = g2.groupby('patient_id').first()
s2.delta.describe(percentiles=[0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99])


# Time between first and second request
np.mean(s2.delta <= 28)
np.mean(s2.delta <= 70)
np.mean(s2.delta >= 180)
np.mean(s2.delta >= 365)

# Explore approx number of GP practices 
out_path = Path(r'Z:\fit_nonreturn_paper_20250417\data')
df = pd.read_csv(out_path / 'first_fit_nonret.csv')
assert not df.gp.isna().any()
n = df.gp.value_counts()
n.describe(percentiles=[0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
df.gp.nunique()

# Check num deaths within 70 days after first request
s = gsub.loc[(~gsub.days_to_death.isna()) & (gsub.days_to_death <= 70) & (gsub.censored == 1)]
n_death = s.patient_id.nunique()  #122
n_death / gsub.shape[0] * 100

36 / (gsub.shape[0] + 36)  # percent of patients with request >365


gsub['fit_request_year'] = gsub.fit_request_date_corrected.dt.year
gsub.groupby('fit_request_year').size()

#endregion
