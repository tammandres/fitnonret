"""Copies FIT data in parquet format to the shared drive,
from three airlocked tables"""

import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path


data_path = Path("C:/Users/n4XY/Desktop/airlock_import")
os.listdir(data_path)

# This is most tables from the data product: ignoring gpreqs and demographics as these were old versions
src1 = data_path / 'fit_airlock_part1_20240619'
items1 = [i for i in os.listdir(src1) if i not in ['gpreqs', 'demographics']]
print(items1)

# This is gpreqs, but still an older version - can be ignored.
#src2 = data_path / 'fit_airlock_part1_fixtimezone_20240625'
#items2 = os.listdir(src2)
#print(items2)

#src3 = data_path / 'fit_airlock_part2_20240620'
#items3 = os.listdir(src3)
#print(items3)

# Copy 
target_path = Path("Z:/fit_dataproduct_20240626")
target_path.mkdir(exist_ok=True)

for i in items1:
    shutil.copytree(src1 / i, target_path / i)

#for i in items2: #  This line only copies gpreqs, an old version, probably not necessary
#    shutil.copytree(src2 / i, target_path / i)  

#for i in items3:
#    shutil.copytree(src3 / i, target_path / i)  

dest_path = target_path / 'gpreqs'  # Here we copy the most updated version of gpreqs
dest_path.mkdir(exist_ok=True)
fname = 'gpreqs_1.parquet'
shutil.copyfile(data_path / fname, dest_path / fname)


## Note: there was an issue in how the demographics table was created previously
## e.g. see slack thread w J in July 17 2024
## This was fixed in the new iteration of the table
dest_path = target_path / 'demographics'
dest_path.mkdir(exist_ok=True)
fname = 'demographics_1.parquet'
shutil.copyfile(data_path / fname, dest_path / fname)


# Dbl check demographics
demo = pd.read_parquet(target_path / 'demographics')
fit = pd.read_parquet(target_path / 'fit_values')
assert fit.patient_id.isin(demo.patient_id).all()


# Note how the new gpreqs table has many more rows
#df0 = pd.read_parquet(src2 / 'gpreqs')
#df1 = pd.read_parquet(data_path / 'gpreqs_1.parquet')
#print(df0.shape, df1.shape)