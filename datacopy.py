"""Copies FIT data in parquet format to the shared drive,
from three airlocked tables"""

import numpy as np
import pandas as pd
import os
import shutil
from pathlib import Path


data_path = Path("C:/Users/n4XY/Desktop/airlock_import")
os.listdir(data_path)

# Here we copy most data items in the data product,
# except gpreqs and demographics, as newer versions of these are copied separately below
src1 = data_path / 'fit_airlock_part1_20240619'
items1 = [i for i in os.listdir(src1) if i not in ['gpreqs', 'demographics']]
print(items1)

target_path = Path("Z:/fit_dataproduct_20240626")
target_path.mkdir(exist_ok=True)

for i in items1:
    shutil.copytree(src1 / i, target_path / i)

# Here we copy the most updated version of gpreqs
dest_path = target_path / 'gpreqs'  
dest_path.mkdir(exist_ok=True)
fname = 'gpreqs_1.parquet'
shutil.copyfile(data_path / fname, dest_path / fname)

# Here we copy the most updated version of demographics
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
