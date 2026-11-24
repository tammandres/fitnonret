"""Rename key figures"""
from pathlib import Path
import os
import shutil


# Paths
airlock_path = Path(r'/Users/andres/Library/CloudStorage/OneDrive-Nexus365/PostDoc2024/projects/fitnonret/airlock/20261118_fitnonret-fixed')
out_path     = Path(r'/Users/andres/Library/CloudStorage/OneDrive-Nexus365/PostDoc2024/projects/fitnonret/BJC_revision_20261028/figs')
out_path.mkdir(exist_ok=True)


for c in os.listdir(airlock_path):
    if c.endswith('png'):
        print(c)

shutil.copyfile(airlock_path / 'gam_link_nonret2_days70.png',
                out_path / 'Figure2_gam_link_nonret2_days70.png')

shutil.copyfile(airlock_path / 'logistic_pred_nonret2_days70.png',
                out_path / 'Figure3_logistic_pred_nonret2_days70.png')

shutil.copyfile(airlock_path / 'return_time_panel_data-full_panel-a-70-day-fu_type2.png',
                out_path / 'Figure4_return_time_panel_data-full_panel-a-70-day-fu_type2.png')

shutil.copyfile(airlock_path / 'gam_pred_nonret2_days70.png',
                out_path / 'FigureS1_gam_pred_nonret2_days70.png')

shutil.copyfile(airlock_path / 'check_test_set_stability_nonret2_days70_reps10.png',
                out_path / 'FigureS2_check_test_set_stability_nonret2_days70_reps10.png')

