"""
Splits for Camelyon.
"""
import copy
import sys
import os
sys.path.extend([
    "./",
])
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import h5py
from PIL import Image

from src.utils import util_path

column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})


def extract_info(path, subset_name):
    """
    Extracts the patient_id, label, and center from the patch_path.
    Adjusts the extraction logic based on the subset name for specific handling of 'pannuke'.
    """
    parts = path.split('/')
    patient_id, label, center = None, None, None

    if subset_name == 'pannuke':
        # Specific extraction for pannuke
        patient_id = parts[2]
        center = parts[1]
        label = parts[3]
    else:
        # General extraction for other subsets
        for part in parts:
            if 'patient' in part:
                patient_id = part
            elif 'centre' in part or 'fold' in part:
                center = part
            elif any(x in part for x in ['normal', 'tumor']):
                label = part

    return patient_id, label, center


if __name__ == "__main__":

    # Parameters.
    dataset_name = 'camelyon'
    subset_name_list = [
        'cam16',
        'all500',
        'extra17',
        'test_data2',
        'pannuke',
    ]
    # Ignore tumor_extra17
    concepts_list = [
        'nuclei_correlation',
        'nuclei_contrast',
        'ncount',
        'narea',
        'full_correlation',
        'full_contrast',
    ]

    # Directories.
    interim_dir = os.path.join('data', dataset_name, 'interim') # here find the concepts and data splits
    concept_dir = os.path.join(interim_dir, 'cmeasures')
    reports_dir =  os.path.join('reports', dataset_name)

    # Load the CSV file
    df = pd.read_csv(os.path.join(concept_dir, f'concepts.csv'))

    # Apply the extraction function to each row
    df[['patient_id', 'label', 'center']] = df.apply(
        lambda row: pd.Series(extract_info(row['patch_path'], row['subset_name'])),
        axis=1
    )

    # Save to the disk.
    df.to_csv(os.path.join(concept_dir, 'concepts_patients.csv'), index=False)

    # Check for NaN values
    nan_counts = df.isna().sum()
    print("Number of NaN values per column:")
    print(nan_counts)

    # Check for values equal to 'None'
    none_counts = df.apply(lambda x: x == 'None').sum()
    print("Number of 'None' values per column:")
    print(none_counts)

    # Group by the subset_name, center, and label.
    slice_label_counts = df.groupby(['subset_name', 'center', 'label'])['patch_path'].count()
    patients_label_counts = df.groupby(['subset_name', 'center', 'label'])['patient_id'].nunique()

    slice_label_counts.to_csv(os.path.join(concept_dir, 'slice_label_counts.csv'))
    patients_label_counts.to_csv(os.path.join(concept_dir, 'patients_label_counts.csv'))

    # Now I want to label a ['train', 'val', 'test', 'ex_test'] splits.
    # label all the patches of 'test_data2' as 'ex_test'
    df['split'] = df['subset_name'].apply(lambda x: 'ex_test' if x == 'test_data2' else None)

    # split ['all500', 'cam16', 'extra17', 'pannuke'] into 'train', 'val', 'test' stratifying to the centers and labels
    df_strat = df[df['split'].isnull()].copy()
    df_strat['strat'] = df_strat['center'] + '_' + df_strat['label']
    train_val, test = train_test_split(df_strat, test_size=0.1, stratify=df_strat['strat'],  random_state=42, shuffle=True)
    train, val = train_test_split(train_val, test_size=0.1, stratify=train_val['strat'], random_state=42, shuffle=True)

    df.loc[train.index, 'split'] = 'train'
    df.loc[val.index, 'split'] = 'val'
    df.loc[test.index, 'split'] = 'test'

    # Check the distribution of centers and label of the splits
    split_counts = df.groupby('split')['center'].value_counts()
    split_counts.to_csv(os.path.join(concept_dir, 'split_counts.csv'))
    label_counts = df.groupby('split')['label'].value_counts()
    label_counts.to_csv(os.path.join(concept_dir, 'label_counts.csv'))

    # Create a df that show the distribution of the splits
    split_label_counts = df.groupby(['split', 'center', 'label'])['patch_path'].count()
    split_label_counts.to_csv(os.path.join(concept_dir, 'split_label_counts.csv'))

    # Save to the disk.
    df.to_csv(os.path.join(concept_dir, 'concepts_patients_splits.csv'), index=False)

    print("May the force be with you!")