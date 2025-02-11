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
        ################################################## RICCARDO ##################################################
    
    # Control, with respect to path: data\camelyon\interim\cmeasures\info\patients_label_counts.csv
    for i in df['subset_name'].unique():
        df_queried = df[df['subset_name'] == i]
        counts = len(df_queried['patient_id'].unique())
        print(i,': ', counts)
    
    # N patients:
    # all500 :  34
    # cam16 :  69
    # extra17 :  78
    # pannuke :  879
    # test_data2 :  1
    #%%
    # Correcting pannuke patch idx
    import re
    df = df[df['subset_name'] == 'pannuke'] 
    counter = 0
    previous_number = 0    
    less_image_sub_list = []

    for idx, row in df.iterrows():
        match = re.search(r"fold\d+/(\d+)/(normal|tumor)", row['patch_path'])
        if match:
            current_number = int(match.group(1))
            if current_number == previous_number:
                df.at[idx, 'patch_idx'] = counter
                print(counter, idx, current_number)
                counter += 1
            # if different number (e.g. aptient), check if there are 5 images for him
            else:
                # check number of image per sub
                if counter != 5:
                    less_image_sub_list.append(match)
                    print(75*'#',3*'\n', f'Less than 5 element in index {idx}',3*'\n', 75*'#')
                # if we reached 5 images, update the number
                else:
                    print('New_sub')
                # update sub and counter
                previous_number = current_number
                counter = 0
                # write patch idx in the new sub
                df.at[idx, 'patch_idx'] = counter
                print(counter, idx, current_number)
                counter += 1

# founded: 137692, 138921, 139365, 140603,  141745, 141984, 142907, 144675

    #%% 
    # Write new column to take track of the nodes
    df['nodule'] = int(0)
    pattern = r'node(\d+)'
    import re
    for idx, row in df[0:10].iterrows():
        node = int(re.findall(pattern, df.at[idx, 'patch_path'])[0])
        df.at[idx, 'nodule'] = node
    
    df['nodule'] = df['nodule'].astype('uint8') # less space


def train_val_test_split_df(dataframe, percentages=None, mode=None, manual_sel=None, seed=None):
    '''
    :param dataframe:     Dataframe to be split by
    :param percentages:   Ordered percentage to divide the dataframe in train, val, test
    :param mode:          Which way to split dataframe, any column value can be used ('sub', 'rep', 'day')
    :param manual_sel:    If not none, select which "mode" is going to be the train val and test.
                          Need to be a list of 3 list, for train val and test respectively
    :seed                 Applyu a seed for rerpoducibility

    :return:              train, validation and test dataframe with the same structure of the original one
    '''
    if seed:
        np.random.seed(seed)

    if percentages is None:
        percentages = [0.7, 0.2, 0.1]
    elif sum(percentages) != 1:
        print('Percentages to divide dataframe are not equal to one! you are leaving out data!')
    if mode is None:  # random sample
        idx_list = np.unique(dataframe.index)
        n_samples = len(idx_list)
        # shuffle indexes
        np.random.shuffle(idx_list)
        # train val and test sizes
        tr_size, val_size = round(n_samples * percentages[0]), round(n_samples * percentages[1])
        # train, val, and test indexes
        tr_idx, val_idx, test_idx = idx_list[: tr_size], idx_list[tr_size: tr_size + val_size], idx_list[
                                                                                                tr_size + val_size:]
        # train val and test dataframes
        tr_df, val_df, test_df = dataframe.iloc[tr_idx], dataframe.iloc[val_idx], dataframe.iloc[test_idx]

    else:
        if mode not in dataframe.columns:
            raise ValueError('"Criterion not present in dataframes columns!')
        if manual_sel is not None:
            if len(manual_sel) != 3:
                raise ValueError('Provide a list with 3 ordered list for manual splitting\n'
                                 'Example: manual_sel = [[2,4,6],[1,3],[4]]')
            tr_df = dataframe[dataframe[mode].isin(manual_sel[0])]
            val_df = dataframe[dataframe[mode].isin(manual_sel[1])]
            test_df = dataframe[dataframe[mode].isin(manual_sel[2])]

        else:
            mode_list = np.unique(dataframe[mode])
            np.random.shuffle(mode_list)
            # train val and test sizes
            tr_size, val_size = round(len(mode_list) * percentages[0]), round(len(mode_list) * percentages[1])
            # train, val, and test with modality names stored in
            tr_list, val_list, test_list = mode_list[:tr_size], mode_list[tr_size:tr_size + val_size], mode_list[
                                                                                                       tr_size + val_size:]
            # train val and test dataframes
            tr_df, val_df = dataframe[dataframe[mode].isin(tr_list)], dataframe[dataframe[mode].isin(val_list)]
            test_df = dataframe[dataframe[mode].isin(test_list)]

    if len(tr_df) + len(val_df) + len(test_df) != len(dataframe):
        IndexError(
            'Something went wrong when splitting dataframe! Some data are not part of either the train, val and test')

    return tr_df, val_df, test_df
    
    # Create columns for split value
    df['split'] = df['subset_name'].apply(lambda x: 'ex_test' if x == 'test_data2' else None)

    # Take out the 'test_data2" subset
    idx = df[df['subset_name'] == 'test_data2'].index[0]
    df_to_split = df.iloc[:idx]
    # df_ex_test = df.iloc[idx:]

    # First create a test set
    df_tr_cross_val, _, df_test = train_val_test_split_df(dataframe=df_to_split, percentages=[0.9, 0.0, 0.1], mode='patient_id', seed=29)
    # Now create train and val (isnert k-.fold here if necessary)
    df_tr, df_val, _ = train_val_test_split_df(dataframe=df_tr_cross_val, percentages=[0.9, 0.1, 0.0], mode='patient_id', seed=29)

    df.loc[df_tr.index, 'split'] = 'train'
    df.loc[df_val.index, 'split'] = 'val'
    df.loc[df_test.index, 'split'] = 'test'

    del df_tr_cross_val, df_to_split
    
################################################## RICCARDO ###################################################

    
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
