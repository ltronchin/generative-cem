"""
Hands on histopathology
"""
import copy
import sys
import os
sys.path.extend([
    "./",
])
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

import h5py
from PIL import Image
from src.utils import util_path

# For aspect ratio 4:3.
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})
# sns.set_context("paper")
# sns.set_theme(style="ticks")
# For Latex.
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

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
    #concept_dir = os.path.join(interim_dir, 'normalized_cmeasures')
    raw_dir = os.path.join('data', dataset_name, 'raw') # here find the patches
    reports_dir =  os.path.join('reports', dataset_name)

    data = {}
    for subset_name in subset_name_list:
        patches_dir = os.path.join(raw_dir, subset_name)
        print(f"Loading patches from {patches_dir}")
        if subset_name == 'pannuke':
            patches_filename = os.path.join(patches_dir, 'patches_fix.hdf5')
        else:
            patches_filename = os.path.join(patches_dir, 'patches.hdf5')
        patches = h5py.File(patches_filename, 'r')
        data[subset_name] = patches

    concepts = pd.DataFrame()
    concepts_dict = {}
    for concept_name in concepts_list:

        concept_df = pd.read_csv(os.path.join(concept_dir, f'{concept_name}.csv'), header=None)
        concept_df = concept_df.rename(columns={0: 'subset_name', 1: 'patch_path', 2: 'patch_idx', 3: concept_name})
        concepts_dict[concept_name] = copy.deepcopy(concept_df)

        if concepts.empty:
            concepts = concept_df
        else:
            concepts = pd.merge(concepts, concept_df, on=['subset_name', 'patch_path', 'patch_idx'], how='outer')

    # Remove all the rows that contains "tumor_extra17" in the subset_name column.
    concepts = concepts[~concepts['subset_name'].str.contains('tumor_extra17')]

    nan_counts = concepts.isna().sum()
    print("Number of NaN values per column:")
    print(nan_counts)
    rows_with_nans = concepts[concepts.isna().any(axis=1)]

    inf_counts = concepts.isin([float('inf')]).sum()
    print("\nNumber of inf values per column:")
    print(inf_counts)

    concepts = concepts.dropna()
    concepts.reset_index(drop=True, inplace=True)

    # Save to disk.
    concepts.to_csv(os.path.join(concept_dir, 'concepts_raw.csv'), index=False)

    # Perform some analysis.

    for concept in concepts_list:
        # Box plot
        fig = plt.figure()
        sns.boxplot(x=concepts[concept])
        plt.title(f'Box Plot of {concept}')
        plt.xlabel(concept)
        plt.tight_layout()
        plt.show()
        # Save to the disk as pdf keeping quality.
        fig.savefig(os.path.join(reports_dir, f'{concept}_boxplot.pdf'),bbox_inches='tight',format='pdf', dpi=300)

    for concept in concepts_list:
        # Histogram
        fig = plt.figure()
        sns_obj = sns.histplot(concepts[concept])
        plt.title(f'Histogram of {concept}')
        plt.xlabel(concept)
        plt.ylabel('Frequency')
        plt.ylim(0,5000)
        plt.tight_layout()
        plt.show()
        # Save to the disk.
        fig.savefig(os.path.join(reports_dir, f'{concept}_histogram.pdf'), bbox_inches='tight', format='pdf', dpi=300)

    print("May the force be with you!")
