"""
Hands on histopathology
"""
import copy
import sys
import os
sys.path.extend([
    "./",
])
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import time
column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import Compose, Lambda

class CamelyonDataset(Dataset):
    """Custom Dataset for loading Camelyion image patches with labels and concepts."""
    def __init__(self, subset, csv_file, root_dir, normalize=True, to_tensor=True, transform=None):
        """
        Args:
            subset (list): List of subset names to load.
            csv_file (string): Path to the csv file with annotations.
            split (string): The split to use, e.g., 'train', 'val', 'test', 'ex_test'.
            root_dir (string): Directory with all the image patches.
            normalize (bool, optional): Whether to normalize the image patches.
            to_tensor (bool, optional): Whether to convert the image patches to PyTorch tensors.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.subset = subset
        self.df = pd.read_csv(csv_file)
        # Filter the data frame based on the split.
        # self.df = self.df[self.df['split'] == split]
        # self.df = self.df.reset_index(drop=True)
        self.root_dir = root_dir
        self.normalize = normalize
        self.to_tensor = to_tensor
        self.transform = transform
        # Convert labels to numeric values if necessary
        self.label_mapping = {label: idx for idx, label in enumerate(self.df['label'].unique())}

        self.patches = self.load_patches()

    def load_patches(self):
        data = {}
        for subset_name in self.subset:
            patches_dir = os.path.join(self.root_dir, subset_name)
            print(f"Loading patches from {patches_dir}")
            if subset_name == 'pannuke':
                patches_filename = os.path.join(patches_dir, 'patches_fix.hdf5')
            else:
                patches_filename = os.path.join(patches_dir, 'patches.hdf5')
            patches = h5py.File(patches_filename, 'r')
            data[subset_name] = patches

        return data

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        subset_name = self.df.iloc[idx, 0]
        patches_list = self.df.iloc[idx, 1]
        patch_id = self.df.iloc[idx, 2]

        # Load the patch.
        img = self.patches[subset_name][patches_list][patch_id]
        if self.normalize:
            img = img.astype(np.float32) / 255.0
        if self.to_tensor:
            img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)

        label = self.label_mapping[self.df.iloc[idx, -3]]
        concepts = self.df.iloc[idx, 3:9].values.astype('float')

        return {'image': img, 'label': label, 'concepts': concepts}

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
    concepts_list = [
        'nuclei_correlation',
        'nuclei_contrast',
        'ncount',
        'narea',
        'full_correlation',
        'full_contrast',
    ]

    os.chdir('/home/riccardo/Github/generative-cem')
    # Directories.
    interim_dir = os.path.join('data', dataset_name, 'interim')  # here find the concepts and data splits
    concept_dir = os.path.join(interim_dir, 'cmeasures')
    raw_dir = '/home/lorenzo/generative-cem/data/camelyon/raw' # here find the patches
    reports_dir = os.path.join('reports', dataset_name)

    # Initialize the dataset (example)
    dataset_train = CamelyonDataset(subset=subset_name_list, csv_file=os.path.join(concept_dir, f'df_train.csv'), root_dir=raw_dir, normalize=True, to_tensor=True, transform=None)
    #dataset_train = CamelyonDatasetEfficient(subset=subset_name_list, csv_file=os.path.join(concept_dir, f'concepts_patients_splits.csv'), split='train', root_dir=raw_dir, transform=None)

    # Initialize the data loader
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, shuffle=True)

    # Iterate over the data using tqdm
    for i, data in enumerate(tqdm(dataloader_train)):
        # Get the inputs
        x, y, c = data['image'], data['label'], data['concepts']
        if x.min() < 0 or x.max() > 1:
            raise ValueError("The image is not normalized.")

    print("May the force be with you!")
