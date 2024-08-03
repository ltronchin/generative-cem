"""
This file contains the utility functions for handling the datasets.
"""
import os
import sys

sys.path.extend([
    "./",
])
import seaborn as sns

from torch.utils.data import Dataset
import torch
import numpy as np
import pandas as pd
import h5py

column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

class dSpritesDataset(Dataset):
    def __init__(self, data_dir, fname, use_concepts=False, concept_to_keep = []):
        """
        Args:
            data_dir (string): Directory with all the images.
            fname (string): Name of the file with the data.
            use_concepts (bool): Whether to use the concepts as features.
            concept_to_keep (list): Numeric position of the concept to retain.
        """
        self.data_dir = data_dir
        self.img_size = 64

        data = np.load(os.path.join(self.data_dir, fname), allow_pickle=True)
        self.images = data['X']
        self.labels = data['Y']
        self.use_concepts = use_concepts
        self.concept_to_keep = concept_to_keep
        if self.use_concepts:
            try:
                self.concepts = data['G']
            except KeyError:
                self.use_concepts = False

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        # Show the image.
        # plt.imshow(image, cmap='gray')
        # plt.axis('off')
        # plt.show()

        im = np.expand_dims(image, 0) # Add channel dimension.
        #im = np.repeat(im, 3, axis=0) # Repeat the channel 3 times.
        im = torch.from_numpy(im).float() # Convert to float.

        label = self.labels[idx]
        label = np.array(label).astype(np.int64)

        if self.use_concepts:
            c = np.array(self.concepts[idx], dtype=np.float32)
            # print(c.shape)

            if self.concept_to_keep:
                c_to_keep = c[self.concept_to_keep]
                c_excl = np.delete(c, self.concept_to_keep, axis=0)
                # print(c_excl.shape)
            else: 
                c_to_keep = c
                c_excl = np.empty(0, dtype=np.float32)
            return {'image': im, 'label': label, 'c_to_keep': c_to_keep, 'c_excl': c_excl}
        else:
            return {'image': im, 'label': label}
        
#%%        
    
        
class CamelyonDataset(Dataset):
    """Custom Dataset for loading Camelyon image patches with labels and concepts."""
    def __init__(self, subset, csv_file, root_dir, norm_img=True, to_tensor=True, transform=None, norm_conc=True, concept_to_keep=[]):
        """
        Args:
            subset (list): List of subset names to load.
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the image patches.
            norm_img (bool, optional): Whether to normalize the image patches.
            to_tensor (bool, optional): Whether to convert the image patches to PyTorch tensors.
            transform (callable, optional): Optional transform to be applied on a sample.
            norm_conc (bool, optional): Whether to normalize the concepts.
            concept_to_keep (list, optional): List of positions of concepts to retain.
        """
        self.subset = subset
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.normalize_image = norm_img
        self.to_tensor = to_tensor
        self.transform = transform
        self.normalize_concepts = norm_conc
        self.concept_to_keep = concept_to_keep

        # Convert labels to numeric values if necessary
        self.label_mapping = {label: idx for idx, label in enumerate(self.df['label'].unique())}
        # self.df['label'] = self.df['label'].map(self.label_mapping).astype(np.float32)

        self.patches = self.load_patches()

        # Preprocess concepts
        self.concepts = self.df.iloc[:, 3:9].values.astype('float32')
        if self.normalize_concepts:
            self.concept_mean = self.concepts.mean(axis=0)
            self.concept_std = self.concepts.std(axis=0)
            self.concepts = (self.concepts - self.concept_mean) / self.concept_std
        else:
            self.concept_mean = None
            self.concept_std = None

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
        if self.normalize_image:
            img = img.astype(np.float32) / 255.0
        if self.to_tensor:
            img = torch.from_numpy(img).permute(2, 0, 1)
        if self.transform:
            img = self.transform(img)

        label = self.label_mapping[self.df.iloc[idx, -3]]
        concepts = self.concepts[idx]

        if self.concept_to_keep:
            c_to_keep = np.squeeze(concepts[self.concept_to_keep], axis=0)
            c_excl = np.delete(concepts, self.concept_to_keep, axis=0)
        else:
            c_to_keep = concepts
            c_excl = np.empty(0, dtype=np.float32)

        return {'image': img, 'label': label, 'c_to_keep': c_to_keep, 'c_excl': c_excl}

# %%
