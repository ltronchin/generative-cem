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