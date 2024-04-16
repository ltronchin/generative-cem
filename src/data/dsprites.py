"""
Hands on dSprites
Latent factor values

Color: white
Shape: square, ellipse, heart
Scale: 6 values linearly spaced in [0.5, 1]
Orientation: 40 values in [0, 2 pi]
Position X: 32 values in [0, 1]
Position Y: 32 values in [0, 1]
"""

import copy
import sys
import os
sys.path.extend([
    "./",
])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

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

    # Directories.
    dataset_name = 'dsprites'
    raw_dir = os.path.join('data', dataset_name, 'raw')  # here find the patches
    reports_dir = os.path.join('reports', dataset_name)

    # Load dataset
    dataset_zip = np.load(os.path.join(raw_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True, encoding='latin1') # encoding='latin1' -> for python 2.7 (tackle the problem of opening the file in current Python version)

    print('Keys in the dataset:', dataset_zip.keys())
    imgs = dataset_zip['imgs']
    latents_values = dataset_zip['latents_values']
    latents_classes = dataset_zip['latents_classes']
    metadata = dataset_zip['metadata'][()]

    # Save to interim the latent values and latents classes as dataframe.
    latents_values_df = pd.DataFrame(latents_values, columns=metadata['latents_names'])
    latents_classes_df = pd.DataFrame(latents_classes, columns=metadata['latents_names'])

    latents_values_df.to_csv(os.path.join(raw_dir, 'latents_values.csv'), index=False)
    latents_classes_df.to_csv(os.path.join(raw_dir, 'latents_classes.csv'), index=False)

    # Print the first image with the latents values and classes.
    plt.imshow(imgs[0], cmap='gray')
    plt.axis('off')
    plt.show()
    print('Latents classes: \n', latents_classes[0])
    print('Latents values: \n', latents_values[0])

    print('Metadata: \n', metadata)

    print("May the force be with you!")