import copy
import sys
import os
sys.path.extend([
    "./",
])
import numpy as np
import pickle

from src.utils import util_path

if __name__ == "__main__":

	# Directories.
	dataset_name = 'dsprites'
	raw_dir = os.path.join('data', dataset_name, 'raw')  # here find the patches
	interim_dir = os.path.join('data', dataset_name, 'interim')
	reports_dir = os.path.join('reports', dataset_name)
	fname = 'dsprites_leakage_partitions__pos_x_size__partial'

	# Load dataset
	data = np.load(os.path.join(raw_dir, 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'), allow_pickle=True,  encoding='latin1')
	images = data['imgs']
	latents = data['latents_values']
	y = np.ones(len(latents)) * 2 # Initialize all the target variables to 0.

	# Define the target variable according to the posX and size.
	mask_pos_x = (latents[:, 4] > 0.75)
	mask_size = (latents[:, 2] > 0.6)
	mask = (mask_pos_x & mask_size)
	n_pos = np.sum(mask)
	y[mask] = 1
	y[~mask] = 0

	# Do not consider the latent used in the generation process (2, 4) 1 3 5
	latents = latents[:, [1, 3, 5]]

	# Show some images sampled randomly within class 1 and 0
	import matplotlib.pyplot as plt
	import random
	fig, ax = plt.subplots(1, 5, figsize=(15, 3))
	for i in range(5):
		idx = random.choice(np.where(y == 1)[0])
		ax[i].imshow(images[idx], cmap='gray')
		ax[i].axis('off')
		ax[i].set_title(f'Label: {y[idx]}')
	# Tight layout.
	plt.tight_layout()
	# Save to the disk.
	plt.savefig(os.path.join(reports_dir, 'dsprites_leakage_partitions__pos_x_size.png'))
	plt.show()

	# Perform the train-test split with 80% of the data for training and 10% Val, 10% for testing (use sklearn and indexes)
	from sklearn.model_selection import train_test_split
	train_val_idx, test_idx = train_test_split(range(len(images)), test_size=0.2, random_state=42)
	train_idx, val_idx = train_test_split(train_val_idx, test_size=0.2, random_state=42)

	train_X, val_X, test_X = images[train_idx], images[val_idx], images[test_idx]
	train_g, val_g, test_g = latents[train_idx], latents[val_idx], latents[test_idx]
	train_y, val_y, test_y = y[train_idx], y[val_idx], y[test_idx]

	# Standardize the latent variables.
	train_g_mean = np.mean(train_g, axis=0)
	train_g_std = np.std(train_g, axis=0)
	train_g = (train_g - train_g_mean) / train_g_std
	val_g = (val_g - train_g_mean) / train_g_std
	test_g = (test_g - train_g_mean) / train_g_std

	# Print the shapes of the train and test sets.
	print('Train X shape:', train_X.shape)
	print('Val X shape:', val_X.shape)
	print('Test X shape:', test_X.shape)

	util_path.create_dir(os.path.join(interim_dir, fname))
	np.savez(os.path.join(interim_dir, fname, 'dsprites_leakage_train'),
		X = train_X,
		G = train_g,
		Y = train_y)

	np.savez(os.path.join(interim_dir, fname, 'dsprites_leakage_val'),
		X = val_X,
		G = val_g,
		Y = val_y)

	np.savez(os.path.join(interim_dir, fname,  'dsprites_leakage_test'),
		X = test_X,
		G = test_g,
		Y = test_y)

	print('May the force be with you!')

