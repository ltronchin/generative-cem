import numpy as np
import pickle


data = np.load('dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

images = data['imgs']
latents = data['latents_values']
y = latents[:,1]

shapes_mask = (latents[:,1] == 0) | (latents[:,1] == 2)

mask_x = (latents[:,4] < 0.52) & (latents[:,4] > 0.48) 
mask_y = (latents[:,5] < 0.52) & (latents[:,5] > 0.48) 

mask_1_test = (latents[:,4] < 0.25) & (latents[:,5] < 0.25) 
mask_2_test = (latents[:,4] > 0.75) & (latents[:,5] > 0.75) 

mask_training = mask_x & mask_y # & shapes_mask
mask_test = (mask_1_test | mask_2_test) # & shapes_mask

mask_training = np.array(mask_training)
mask_test = np.array(mask_test)

print('Number of examples:', mask_training.sum(), mask_test.sum())

train_X = images[mask_training]
train_g = latents[mask_training][:,[1,2,3]]
train_y = y[mask_training]

test_X = images[mask_test]
test_g = latents[mask_test][:,[1,2,3,4,5]]

y_test = np.zeros(len(latents))
mask_2_test = np.array(mask_2_test)
y_test[mask_2_test] = np.ones(mask_2_test.sum())
test_y = y_test[mask_test]

np.savez('dsprites/dsprites_leakage_train',
	X = train_X,
	G = train_g,
	Y = train_y)

perm = np.random.permutation(len(test_X))
np.savez('dsprites/dsprites_leakage_test',
	X = test_X[perm][:5000],
	G = test_g[perm][:5000],
	Y = test_y[perm][:5000])	




