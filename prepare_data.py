# Script version of examine.ipynb with some updates
import numpy as np
import os

## Set main data directory. This is the folder that contains all preprocessed .npy files not yet split in train/val/test
data_dir = './data'

def pad_to_pow2(arr):
    """
    Input: arr of shape (N, C, H, W)
    Pads H, W up to the next power of 2 with constant value 0.
    """
    _, _, h, w = arr.shape
    h_pad = 2**int(np.ceil(np.log2(h))) - h
    w_pad = 2**int(np.ceil(np.log2(w))) - w
    pad_top, pad_bottom = h_pad // 2, h_pad - h_pad // 2
    pad_left, pad_right = w_pad // 2, w_pad - w_pad // 2
    return np.pad(
        arr,
        ((0, 0), (0, 0),
         (pad_top, pad_bottom),
         (pad_left, pad_right)),
        mode='constant'
    )

## Get only .npy files
data_lst = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

# make dir 'train', 'val', 'test' inside data_dir if they don't exist
train_dir = os.path.join(data_dir, 'train')
os.makedirs(train_dir, exist_ok=True)

val_dir = os.path.join(data_dir, 'val')
os.makedirs(val_dir, exist_ok=True)

test_dir = os.path.join(data_dir, 'test')
os.makedirs(test_dir, exist_ok=True)

train_mris = []
val_mris = []
test_mris = [] 

## Make random split with 60% train, 20% val, 20% test and store as single .npy files
train_percent = 0.6
val_percent   = 0.2

n_files = len(data_lst)
indices = np.random.permutation(n_files)

n_train = int(train_percent * n_files)
n_val   = int(val_percent * n_files)
n_test  = n_files - n_train - n_val  # whatever remains

train_idx = indices[:n_train]
val_idx   = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

for i in train_idx:
    train_mris.append(np.load(os.path.join(data_dir, data_lst[i]), allow_pickle=True))

for i in val_idx:
    val_mris.append(np.load(os.path.join(data_dir, data_lst[i]), allow_pickle=True))

for i in test_idx:
    test_mris.append(np.load(os.path.join(data_dir, data_lst[i]), allow_pickle=True))

# Concatenate along slice axis (last axis)
train_data = np.concatenate(train_mris, axis=-1).astype(np.float32)
val_data   = np.concatenate(val_mris,   axis=-1).astype(np.float32)
test_data  = np.concatenate(test_mris,  axis=-1).astype(np.float32)

# Move slice axis to front: (n_slices, H, W)
train_data = np.moveaxis(train_data, -1, 0)
val_data   = np.moveaxis(val_data,   -1, 0)
test_data  = np.moveaxis(test_data,  -1, 0)

# Add channel dimension: (n_slices, 1, H, W)
train_data = np.expand_dims(train_data, 1)
val_data   = np.expand_dims(val_data,   1)
test_data  = np.expand_dims(test_data,  1)

## Changed order of operations to: pad first, then normalize
train_data = pad_to_pow2(train_data)
val_data   = pad_to_pow2(val_data)
test_data  = pad_to_pow2(test_data)

## Use only padded training data to compute normalization stats to prevent black box around image
train_min = train_data.min()
train_max = train_data.max()

epsilon = 1e-8  # for numeric stability

train_data = (train_data - train_min) / (train_max - train_min + epsilon)
val_data   = (val_data   - train_min) / (train_max - train_min + epsilon)
test_data  = (test_data  - train_min) / (train_max - train_min + epsilon)

# Save
np.save(os.path.join(train_dir, 'train_data.npy'), train_data)
np.save(os.path.join(val_dir,   'val_data.npy'),   val_data)
np.save(os.path.join(test_dir,  'test_data.npy'),  test_data)