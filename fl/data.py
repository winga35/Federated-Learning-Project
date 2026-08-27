import hashlib
import os

import numpy as np
from sklearn.model_selection import train_test_split
from torchvision.datasets import CIFAR10


def load_unified(root=None):
    #We set root as the location of our CIFAR10. 
    root = root or os.environ["FL_DATA_ROOT"]

    #We take each dataset
    train = CIFAR10(root=root, train=True, download=True)
    test = CIFAR10(root=root, train=False, download=True)

    #We combine them as specified
    x = np.concatenate([train.data, test.data], axis=0)
    y = np.concatenate([np.asarray(train.targets), np.asarray(test.targets)])

    print("Datasets unified")

    return x, y, train.classes


def generate_stratified_splits_60_10_30(y, seed=42):
    #We create a array [0, 1, 2, ..., 59999]
    idx = np.arange(len(y))

    #First we split 70% training indexes and 30% Test indexes
    trainval_idx, test_idx = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)

    #Then 60% Training indexes and 10% Validation Indexes
    train_idx, val_idx = train_test_split(trainval_idx, test_size=1 / 7, stratify=y[trainval_idx], random_state=seed)

    print(f"Split: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test")

    return train_idx, val_idx, test_idx



def make_splits():
    x, y, classes = load_unified()
    train_idx, val_idx, test_idx = generate_stratified_splits_60_10_30(y)

    train_counts = np.bincount(y[train_idx])
    val_counts = np.bincount(y[val_idx])
    test_counts = np.bincount(y[test_idx])

    for i, name in enumerate(classes):
        print(f"{name:<12}{train_counts[i]:>7}{val_counts[i]:>6}{test_counts[i]:>7}")

    split_hash = hashlib.sha256(train_idx.tobytes()).hexdigest()[:16]
    np.savez_compressed("splits/split_seed42.npz", train_idx=train_idx, val_idx=val_idx,
                        test_idx=test_idx, seed=42, split_hash=split_hash)
    print("Saved splits/split_seed42.npz", split_hash)
