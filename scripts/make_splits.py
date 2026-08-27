import hashlib

import numpy as np

from fl.data import load_unified, generate_stratified_splits_60_10_30

x, y, classes = load_unified()
train_idx, val_idx, test_idx = generate_stratified_splits_60_10_30(y)

train_counts = np.bincount(y[train_idx])
val_counts = np.bincount(y[val_idx])
test_counts = np.bincount(y[test_idx])

for i, name in enumerate(classes):
    print(f"{name:<12}{train_counts[i]:>7}{val_counts[i]:>6}{test_counts[i]:>7}")

split_hash = hashlib.sha256(train_idx.tobytes()).hexdigest()[:16]
np.savez_compressed("splits/split_seed42.npz", train_idx=train_idx, val_idx=val_idx, test_idx=test_idx, seed=42, split_hash=split_hash)

print("Saved splits/split_seed42.npz", split_hash)
