import os
import random

REAL_DIR = 'DataSet/Original'
AI_DIR = 'DataSet/Artificial'
VAL_SPLIT = 0.2
SEED = 42

# Collect all file paths and labels
data = []
for img in os.listdir(REAL_DIR):
    data.append((os.path.join(REAL_DIR, img), 0))
for img in os.listdir(AI_DIR):
    data.append((os.path.join(AI_DIR, img), 1))

# Shuffle
random.seed(SEED)
random.shuffle(data)

# Split
val_size = int(len(data) * VAL_SPLIT)
val_data = data[:val_size]
train_data = data[val_size:]

# Write split files
with open('train_split.txt', 'w') as f:
    for path, label in train_data:
        f.write(f"{path}\t{label}\n")

with open('val_split.txt', 'w') as f:
    for path, label in val_data:
        f.write(f"{path}\t{label}\n")

print("New Train/ Val Split Generated.")
