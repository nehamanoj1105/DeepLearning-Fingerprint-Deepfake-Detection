# import torch
# from torch.utils.data import DataLoader, random_split
# from torchvision import transforms, datasets, models
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Define your transforms (customize as needed)
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Imagenet means
#                          std=[0.229, 0.224, 0.225])   # Imagenet stds
# ])

# # Load dataset from folder (assuming structure: root/class_x/xxx.png)
# dataset = datasets.ImageFolder('DataSet', transform=transform)

# # Split dataset into train and val (80%-20%)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_data, val_data = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))

# train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
# val_loader = DataLoader(val_data, batch_size=8, shuffle=False)

# # Load pretrained EfficientNet-B0
# model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

# # Replace classifier head for 2 classes
# num_features = model.classifier[1].in_features  # usually 1280 for EffNetB0
# model.classifier[1] = nn.Linear(num_features, 2)

# model = model.to(device)

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Print model summary (optional)
# print(model)

# import torch
# import torch.nn.functional as F
# from tqdm import tqdm  # for progress bar

# num_epochs = 10

# for epoch in range(num_epochs):
#     # --- Training ---
#     model.train()
#     train_loss = 0
#     correct_train = 0
#     total_train = 0

#     for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train"):
#         images, labels = images.to(device), labels.to(device)

#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         train_loss += loss.item() * images.size(0)
#         _, predicted = torch.max(outputs, 1)
#         correct_train += (predicted == labels).sum().item()
#         total_train += labels.size(0)

#     train_loss /= total_train
#     train_acc = correct_train / total_train

#     # --- Validation ---
#     model.eval()
#     val_loss = 0
#     correct_val = 0
#     total_val = 0

#     with torch.no_grad():
#         for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val"):
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             val_loss += loss.item() * images.size(0)
#             _, predicted = torch.max(outputs, 1)
#             correct_val += (predicted == labels).sum().item()
#             total_val += labels.size(0)

#     val_loss /= total_val
#     val_acc = correct_val / total_val

#     print(f"Epoch {epoch+1}/{num_epochs}: "
#           f"Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f} | "
#           f"Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}")

# effnet_probs = np.concatenate(effnet_all_probs, axis=0)    # EfficientNet prediction probs
# effnet_labels = np.concatenate(effnet_all_labels, axis=0)  # EfficientNet true labels

# # Save EfficientNet model
# model.save(r'C:\Aakash PDFs\Cyber S4\ML\Project\EfficientNetB0.h5')
# print("EfficientNet model saved...")

# # Save EfficientNet predictions and labels
# np.save('effnet_preds.npy', effnet_probs)
# np.save('effnet_labels.npy', effnet_labels)
# print("Saved EfficientNet prediction probabilities and labels.")

import os
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Configuration ===
REAL_DIR = r'C:\Aakash PDFs\Cyber S4\ML\Project\DataSet\Original'  # Update this path
AI_DIR = r'C:\Aakash PDFs\Cyber S4\ML\Project\DataSet\Artificial'      # Update this path
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_AT = 200  # Layers to unfreeze later
MODEL_PATH = r'C:\Aakash PDFs\Cyber S4\ML\Project\EffNet.keras'

# === Prepare file paths and labels ===
def get_image_paths_and_labels(real_dir, ai_dir):
    image_paths = []
    labels = []

    for img_name in os.listdir(real_dir):
        image_paths.append(os.path.join(real_dir, img_name))
        labels.append(0)  # Real fingerprint

    for img_name in os.listdir(ai_dir):
        image_paths.append(os.path.join(ai_dir, img_name))
        labels.append(1)  # Artificial fingerprint

    return image_paths, labels

image_paths, labels = get_image_paths_and_labels(REAL_DIR, AI_DIR)

# Shuffle dataset
dataset_size = len(image_paths)
indices = tf.random.shuffle(tf.range(dataset_size))
image_paths = tf.gather(image_paths, indices)
labels = tf.gather(labels, indices)

# Split train and val (80-20 split)
val_size = int(0.2 * dataset_size)
train_image_paths = image_paths[val_size:]
train_labels = labels[val_size:]
val_image_paths = image_paths[:val_size]
val_labels = labels[:val_size]

# === Dataset loading and preprocessing ===
def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)  # adjust if png
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # Data augmentation for training
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    
    # Preprocess for EfficientNet
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

def load_and_preprocess_image_noaug(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image, label

# Create tf.data datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_image_paths, train_labels))
train_dataset = train_dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_image_paths, val_labels))
val_dataset = val_dataset.map(load_and_preprocess_image_noaug, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# === Build Model ===
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

# === Initial Training ===
print("Intially Trained Model... \n")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

# === Fine-tuning ===
base_model.trainable = True
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])

print("Model after Fine Tuning... \n")
history_fine = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=7,
    callbacks=callbacks,
    workers=4,
    use_multiprocessing=True
)

# === Evaluation ===
val_preds = model.predict(val_dataset)
val_preds_labels = (val_preds > 0.5).astype(int).flatten()

true_labels = []
for _, lbls in val_dataset.unbatch():
    true_labels.append(int(lbls.numpy()))

from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(true_labels, val_preds_labels))
print(classification_report(true_labels, val_preds_labels))

loss, acc = model.evaluate(val_dataset)
print(f"✅ Final Validation Accuracy: {acc:.2f}")

# === Save Final Model ===
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")


