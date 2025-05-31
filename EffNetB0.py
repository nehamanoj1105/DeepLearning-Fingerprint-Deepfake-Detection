import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# === Configuration ===
DATASET_DIR = r'C:\Aakash PDFs\Cyber S4\ML\Project\DataSet'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_AT = 200  # Layers to unfreeze later
MODEL_PATH = r'C:\Aakash PDFs\Cyber S4\ML\Project\EffNet.h5'

# === Data Preprocessing and Augmentation ===
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

# === Build Model ===
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze base model initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

# === Callbacks ===
callbacks = [
    EarlyStopping(patience=4, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=2, min_lr=1e-6),
    ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_loss')
]

# === Initial Training ===
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# === Fine-Tuning ===
base_model.trainable = True

# Freeze earlier layers, unfreeze top EfficientNet layers
for layer in base_model.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(optimizer=Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=7,
    callbacks=callbacks
)

# === Evaluation ===
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

val_preds = model.predict(val_generator)
val_preds_labels = (val_preds > 0.5).astype(int).flatten()
true_labels = val_generator.classes

print(confusion_matrix(true_labels, val_preds_labels))
print(classification_report(true_labels, val_preds_labels))


loss, acc = model.evaluate(val_generator)
print(f"✅ Final Validation Accuracy: {acc:.2f}")


# === Save Final Model ===
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")
