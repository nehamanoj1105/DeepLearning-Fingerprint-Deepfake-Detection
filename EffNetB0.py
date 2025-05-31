import os
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
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
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])  # Set shape here!
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])

    # Optional augmentation (only on training dataset)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.1)

    # Preprocess for EfficientNet (normalization)
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

for images, labels in train_dataset.take(1):
    print(images.shape)  # Should be (batch_size, IMG_SIZE, IMG_SIZE, 3)
    print(labels.shape)  # Should be (batch_size,)
    print(labels.dtype)  # Should be int32 or int64


# === Initial Training ===
print("Intially Trained Model... \n")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
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
)

# === Evaluation ===
val_preds = model.predict(val_dataset)
val_preds_labels = (val_preds > 0.5).astype(int).flatten()

true_labels = []
for _, lbls in val_dataset.unbatch():
    true_labels.append(int(lbls.numpy()))


print(confusion_matrix(true_labels, val_preds_labels))
print(classification_report(true_labels, val_preds_labels))

loss, acc = model.evaluate(val_dataset)
print(f"✅ Final Validation Accuracy: {acc:.2f}")

# === Save Final Model ===
model.save(MODEL_PATH)
print(f"Model saved to: {MODEL_PATH}")


