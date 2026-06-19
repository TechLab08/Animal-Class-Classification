import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import itertools
import warnings
warnings.filterwarnings("ignore")

# ------------------ CONSTANTS ------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATA_DIR = 'data/'      # Your dataset folder
LEARNING_RATE = 0.001
EPOCHS = 50
SEED = 123

# ------------------ LOAD DATA ------------------
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

CLASS_NAMES = np.array(train_ds.class_names)
NUM_CLASSES = len(CLASS_NAMES)
print(f"Detected {NUM_CLASSES} classes: {CLASS_NAMES}")

# Improve I/O performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------ DATA AUGMENTATION ------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.5),
    tf.keras.layers.RandomZoom(0.3),
])

# ------------------ BUILD MODEL ------------------
base_model = tf.keras.applications.EfficientNetV2B0(
    input_shape=IMAGE_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze base

inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']   # accuracy during training
)
model.summary()

# ------------------ TRAIN MODEL ------------------
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("Best_Cattle_Breed.h5",
                                       save_best_only=True),
    tf.keras.callbacks.EarlyStopping(patience=10,
                                     restore_best_weights=True)
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

print("✅ Training finished. Model saved as 'Best_Cattle_Breed.h5'")

# ========================================================
# --------- POST-TRAINING EVALUATION METRICS ------------
# ========================================================

print("\n🔍 Calculating detailed evaluation metrics...")

# Extract images and labels from validation dataset
y_true = []
y_pred = []
for images, labels in val_ds:
    preds = model.predict(images, verbose=0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# --- 1. Classification report: Precision / Recall / F1 per class ---
report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    digits=4
)
print("\n📊 Classification Report:")
print(report)

# --- 2. Confusion Matrix ---
cm = confusion_matrix(y_true, y_pred)

def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

plot_confusion_matrix(cm, CLASS_NAMES)

# --- 3. Overall metrics summary ---
acc = np.mean(y_true == y_pred)
print(f"✅ Final Validation Accuracy: {acc:.4f}")


