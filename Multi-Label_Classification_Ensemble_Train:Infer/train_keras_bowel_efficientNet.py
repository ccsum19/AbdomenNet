#!/usr/bin/env python
# coding: utf-8

get_ipython().system('pip install kaggle')
get_ipython().system(' pip install -q git+https://github.com/keras-team/keras-cv')

# mount the drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_cv
import keras_core as keras
from keras_core import layers

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# Config

class Config:
    SEED = 42
    IMAGE_SIZE = [512, 512]
    BATCH_SIZE = 16
    EPOCHS = 15
    TARGET_COLS = [
        "bowel"
    ]
    AUTOTUNE = tf.data.AUTOTUNE

config = Config()

keras.utils.set_random_seed(seed=config.SEED)


# Dataset
BASE_PATH = "/content/drive/MyDrive/rsna-atd-512x512-png-v2-dataset"

# train
dataframe = pd.read_csv(f"{BASE_PATH}/train.csv")
dataframe["image_path"] = f"{BASE_PATH}/train_images"                    + "/" + dataframe.patient_id.astype(str)                    + "/" + dataframe.series_id.astype(str)                    + "/" + dataframe.instance_number.astype(str) +".png"
dataframe = dataframe.drop_duplicates()

dataframe.head(2)

def bowel_assign_value(row):
    if row['bowel_healthy'] == 1:
        return 0
    else:
        return 1

dataframe['bowel'] = dataframe.apply(bowel_assign_value, axis=1)
dataframe.head(2)


negative = dataframe[dataframe['bowel'] == 0]
positive = dataframe[dataframe['bowel'] == 1]
num_samples = min(len(negative), len(positive))
negative_samples = negative.sample(n=num_samples, random_state=42)
positive_samples = positive.sample(n=num_samples, random_state=42)
bowel_dataframe = pd.concat([negative_samples, positive_samples], axis=0)

# Function to handle the split for each group
def split_group(group, test_size=0.2):
    if len(group) == 1:
        return (group, pd.DataFrame()) if np.random.rand() < test_size else (pd.DataFrame(), group)
    else:
        return train_test_split(group, stratify=group["bowel"], test_size=test_size, random_state=42)

# Initialize the train and validation datasets
bowel_train_data = pd.DataFrame()
bowel_val_data = pd.DataFrame()

# Iterate through the groups and split them, handling single-sample groups
for _, group in bowel_dataframe.groupby(config.TARGET_COLS):
    bowel_train_group, bowel_val_group = split_group(group)
    bowel_train_data = pd.concat([bowel_train_data, bowel_train_group], ignore_index=True)
    bowel_val_data = pd.concat([bowel_val_data, bowel_val_group], ignore_index=True)

def decode_image_and_label(image_path, label):
    file_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.float32)

    return (image, label)

# 레이어 외부에서 RandomFlip 레이어를 생성
random_flip_layer = tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal")
random_rotation_layer = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)

class CustomAugmenter(tf.keras.layers.Layer):
    def __init__(self, cutout_params, **kwargs):
        super(CustomAugmenter, self).__init__(**kwargs)
        self.cutout_layer = keras_cv.layers.Augmenter([keras_cv.layers.RandomCutout(**cutout_params)])

    def call(self, inputs, training=None):
        if training:
            inputs = random_flip_layer(inputs)
            inputs = random_rotation_layer(inputs)
            inputs = self.cutout_layer(inputs)
        return inputs

def apply_augmentation(images, labels):
    # 이미지 증강 파이프라인을 정의
    augmenter = CustomAugmenter(cutout_params={"height_factor": 0.2, "width_factor": 0.2})

    # 이미지 증강을 적용
    augmented_images = augmenter(images, training=True)

    return (augmented_images, labels)

def build_dataset(image_paths, labels):
    ds = (
        tf.data.Dataset.from_tensor_slices((image_paths, labels))
        .map(decode_image_and_label, num_parallel_calls=config.AUTOTUNE)
        .shuffle(config.BATCH_SIZE * 10)
        .batch(config.BATCH_SIZE)
        .map(apply_augmentation, num_parallel_calls=config.AUTOTUNE)  # 이미지 증강 적용
        .prefetch(config.AUTOTUNE)
    )
    return ds

paths = bowel_train_data.image_path.tolist()
labels = bowel_train_data[config.TARGET_COLS].values

ds = build_dataset(image_paths=paths, labels=labels)
images, labels = next(iter(ds))
images.shape, [label.shape for label in labels]


keras_cv.visualization.plot_image_gallery(
    images=images,
    value_range=(0, 1),
    rows=2,
    cols=2,
)


# Model
# efficient Net

def build_binary_classification_model(warmup_steps, decay_steps, head_name):
    # Define Input
    inputs = keras.Input(shape=config.IMAGE_SIZE + [3,], batch_size=config.BATCH_SIZE)

    # Define Backbone
    backbone = keras_cv.models.EfficientNetV2Backbone.from_preset("efficientnetv2_b3")
    backbone.include_rescaling = False
    x = backbone(inputs)

    # GAP to get the activation maps
    gap = keras.layers.GlobalAveragePooling2D()
    x = gap(x)

    # Define 'necks' for the binary classification head
    x_head = keras.layers.Dense(32, activation='silu')(x)

    # Define binary classification head
    output = keras.layers.Dense(1, name=head_name, activation='sigmoid')(x_head)

    # Create model
    print(f"[INFO] Building the {head_name} model...")
    model = keras.Model(inputs=inputs, outputs=output)

    # Cosine Decay
    cosine_decay = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-4,
        decay_steps=decay_steps,
        alpha=0.0,
        warmup_target=1e-3,
        warmup_steps=warmup_steps,
    )

    # Compile the model
    optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)
    loss = keras.losses.BinaryCrossentropy()
    metrics = ["accuracy"]

    print(f"[INFO] Compiling the {head_name} model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


# Train
# get image_paths and labels
print("[INFO] Building the dataset...")
train_paths = bowel_train_data.image_path.values; train_labels = bowel_train_data[config.TARGET_COLS].values.astype(np.float32)
valid_paths = bowel_val_data.image_path.values; valid_labels = bowel_val_data[config.TARGET_COLS].values.astype(np.float32)

# train and valid dataset
train_ds = build_dataset(image_paths=train_paths, labels=train_labels)
val_ds = build_dataset(image_paths=valid_paths, labels=valid_labels)

total_train_steps = train_ds.cardinality().numpy() * config.BATCH_SIZE * config.EPOCHS
warmup_steps = int(total_train_steps * 0.10)
decay_steps = total_train_steps - warmup_steps

print(f"{total_train_steps=}")
print(f"{warmup_steps=}")
print(f"{decay_steps=}")


# In[ ]:


# List of model names
model_names = ["bowel"]

# Create a 1x2 grid for the subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Flatten axes to iterate through them
axes = axes.flatten()

for i, name in enumerate(model_names):
    # Build the model
    if name in ["bowel"]:
        model = build_binary_classification_model(warmup_steps, decay_steps, name)

    # Train the model
    history = model.fit(train_ds, epochs=config.EPOCHS, validation_data=val_ds)


    # Plot training accuracy
    axes[0].plot(history.history['accuracy'], label='Training ' + name)
    # Plot validation accuracy
    axes[1].plot(history.history['val_accuracy'], label='Validation ' + name)

    axes[0].set_title("Training Accuracy")
    axes[1].set_title("Validation Accuracy")
    axes[0].set_xlabel('Epoch')
    axes[1].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[0].legend()
    axes[1].legend()

    plt.tight_layout()
    plt.show()

from tensorflow.keras.models import load_model

# Save the model
model.save(BASE_PATH + "/bowel_rsna.keras")
