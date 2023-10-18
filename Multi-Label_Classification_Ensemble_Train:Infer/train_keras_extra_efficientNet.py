#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install kaggle')


# In[2]:


get_ipython().system(' pip install -q git+https://github.com/keras-team/keras-cv')


# In[3]:


# mount the drive
from google.colab import drive
drive.mount('/content/drive')


# In[4]:


import os
# You can use `tensorflow`, `pytorch`, `jax` here
# KerasCore makes the notebook backend agnostic :)
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras_cv
import keras_core as keras
from keras_core import layers

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


# keras starter 기반으로 작성된 노트북

# Config

# In[12]:


class Config:
    SEED = 42
    IMAGE_SIZE = [512, 512]
    BATCH_SIZE = 16
    EPOCHS = 15
    TARGET_COLS = [
        "extravasation_injury"
    ]
    AUTOTUNE = tf.data.AUTOTUNE

config = Config()


# In[13]:


keras.utils.set_random_seed(seed=config.SEED)


# Dataset

# In[14]:


BASE_PATH = "/content/drive/MyDrive/rsna-atd-512x512-png-v2-dataset"


# In[15]:


# train
dataframe = pd.read_csv(f"{BASE_PATH}/train.csv")
dataframe["image_path"] = f"{BASE_PATH}/train_images"                    + "/" + dataframe.patient_id.astype(str)                    + "/" + dataframe.series_id.astype(str)                    + "/" + dataframe.instance_number.astype(str) +".png"
dataframe = dataframe.drop_duplicates()

dataframe.head(2)


# In[16]:


def extra_assign_value(row):
    if row['extravasation_healthy'] == 1:
        return 0
    else:
        return 1

dataframe['extra'] = dataframe.apply(extra_assign_value, axis=1)
dataframe.head(2)


# In[17]:


negative = dataframe[dataframe['extra'] == 0]
positive = dataframe[dataframe['extra'] == 1]
num_samples = min(len(negative), len(positive))
negative_samples = negative.sample(n=num_samples, random_state=42)
positive_samples = positive.sample(n=num_samples, random_state=42)
extra_dataframe = pd.concat([negative_samples, positive_samples], axis=0)


# In[18]:


# Function to handle the split for each group
def split_group(group, test_size=0.2):
    if len(group) == 1:
        return (group, pd.DataFrame()) if np.random.rand() < test_size else (pd.DataFrame(), group)
    else:
        return train_test_split(group, stratify=group["extra"], test_size=test_size, random_state=42)

# Initialize the train and validation datasets
extra_train_data = pd.DataFrame()
extra_val_data = pd.DataFrame()

# Iterate through the groups and split them, handling single-sample groups
for _, group in extra_dataframe.groupby(config.TARGET_COLS):
    extra_train_group, extra_val_group = split_group(group)
    extra_train_data = pd.concat([extra_train_data, extra_train_group], ignore_index=True)
    extra_val_data = pd.concat([extra_val_data, extra_val_group], ignore_index=True)


# In[19]:


def decode_image_and_label(image_path, label):
    file_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0

    label = tf.cast(label, tf.float32)

    return (image, label)


# In[20]:


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


# In[21]:



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


# In[22]:


paths = extra_train_data.image_path.tolist()
labels = extra_train_data[config.TARGET_COLS].values

ds = build_dataset(image_paths=paths, labels=labels)
images, labels = next(iter(ds))
images.shape, [label.shape for label in labels]


# In[23]:


keras_cv.visualization.plot_image_gallery(
    images=images,
    value_range=(0, 1),
    rows=2,
    cols=2,
)


# In[24]:


import tensorflow as tf
from sklearn.metrics import confusion_matrix

# Custom metric to calculate sensitivity
def sensitivity(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 1), tf.equal(tf.round(y_pred), 1)), dtype=tf.float32))
    actual_positives = tf.reduce_sum(tf.cast(tf.equal(y_true, 1), dtype=tf.float32))
    return true_positives / (actual_positives + tf.keras.backend.epsilon())

# Custom metric to calculate specificity
def specificity(y_true, y_pred):
    true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_true, 0), tf.equal(tf.round(y_pred), 0)), dtype=tf.float32))
    actual_negatives = tf.reduce_sum(tf.cast(tf.equal(y_true, 0), dtype=tf.float32))
    return true_negatives / (actual_negatives + tf.keras.backend.epsilon())


# Model

# efficient Net

# In[25]:


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
    metrics = ["accuracy", sensitivity, specificity]

    print(f"[INFO] Compiling the {head_name} model...")
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    return model


# Train

# In[26]:


# get image_paths and labels
print("[INFO] Building the dataset...")
train_paths = extra_train_data.image_path.values; train_labels = extra_train_data[config.TARGET_COLS].values.astype(np.float32)
valid_paths = extra_val_data.image_path.values; valid_labels = extra_val_data[config.TARGET_COLS].values.astype(np.float32)

# train and valid dataset
train_ds = build_dataset(image_paths=train_paths, labels=train_labels)
val_ds = build_dataset(image_paths=valid_paths, labels=valid_labels)

total_train_steps = train_ds.cardinality().numpy() * config.BATCH_SIZE * config.EPOCHS
warmup_steps = int(total_train_steps * 0.10)
decay_steps = total_train_steps - warmup_steps

print(f"{total_train_steps=}")
print(f"{warmup_steps=}")
print(f"{decay_steps=}")


# In[27]:


# List of model names
model_names = ["extra"]

# Create a 1x2 grid for the subplots
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Flatten axes to iterate through them
axes = axes.flatten()

for i, name in enumerate(model_names):
    # Build the model
    if name in ["extra"]:
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


# In[28]:


# Save the model
model.save(BASE_PATH + "/extra_rsna.keras")


# ## INFER

# In[ ]:


class Config_Infer:
    IMAGE_SIZE = [256, 256]
    RESIZE_DIM = 256
    BATCH_SIZE = 64
    AUTOTUNE = tf.data.AUTOTUNE
    TARGET_COLS  = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high"]

config_infer = Config_Infer()


# In[ ]:


test_df = pd.read_csv(f"{BASE_PATH}/test.csv")

test_df["image_path"] = f"{BASE_PATH}/test_images"                    + "/" + test_df.patient_id.astype(str)                    + "/" + test_df.series_id.astype(str)                    + "/" + test_df.instance_number.astype(str) +".png"

test_df.head(2)


# In[ ]:


def decode_image_infer(image_path):
    file_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image

def build_dataset_infer(image_paths):
    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(decode_image_infer, num_parallel_calls=config.AUTOTUNE)
        .shuffle(config.BATCH_SIZE * 10)
        .batch(config.BATCH_SIZE)
        .prefetch(config.AUTOTUNE)
    )
    return ds


# In[ ]:


paths  = test_df.image_path.tolist()

ds = build_dataset_infer(image_paths=paths)
images = next(iter(ds))

images.shape


# In[ ]:


get_ipython().system('pip install tqdm')


# In[ ]:


from tqdm import tqdm
import gc


# In[ ]:


def post_proc(pred, binary_threshold=0.7):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype="float32")

    # bowel, extravasation
    proc_pred[:, 0] = pred[:, 0] #if pred[:, 0] >= binary_threshold else np.mean(pred[:, 0])
    proc_pred[:, 1] = 1 - proc_pred[:, 0]
    proc_pred[:, 2] = pred[:, 1] #if pred[:, 1] >= binary_threshold else np.mean(pred[:, 1])
    proc_pred[:, 3] = 1 - proc_pred[:, 2]

    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5]
    proc_pred[:, 7:10] = pred[:, 5:8]
    proc_pred[:, 10:13] = pred[:, 8:11]

    return proc_pred


# In[ ]:


# Getting unique patient IDs from test dataset
patient_ids = test_df["patient_id"].unique()

# Initializing array to store predictions
patient_preds = np.zeros(
    shape=(len(patient_ids), 2*2 + 3*3),
    dtype="float32"
)

# Iterating over each patient
for pidx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids), desc="Patients "):
    print(f"Patient ID: {patient_id}")

    # Query the dataframe for a particular patient
    patient_df = test_df[test_df["patient_id"] == patient_id]

    # Getting image paths for a patient
    patient_paths = patient_df.image_path.tolist()

    # Building dataset for prediction
    dtest = build_dataset_infer(patient_paths)

    # Predicting with the model
    pred = model.predict(dtest)
    pred = np.concatenate(pred, axis=-1).astype("float32")
    pred = pred[:len(patient_paths), :]
    pred = np.mean(pred.reshape(1, len(patient_paths), 11), axis=0)
    pred = np.max(pred, axis=0, keepdims=True)

    patient_preds[pidx, :] += post_proc(pred)[0]

    print(patient_preds)

    # Deleting variables to free up memory
    del patient_df, patient_paths, dtest, pred; gc.collect()


# In[ ]:




