#!/usr/bin/env python
# coding: utf-8

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras_core as keras
import keras_cv

import gc
import cv2
import pydicom
from joblib import Parallel, delayed

import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
from glob import glob

from tensorflow.keras.models import load_model

BASE_PATH = "/kaggle/input/rsna-2023-abdominal-trauma-detection"
IMAGE_DIR = "/tmp/dataset/rsna-atd"
STRIDE = 10

class Config:
    IMAGE_SIZE = [256, 256]
    RESIZE_DIM = 512
    BATCH_SIZE = 16
    AUTOTUNE = tf.data.AUTOTUNE
    TARGET_COLS  = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high"]

config = Config()

meta_df = pd.read_csv(f"{BASE_PATH}/test_series_meta.csv")

# Checking if patients are repeated by finding the number of unique patient IDs
num_rows = meta_df.shape[0]
unique_patients = meta_df["patient_id"].nunique()

print(f"{num_rows=}")
print(f"{unique_patients=}")

meta_df["dicom_folder"] = BASE_PATH + "/" + "test_images"                                    + "/" + meta_df.patient_id.astype(str)                                    + "/" + meta_df.series_id.astype(str)

test_folders = meta_df.dicom_folder.tolist()
test_paths = []
for folder in tqdm(test_folders):
    test_paths += sorted(glob(os.path.join(folder, "*dcm")))[::STRIDE]

test_df = pd.DataFrame(test_paths, columns=["dicom_path"])
test_df["patient_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-3]).astype(int)
test_df["series_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-2]).astype(int)
test_df["instance_number"] = test_df.dicom_path.map(lambda x: x.split("/")[-1].replace(".dcm","")).astype(int)

test_df["image_path"] = f"{IMAGE_DIR}/test_images"                    + "/" + test_df.patient_id.astype(str)                    + "/" + test_df.series_id.astype(str)                    + "/" + test_df.instance_number.astype(str) +".png"


# Checking if patients are repeated by finding the number of unique patient IDs
num_rows = test_df.shape[0]
unique_patients = test_df["patient_id"].nunique()

print(f"{num_rows=}")
print(f"{unique_patients=}")

#!rm -r {IMAGE_DIR}
os.makedirs(f"{IMAGE_DIR}/train_images", exist_ok=True)
os.makedirs(f"{IMAGE_DIR}/test_images", exist_ok=True)

def standardize_pixel_array(dcm):
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype
        new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)
    return pixel_array

def read_xray(path, fix_monochrome=True):
    dicom = pydicom.dcmread(path)
    data = standardize_pixel_array(dicom)
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-5)
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1.0 - data
    return data

def resize_and_save(file_path):
    img = read_xray(file_path)
    h, w = img.shape[:2]  # orig hw
    img = cv2.resize(img, (config.RESIZE_DIM, config.RESIZE_DIM), cv2.INTER_LINEAR)
    img = (img * 255).astype(np.uint8)

    sub_path = file_path.split("/",4)[-1].split(".dcm")[0] + ".png"
    infos = sub_path.split("/")
    sub_path = file_path.split("/",4)[-1].split(".dcm")[0] + ".png"
    infos = sub_path.split("/")
    pid = infos[-3]
    sid = infos[-2]
    iid = infos[-1]; iid = iid.replace(".png","")
    new_path = os.path.join(IMAGE_DIR, sub_path)
    os.makedirs(new_path.rsplit("/",1)[0], exist_ok=True)
    cv2.imwrite(new_path, img)
    return

get_ipython().run_cell_magic('time', '', '\nfile_paths = test_df.dicom_path.tolist()\n_ = Parallel(n_jobs=2, backend="threading")(\n    delayed(resize_and_save)(file_path) for file_path in tqdm(file_paths, leave=True, position=0)\n)\n\ndel _; gc.collect()')

def decode_image_binary(image_path):
    file_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, config.IMAGE_SIZE, method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image

def decode_image_multiple(image_path):
    file_bytes = tf.io.read_file(image_path)
    image = tf.io.decode_png(file_bytes, channels=3, dtype=tf.uint8)
    image = tf.image.resize(image, [512, 512], method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image


def build_dataset_binary(image_paths):
    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(decode_image_binary, num_parallel_calls=config.AUTOTUNE)
        .shuffle(config.BATCH_SIZE * 10)
        .batch(config.BATCH_SIZE)
        .prefetch(config.AUTOTUNE)
    )
    return ds

def build_dataset_multiple(image_paths):
    ds = (
        tf.data.Dataset.from_tensor_slices(image_paths)
        .map(decode_image_multiple, num_parallel_calls=config.AUTOTUNE)
        .shuffle(config.BATCH_SIZE * 10)
        .batch(config.BATCH_SIZE)
        .prefetch(config.AUTOTUNE)
    )
    return ds


# # Inference

def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype="float32")

    # bowel, extravasation
    proc_pred[:, 0] = np.where((0.4 < pred[:, 0]) & (pred[:, 0] < 0.7), np.float32(0.973956), pred[:, 0])
    proc_pred[:, 1] = np.where((0.4 < pred[:, 0]) & (pred[:, 0] < 0.7), np.float32(0.098461), 1 - proc_pred[:, 0])
    proc_pred[:, 2] = np.where((0.4 < pred[:, 1]) & (pred[:, 1] < 0.7), np.float32(0.931948), pred[:, 1])
    proc_pred[:, 3] = np.where((0.4 < pred[:, 1]) & (pred[:, 1] < 0.7), np.float32(0.104006), 1 - proc_pred[:, 2])

    # kidney, liver, spleen
    proc_pred[:, 4:7] = np.where(np.max(pred[:, 2:5]) > 0.9, pred[:, 2:5], np.array([0.93764, 0.104006, 0.103077]))
    proc_pred[:, 7:10] = np.where(np.max(pred[:, 5:8]) > 0.9, pred[:, 5:8], np.array([0.893683, 0.23424, 0.095384]))
    proc_pred[:, 10:13] = np.where(np.max(pred[:, 8:11]) > 0.9, pred[:, 8:11], np.array([0.883248, 0.179976, 0.140182]))

    return proc_pred


# **BOWEL/EXTRA**

binary_model = load_model("/kaggle/input/exploring-kerascv-for-rsna-trauma/rsna-atd-mas.keras")


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
    dtest = build_dataset_binary(patient_paths)

    # Predicting with the model
    pred = binary_model.predict(dtest)
    pred = np.concatenate(pred, axis=-1).astype("float32")
    pred = pred[:len(patient_paths), :]
    pred = np.mean(pred.reshape(1, len(patient_paths), 11), axis=0)
    pred = np.max(pred, axis=0, keepdims=True)

    patient_preds[pidx, :] += post_proc(pred)[0]


    # Deleting variables to free up memory
    del patient_df, patient_paths, dtest, pred; gc.collect()

# Create Submission
pred_df = pd.DataFrame({"patient_id":patient_ids,})
pred_df[config.TARGET_COLS] = patient_preds.astype("float32")
pred_df

model_names = ["kidney", "liver", "spleen"]
output_dim = 3
patient_ids = test_df["patient_id"].unique()

dataframes = {}

for model_name in model_names:
    model_filename = f"{model_name}_rsna.keras"
    model_path = os.path.join("/kaggle/input/multi-model", model_filename)

    # Load the model without compiling
    model = tf.keras.models.load_model(model_path, compile = False)

    initial_learning_rate = 0.001
    decay_steps = 1000
    alpha = 0.0

    cosine_decay = tf.keras.experimental.CosineDecay(
        initial_learning_rate,
        decay_steps,
        alpha=alpha
)


    # Compile the model with the given settings
    optimizer = keras.optimizers.Adam(learning_rate=cosine_decay)
    loss = {
        model_name: keras.losses.CategoricalCrossentropy()
    }
    metrics = {
        model_name: ["accuracy"]
    }
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    # Initializing array to store predictions
    patient_preds = np.zeros(shape=(len(patient_ids), output_dim), dtype="float32")

    # Iterating over each patient
    for pidx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids), desc=f"Patients for {model_name}"):
        print(f"Patient ID: {patient_id}")

        # Query the dataframe for a particular patient
        patient_df = test_df[test_df["patient_id"] == patient_id]

        # Getting image paths for a patient
        patient_paths = patient_df.image_path.tolist()

        # Building dataset for prediction
        dtest = build_dataset_multiple(patient_paths)

        # Predicting with the current model
        pred = model.predict(dtest)
        pred = pred[:len(patient_paths), :]
        pred = np.mean(pred.reshape(1, len(patient_paths), output_dim), axis=0)
        pred = np.max(pred, axis=0, keepdims=True)

        patient_preds[pidx, :] = pred.squeeze()

        # Deleting variables to free up memory
        del patient_df, patient_paths, dtest, pred; gc.collect()

    # Convert the predictions to a DataFrame and store in a dictionary
    predictions_df = pd.DataFrame(patient_preds, columns=[f"{model_name}_pred_{j}" for j in range(output_dim)])
    predictions_df["patient_id"] = patient_ids
    dataframes[model_name] = predictions_df


def adjust_values(df, organ_col_prefix, values_to_set):
    # 열의 이름을 추출
    healthy_col = f"{organ_col_prefix}_healthy"
    low_col = f"{organ_col_prefix}_low"
    high_col = f"{organ_col_prefix}_high"

    # 해당 열에서 최대값을 계산
    max_value = df[[healthy_col, low_col, high_col]].max(axis=1)

    # 조건에 따라 값을 설정
    mask = max_value > 0.9
    df.loc[mask, [healthy_col, low_col, high_col]] = values_to_set


organs = ["kidney", "liver", "spleen"]
values_dict = {
    "kidney": [0.93764, 0.104006, 0.103077],
    "liver": [0.893683, 0.23424, 0.095384],
    "spleen": [0.883248, 0.179976, 0.140182]
}

# 각 기관에 대해 반복
for organ in organs:
    # 해당 기관의 데이터프레임을 가져와서 병합
    merged_df = pred_df.merge(dataframes[organ], on="patient_id", how="left")

    # 병합된 데이터프레임에서 원하는 칼럼 값들로 대체
    pred_col_healthy = f"{organ}_healthy"
    pred_col_low = f"{organ}_low"
    pred_col_high = f"{organ}_high"

    merged_col_0 = f"{organ}_pred_0"
    merged_col_1 = f"{organ}_pred_1"
    merged_col_2 = f"{organ}_pred_2"

    merged_df[pred_col_healthy] = merged_df[merged_col_0]
    merged_df[pred_col_low] = merged_df[merged_col_1]
    merged_df[pred_col_high] = merged_df[merged_col_2]

    # 필요 없는 칼럼들 삭제
    merged_df.drop(columns=[merged_col_0, merged_col_1, merged_col_2], inplace=True)

    # 값을 조정
    adjust_values(merged_df, organ, values_dict[organ])

    # 결과를 저장 (또는 별도의 처리를 위해 사용)
    pred_df = merged_df


# # Submission

get_ipython().system('rm -rf {MODEL_PATH}')

# Align with sample submission
sub_df = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
sub_df = sub_df[["patient_id"]]
sub_df = sub_df.merge(pred_df, on="patient_id", how="left")

# Store submission
sub_df.to_csv("submission.csv",index=False)
sub_df
