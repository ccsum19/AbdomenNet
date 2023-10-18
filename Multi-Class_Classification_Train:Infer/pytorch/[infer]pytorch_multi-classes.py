#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().system('pip install /kaggle/input/pydicom-and-torchmetrics/pydicom-2.4.3-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/pydicom-and-torchmetrics/torchmetrics-1.2.0-py3-none-any.whl')


# ## Imports and Setup

import cv2
import gc
from glob import glob
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import pydicom
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassAccuracy
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm.notebook import tqdm


# ## Configuration

class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    BATCH_SIZE = 20
    EPOCHS = 5
    TARGET_COLS = [
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]

config = Config()


# ## Reproducibility

random_seed = config.SEED
np.random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

BASE_PATH = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
IMAGE_DIR = '/tmp/dataset/rsna-atd'
INPUT_MODEL_PATH = '/kaggle/input/pytorchcv-starter-notebook-train/model_5.pth'
MODEL_PATH = '/kaggle/working/model_5.pth'
STRIDE = 10

class Config:
    SEED = 42
    IMAGE_SIZE = [256, 256]
    RESIZE_DIM = 256
    BATCH_SIZE = 20
    EPOCHS = 5
    TARGET_COLS = [
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
    ]

config = Config()


# ## Initialize the Trained Model

class build_model(nn.Module):
    def __init__(
        self, num_classes_bowel,
        num_classes_extra, num_classes_liver,
        num_classes_kidney, num_classes_spleen
    ):
        super(build_model, self).__init__()

        # define backbone
        self.backbone = models.efficientnet_b5(pretrained=False)
        self.backbone._fc = nn.Identity()

        # delete 'necks' for each head
        self.neck_bowel = nn.Linear(1000, 32) # self.backbone._swish._num_features
        self.neck_extra = nn.Linear(1000, 32)
        self.neck_liver = nn.Linear(1000, 32)
        self.neck_kidney = nn.Linear(1000, 32)
        self.neck_spleen = nn.Linear(1000, 32)

        # define heads
        self.head_bowel = nn.Linear(32, num_classes_bowel)
        self.head_extra = nn.Linear(32, num_classes_extra)
        self.head_liver = nn.Linear(32, num_classes_liver)
        self.head_kidney = nn.Linear(32, num_classes_kidney)
        self.head_spleen = nn.Linear(32, num_classes_spleen)

    def forward(self, x):
        # forward pass through the backbone
        # print(x.shape)
        x = self.backbone(x)
        # print(x.shape)

        # forward pass through 'necks' and heads
        x_bowel = self.head_bowel(self.neck_bowel(x))
        x_extra = self.head_extra(self.neck_extra(x))
        x_liver = self.head_liver(self.neck_liver(x))
        x_kidney = self.head_kidney(self.neck_kidney(x))
        x_spleen = self.head_spleen(self.neck_spleen(x))

        return x_bowel, x_extra, x_liver, x_kidney, x_spleen


get_ipython().system(' cp {INPUT_MODEL_PATH} ./')

model_variable = torch.load(MODEL_PATH, map_location = 'cpu')
model = build_model(
    num_classes_bowel=1,
    num_classes_extra=1,
    num_classes_liver=3,
    num_classes_kidney=3,
    num_classes_spleen=3,
)
model.load_state_dict(model_variable, strict = False)
model.to(device)


meta_df = pd.read_csv(f"{BASE_PATH}/test_series_meta.csv")

# Checking if patients are repeated by finding the number of unique patient IDs
num_rows = meta_df.shape[0]
unique_patients = meta_df["patient_id"].nunique()

print(f"{num_rows=}")
print(f"{unique_patients=}")


meta_df["dicom_folder"] = BASE_PATH + "/" + "test_images"                                    + "/" + meta_df.patient_id.astype(str)                                    + "/" + meta_df.series_id.astype(str)

test_folders = meta_df.dicom_folder.tolist()
test_paths = []
for folder in test_folders:
    test_paths += sorted(glob(os.path.join(folder, "*dcm")))[::STRIDE]


test_df = pd.DataFrame(test_paths, columns=["dicom_path"])
test_df["patient_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-3]).astype(int)
test_df["series_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-2]).astype(int)
test_df["instance_number"] = test_df.dicom_path.map(lambda x: x.split("/")[-1].replace(".dcm","")).astype(int)

test_df["image_path"] = f"{IMAGE_DIR}/test_images"                    + "/" + test_df.patient_id.astype(str)                    + "/" + test_df.series_id.astype(str)                    + "/" + test_df.instance_number.astype(str) +".png"

test_df.head(2)



# Checking if patients are repeated by finding the number of unique patient IDs
num_rows = test_df.shape[0]
unique_patients = test_df["patient_id"].nunique()

print(f"{num_rows=}")
print(f"{unique_patients=}")


get_ipython().system('rm -r {IMAGE_DIR}')
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


# ## Data Pipeline /w dataloader

class RandomCutout(transforms.RandomApply):
    def __init__(self, p, cutout_height_factor=0.2, cutout_width_factor=0.2):
        cutout = transforms.RandomErasing(
            p=1.0, scale=(cutout_height_factor, cutout_width_factor), ratio=(1,1)
        )
        super(RandomCutout, self).__init__([cutout], p=p)

class CustomDataset(Dataset):
    def __init__(self, image_paths, image_size, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        image = transforms.Resize(self.image_size)(image)
        image = transforms.ToTensor()(image)
        image = image / 255.0

        # for additional transformation
        if self.transform:
            image = self.transform(image)

        return image


cutout_transform = RandomCutout(p=0.5, cutout_height_factor=0.2, cutout_width_factor=0.2)
transform = transforms.Compose([cutout_transform])

def build_dataset(image_paths, batch_size, image_size, transform=None):
    dataset = CustomDataset(image_paths, image_size=image_size, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    return dataloader


# ## Inference

def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype="float32")
#     print(proc_pred.shape)

    # bowel, extravasation
    proc_pred[:, 0] = pred[:, 0]
    proc_pred[:, 1] = 1 - proc_pred[:, 0]
    proc_pred[:, 2] = pred[:, 1]
    proc_pred[:, 3] = 1 - proc_pred[:, 2]

    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5]
    proc_pred[:, 7:10] = pred[:, 5:8]
    proc_pred[:, 10:13] = pred[:, 8:11]

    return proc_pred


# Getting unique patient IDs from test dataset
patient_ids = test_df['patient_id'].unique()

# Initializing array to store predictions
patient_preds = np.zeros(
    shape=(len(patient_ids), 2*2 + 3*3),
    dtype="float32"
)

for pidx, patient_id in tqdm(enumerate(patient_ids), total=len(patient_ids), desc="Patients "):
    print(f'Patient ID: {patient_id}')

    # Query the dataframe for a particualr patient
    patient_df = test_df[test_df["patient_id"] == patient_id]

    # Getting image paths for a particular patient
    patient_paths = patient_df.image_path.tolist()

    # Bulding dataset for prediction
    dtest = build_dataset(patient_paths, config.BATCH_SIZE, config.IMAGE_SIZE)

    # Predicting with the model
    model.eval()
    with torch.no_grad():
        preds = []
        for image in dtest:
            image = image.to(device)
            pred_bowel, pred_extra, pred_liver, pred_kidney, pred_spleen = model(image)

            tensors = [F.softmax(tensor, dim=1) for tensor in [pred_bowel, pred_extra, pred_liver, pred_kidney, pred_spleen]]
            pred = torch.cat(tensors, dim=-1).float()

            preds.append(pred)


        preds = torch.cat(preds, dim=0)
        preds = preds[:len(patient_paths), :]
        preds = torch.mean(preds.reshape(1, len(patient_paths), -1), dim=0)
        preds = torch.max(preds, dim=0, keepdim=True)[0]

        patient_preds[pidx, :] = post_proc(pred.cpu().numpy())

        del patient_df, patient_paths, dtest, pred_bowel, pred_extra, pred_liver, pred_kidney, pred_spleen, pred
        gc.collect()

columns = [
    'bowel_healthy',
    'bowel_injury',
    'extravasation_healthy',
    'extravasation_injury',
    'kidney_healthy',
    'kidney_low',
    'kidney_high',
    'liver_healthy',
    'liver_low',
    'liver_high',
    'spleen_healthy',
    'spleen_low',
    'spleen_high'
]

pred_df = pd.DataFrame({"patient_id":patient_ids,})
pred_df[columns] = patient_preds.astype("float32")

sub_df = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")
sub_df = sub_df[["patient_id"]]
sub_df = sub_df.merge(pred_df, on="patient_id", how="left")

sub_df.to_csv("submission.csv",index=False, float_format='%.5f')
sub_df.head(3)
