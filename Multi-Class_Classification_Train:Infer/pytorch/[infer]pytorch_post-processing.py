#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

get_ipython().system('pip install /kaggle/input/pydicom-and-torchmetrics/pydicom-2.4.3-py3-none-any.whl')
get_ipython().system('pip install /kaggle/input/pydicom-and-torchmetrics/torchmetrics-1.2.0-py3-none-any.whl')

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

random_seed = config.SEED
np.random.seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

IMG_DIR = '/tmp/dataset/rsna-atd'
BASE_PATH = '/kaggle/input/rsna-2023-abdominal-trauma-detection'
IMAGE_DIR = '/tmp/dataset/rsna-atd'
INPUT_MODEL_PATH = '/kaggle/input/pytorchcv-starter-notebook-train/model_5.pth'
MODEL_PATH = '/kaggle/working/model_5.pth'
STRIDE = 10
get_ipython().system('rm -r {IMG_DIR}')
os.makedirs(f'{IMG_DIR}/test_images', exist_ok = True)
# test_df = pd.read_csv(f'{BASE_PATH}/test_series_meta.csv')

# test_df['dicom_folder'] = BASE_PATH + '/' + 'test_images'\
#                                     + '/' + test_df.patient_id.astype(str)\
#                                     + '/' + test_df.series_id.astype(str)


meta_df = pd.read_csv(f"{BASE_PATH}/test_series_meta.csv")

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

file_paths = test_df.dicom_path.tolist()
_ = Parallel(n_jobs=2, backend="threading")(
    delayed(resize_and_save)(file_path) for file_path in tqdm(file_paths, leave=True, position=0)
)

del _; gc.collect()

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

# def post_proc_v4(pred):
#     proc_pred = np.empty((1, 2*2 + 3*3), dtype='float32')

#     bowel = pred[0] if 0.2 < pred[0] < 0.8 else np.float32(0.020337)
#     extra = pred[1] if 0.3 < pred[0] < 0.7 else np.float32(0.063553)
#     kidney = pred[2:5] if np.max(pred[2:5]) > 0.9 else np.array([0.942167, 0.036543, 0.02129])
#     liver = pred[5:8] if np.max(pred[5:8]) > 0.9 else np.array([0.897998, 0.082301, 0.019701])
#     spleen = pred[8:11] if np.max(pred[8:11]) > 0.9 else np.array([0.887512, 0.063235, 0.049253])

#     # bowel, extravasation
#     proc_pred[:, 0] = 1 - bowel         # bowel-healthy
#     proc_pred[:, 1] = bowel             # bowel-injured
#     proc_pred[:, 2] = 1 - extra         # extra-healthy
#     proc_pred[:, 3] = extra             # extra-injured

#     # liver, kidney, sneel
#     proc_pred[:, 4:7] = kidney
#     proc_pred[:, 7:10] = liver
#     proc_pred[:, 10:13] = spleen

#     return proc_pred

def post_proc_v3(pred):
    proc_pred = np.empty((1, 2*2 + 3*3), dtype='float32')

    # bowel, extravasation
    proc_pred[:, 0] = 1 - pred[0] # bowel-healthy
    proc_pred[:, 1] = pred[0] # bowel-injured
    proc_pred[:, 2] = 1 - pred[1] # extra-healthy
    proc_pred[:, 3] = pred[1] # extra-injured

    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[2:5]
    proc_pred[:, 7:10] = pred[5:8]
    proc_pred[:, 10:13] = pred[8:11]

    return proc_pred

def post_proc_v4(pred):
    proc_pred = np.empty((1, 2*2 + 3*3), dtype='float32')
    print('print')
    # bowel, extravasation
    proc_pred[:, 0] = np.where((0.3 < pred[0]) & (pred[0] < 0.7), np.float32(0.973956), pred[0])
    proc_pred[:, 1] = np.where((0.3 < pred[0]) & (pred[0] < 0.7), np.float32(0.098461), 1 - pred[0])
    proc_pred[:, 2] = np.where((0.3 < pred[1]) & (pred[1] < 0.7), np.float32(0.931948), pred[1])
    proc_pred[:, 3] = np.where((0.3 < pred[1]) & (pred[1] < 0.7), np.float32(0.104006), 1 - pred[1])

    # kidney, liver, spleen
    proc_pred[:, 4:7] = np.where(np.max(pred[2:5]) > 0.9, pred[2:5], np.array([0.93764, 0.104006, 0.103077]))
    proc_pred[:, 7:10] = np.where(np.max(pred[5:8]) > 0.9, pred[5:8], np.array([0.893683, 0.23424, 0.095384]))
    proc_pred[:, 10:13] = np.where(np.max(pred[8:11]) > 0.9, pred[8:11], np.array([0.883248, 0.179976, 0.140182]))

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

    model_preds = np.zeros(shape=(1, 11), dtype=np.float32)

    # Query the dataframe for a particualr patient
    patient_df = test_df[test_df["patient_id"] == patient_id]

    # Getting image paths for a particular patient
    patient_paths = patient_df.image_path.tolist()

    # Bulding dataset for prediction
    dtest = build_dataset(patient_paths, 1, config.IMAGE_SIZE)

    # Predicting with the model
    model.eval()
    with torch.no_grad():
        preds = []
        for image in dtest:
            image = image.to(device)
            x_bowel, x_extra, x_liver, x_kidney, x_spleen = model(image)

            bowel =F.sigmoid(x_bowel.cpu()).numpy().flatten()
            extra = F.sigmoid(x_extra.cpu()).numpy().flatten()
            kidney = F.softmax(x_liver.cpu(),dim =1).numpy().flatten()
            liver = F.softmax(x_kidney.cpu(),dim =1).numpy().flatten()
            spleen = F.softmax(x_spleen.cpu(),dim =1).numpy().flatten()

            preds.append(np.concatenate((bowel, extra, kidney, liver, spleen), axis=0))

        preds = np.array(preds).astype('float32')

        preds = preds.reshape(len(patient_paths), 11)

        preds = np.max(preds, axis=0)

        patient_preds[pidx, :] += post_proc_v4(preds).reshape((2*2+3*3))

        del patient_df, patient_paths, dtest, bowel, extra, liver, kidney, spleen, preds
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

sub_df.to_csv("submission.csv",index=False, float_format='%.20f')
sub_df.head(3)
