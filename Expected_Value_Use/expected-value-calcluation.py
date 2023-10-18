#!/usr/bin/env python
# coding: utf-8

# For caculating expected value

import numpy as np
import pandas as pd

file_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection"

Target_cols = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high","any_injury"]

file_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection"
train_csv=f"{file_path}/train.csv"
train=pd.read_csv(train_csv)

# Align with sample submission
sub_df = pd.read_csv(f"{file_path}/sample_submission.csv")

for col, val in zip(Target_cols, train[Target_cols].mean()):
    sub_df[col] = val

# Store submission
sub_df.to_csv("submission.csv",index=False)
print(sub_df.head(2))
