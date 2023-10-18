#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import pandas.api.types
import sklearn.metrics

import os
from glob import glob
from tqdm import tqdm, trange

file_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection"

Target_cols = ["bowel_healthy", "bowel_injury", "extravasation_healthy",
                   "extravasation_injury", "kidney_healthy", "kidney_low",
                   "kidney_high", "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high","any_injury"]

file_path = "/kaggle/input/rsna-2023-abdominal-trauma-detection"
train_csv=f"{file_path}/train.csv"
train=pd.read_csv(train_csv)

organs_healthy = [
          'bowel_healthy',
          'extravasation_healthy',
          'kidney_healthy',
          'liver_healthy',
          'spleen_healthy'
]
#calculate and plot
corr_matrix1=train[organs_healthy].corr()
sns.heatmap(corr_matrix1,annot=True);
plt.title('Correlation Heatmap for healthy organs')

low_high = [
            'bowel_injury',
            'extravasation_injury',
            'kidney_low',
            'kidney_high',
            'liver_high',
            'liver_low' ,
            'spleen_low',
            'spleen_high',
            'any_injury'
]
#calculate and plot
corr_matrix2 = train[low_high].corr()
sns.heatmap(corr_matrix2,annot=True , linewidths=1)
plt.title('Correlation heatmap for injury organs')

train_series_meta = pd.read_csv('/kaggle/input/rsna-2023-abdominal-trauma-detection/train_series_meta.csv')
train_series_meta.head()

test_series_meta=pd.read_csv("/kaggle/input/rsna-2023-abdominal-trauma-detection/test_series_meta.csv")
test_series_meta.head()

image_labels = pd.read_csv( "/kaggle/input/rsna-2023-abdominal-trauma-detection/image_level_labels.csv")
image_labels.head()

class ParticipantVisibleError(Exception):
    pass

def normalize_probabilities_to_one(df: pd.DataFrame, group_columns: list) -> pd.DataFrame:
    # Normalize the sum of each row's probabilities to 100%.
    # 0.75, 0.75 => 0.5, 0.5
    # 0.1, 0.1 => 0.5, 0.5
    row_totals = df[group_columns].sum(axis=1)
    if row_totals.min() == 0:
        raise ParticipantVisibleError('All rows must contain at least one non-zero prediction')
    for col in group_columns:
        df[col] /= row_totals
    return df

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    '''
    Pseudocode:
    1. For every label group (liver, bowel, etc):
        - Normalize the sum of each row's probabilities to 100%.
        - Calculate the sample weighted log loss.
    2. Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    3. Calculate the sample weighted log loss for the new label group
    4. Return the average of all of the label group log losses as the final score.
    '''
    del solution[row_id_column_name]
    del submission[row_id_column_name]
    # Run basic QC checks on the inputs
    if not pandas.api.types.is_numeric_dtype(submission.values):
        raise ParticipantVisibleError('All submission values must be numeric')

    if not np.isfinite(submission.values).all():
        raise ParticipantVisibleError('All submission values must be finite')

    if solution.min().min() < 0:
        raise ParticipantVisibleError('All labels must be at least zero')
    if submission.min().min() < 0:
        raise ParticipantVisibleError('All predictions must be at least zero')

    # Calculate the label group log losses
    binary_targets = ['bowel', 'extravasation']
    triple_level_targets = ['kidney', 'liver', 'spleen']
    all_target_categories = binary_targets + triple_level_targets

    label_group_losses = []
    for category in all_target_categories:
        if category in binary_targets:
            col_group = [f'{category}_healthy', f'{category}_injury']
        else:
            col_group = [f'{category}_healthy', f'{category}_low', f'{category}_high']

        solution = normalize_probabilities_to_one(solution, col_group)

        for col in col_group:
            if col not in submission.columns:
                raise ParticipantVisibleError(f'Missing submission column {col}')
        submission = normalize_probabilities_to_one(submission, col_group)
        label_group_losses.append(
            sklearn.metrics.log_loss(
                y_true=solution[col_group].values,
                y_pred=submission[col_group].values,
                sample_weight=solution[f'{category}_weight'].values
            )
        )

    # Derive a new any_injury label by taking the max of 1 - p(healthy) for each label group
    healthy_cols = [x + '_healthy' for x in all_target_categories]
    any_injury_labels = (1 - solution[healthy_cols]).max(axis=1)
    any_injury_predictions = (1 - submission[healthy_cols]).max(axis=1)
    any_injury_loss = sklearn.metrics.log_loss(
        y_true=any_injury_labels.values,
        y_pred=any_injury_predictions.values,
        sample_weight=solution['any_injury_weight'].values
    )

    label_group_losses.append(any_injury_loss)
    return np.mean(label_group_losses)

# Assign the appropriate weights to each category
def create_training_solution(y_train):
    sol_train = y_train.copy()

    # bowel healthy|injury sample weight = 1|2/1
    sol_train['bowel_weight'] = np.where(sol_train['bowel_injury'] == 1, 2, 1)

    # extravasation healthy/injury sample weight = 1|6/1
    sol_train['extravasation_weight'] = np.where(sol_train['extravasation_injury'] == 1, 6, 1)

    # kidney healthy|low|high sample weight = 1|2|4
    sol_train['kidney_weight'] = np.where(sol_train['kidney_low'] == 1, 2, np.where(sol_train['kidney_high'] == 1, 4, 1))

    # liver healthy|low|high sample weight = 1|2|4
    sol_train['liver_weight'] = np.where(sol_train['liver_low'] == 1, 2, np.where(sol_train['liver_high'] == 1, 4, 1))

    # spleen healthy|low|high sample weight = 1|2|4
    sol_train['spleen_weight'] = np.where(sol_train['spleen_low'] == 1, 2, np.where(sol_train['spleen_high'] == 1, 4, 1))

    # any healthy|injury sample weight = 1|6/1
    sol_train['any_injury_weight'] = np.where(sol_train['any_injury'] == 1, 6, 1)
    return sol_train

solution_train = create_training_solution(train)

# predict a constant using the mean of the training data
y_pred = train.copy()
y_pred[Target_cols] = train[Target_cols].mean().tolist()

no_scale_score = score(solution_train,y_pred,'patient_id')
print(f'Training score without scaling: {no_scale_score}')

# Group by different sample weights
scale_by_2 = ['kidney_low','liver_low','spleen_low','spleen_high']
scale_by_4 = ['bowel_injury','kidney_high','liver_high']
scale_by_6 = ['extravasation_injury','any_injury']
scale_healthy = ['bowel_healthy', 'extravasation_healthy', 'kidney_healthy', 'liver_healthy', 'spleen_healthy']

# Scale factors based on described metric
sf_2 = 2.8461531332 * 0.99999999
sf_4 = 4.841531 * 0.99999999
sf_6 = 20.81635153 * 0.99999999
scale_h = 0.99519515313 * 0.99999999

# The score function deletes the ID column so we remake it
solution_train = create_training_solution(train)

# Reset the prediction
y_pred = train.copy()
y_pred[Target_cols] = train[Target_cols].mean().tolist()

# Scale each target
y_pred[scale_by_2] *=sf_2
y_pred[scale_by_4] *=sf_4
y_pred[scale_by_6] *=sf_6
y_pred[scale_healthy] *=scale_h

weight_scale_score = score(solution_train,y_pred,'patient_id')
print(f'Training score with weight scaling: {weight_scale_score}')

# Update scale factors to improve score
# sf_2 = 4
# sf_4 = 6
# sf_6 = 28

# The score function deletes the ID column so we remake it
solution_train = create_training_solution(train)

# Reset the prediction, again
y_pred = train.copy()
y_pred[Target_cols] = train[Target_cols].mean().tolist()

# Scale each target
y_pred[scale_by_2] *=sf_2
y_pred[scale_by_4] *=sf_4
y_pred[scale_by_6] *=sf_6
y_pred[scale_healthy] *=scale_h

improved_scale_score = score(solution_train,y_pred,'patient_id')
print(f'Training score with better scaling: {improved_scale_score}')

submission = pd.read_csv('/kaggle/input/rsna-2023-abdominal-trauma-detection/sample_submission.csv')

# Set output to mean of training data
submission[Target_cols] = train[Target_cols].mean().tolist()

# Scale each category by desired scale factor
submission[scale_by_2] *=sf_2
submission[scale_by_4] *=sf_4
submission[scale_by_6] *=sf_6
submission[scale_healthy] *=scale_h

# Save Submission!
submission.to_csv('submission.csv', index=False)
