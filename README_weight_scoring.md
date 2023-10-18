# Kaggle RSNA 2023 Abdominal Trauma Detection

This model is designed to detect several potential injuries of trauma patients from the Kaggle RSNA 2023 Abdominal Trauma Detection dataset. The goal is to analyze and predict abdominal injuries using the provided dataset. The dataset consists of injury labels for Bowel, Extravasation, Liver, Kidney, and Spleen.

## Data Normalization and Scoring
- Defined a function to normalize the probabilities of each injuries so that they sum up to 1.
- Scoring function calculates the label group log losses for each organs and their respective states (healthy, low, high). Also, it derives a new label called 'any_injury' by taking the max of 1 minus the probability of healthy for each label group.

## Sample Weight Assignment
- A function is created to assign appropriate weights to each category based on their state.

## Prediction Scaling and Submission
- Weights are applied to the predictions.
- An average prediction is created by taking the mean of the training data.
- Predictions are scaled by certain scale factors.
- Final predictions are saved to a submission.csv file.
