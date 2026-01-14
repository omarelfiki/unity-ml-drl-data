# Machine Learning Prediction Models

## Overview
This directory contains the code for training and evaluating machine learning models for predicting reachability, steps, and time-to-threshold based on the collected datasets. The models are implemented using scikit-learn and can be trained and evaluated using the prediction command-line interfaces (CLI).

## Prediction CLI
The prediction CLI can be used to train and evaluate the models on the collected datasets. Existing models can be used to make predictions on new data using the ```--models-dir``` flag.

From within the ```models/``` directory:
```
usage: python -m scripts.predict [-h] [--test_size TEST_SIZE] [--seed SEED] [--thresh THRESH] [--models-dir MODELS_DIR]

options:
  -h, --help            show this help message and exit
  --test_size TEST_SIZE Fraction of data used as test set (default: 0.2).
  --seed SEED           Random seed for reproducible splitting (default: 42).
  --thresh THRESH       Threshold for pred_reach from p_reach (default: 0.5).
  --models-dir <path>   Directory containing standard model names: `logistic_reach_model.joblib`, `linear_steps_model.joblib`, `linear_time_model.joblib`.
```
The prediction CLI will output a versioned directory under ```collected_models/``` containing the results of the prediction such as metadata, joblib files containing the trained models, and a CSV file containing the predictions.

## Models Details
### 1. Logistic Regression
The logistic regression model is trained on the reachability data and predicts reachability probabilities.
### 2. Linear Regression
The linear regression models are trained on the steps and time-to-threshold data and predict reachability, steps, and time-to-threshold.

## Datasets and Validation
The used dataset can be found in the ```data/``` directory as ```prediction_snapshot.csv```. 
**Validation note to be added**

## Results
Complete results are available in the ```collected_models/``` directory. Once a final model is selected, the result for that model will be added to the ```results/``` directory.