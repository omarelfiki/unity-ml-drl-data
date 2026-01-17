# Machine Learning Prediction Models

## Overview
This directory contains the code for training and evaluating machine learning models for predicting reachability, steps, and time-to-threshold based on the collected datasets. The models are implemented using scikit-learn and can be trained and evaluated using the prediction command-line interfaces (CLI).

## Models CLI
Try Models CLI on all collected environments and datasets through GitHub actions here: https://github.com/omarelfiki/unity-ml-drl-data/actions/workflows/run_models.yml

Once completed, the results will be available in the artifacts tab of the workflow run as a zip file containing the collected models and results. A sample of the results can be found in the ```experiments/``` directory.

The ML models CLI can be used to train and evaluate the models on the collected datasets. Existing models can be used to make predictions on new data using the ```--models-dir``` flag.

From within the ```models/``` directory:
```
usage: python -m scripts.run [-h] [--test_size TEST_SIZE] [--seed SEED] [--thresh THRESH] [--env ENV] [--models-dir MODELS_DIR] [--data-csv, type=str DATA_CSV]

options:
  -h, --help            show this help message and exit
  --test_size           Fraction of data used as test set (default: 0.2).
  --seed                Random seed for reproducible splitting (default: 42).
  --thresh              Threshold for pred_reach from p_reach (default: 0.5).
  --env ENV             Environment to use (default: None).
  --models-dir          Directory containing standard model names: `logistic_reach_model.joblib`, `linear_steps_model.joblib`, `linear_time_model.joblib`.
  --data-csv            Path to CSV containing data. Defaults to latest data in ../data/normalized/.
```
The prediction CLI will output a versioned directory under ```experiments/``` containing the results of the prediction such as metadata, joblib files containing the trained models, and a CSV file containing the predictions.

## Cross Validation
The CLI can be used to perform cross validation on the collected datasets.

## Analysis 
The ```analysis/``` directory within each expirment contains results of the analysis of the models.
Analysis is performed using statistical tests and plots which can be found in the analysis directory.



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