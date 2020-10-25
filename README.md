# EPFL Machine Learning Higgs - Spot the Boson


This repository contains our work for Project 1 of CS-433 machine learning course at EPFL. In this project, we build a classification model to identify the signals of the Higgs boson versus the background using experimental data detected by ATLAS

## Structure of this project
```
project
│   README.md
|   report.pdf
│   project1_description.pdf
│   LICENSE   
│
├───data
│      train.csv # the data for training
│      test.csv # the data we need to give predictions
│       
│   
└───scripts
        Cross_validation.py             # functions for cross validation test
        preprocess_data.py              # functions for data preprocess
        Implementations.py              # model implementations
        Implementations_helpers.py      # helper function of model implementations
        Hyperparameter_optimization.py  # functions for optimizing hyperparameter
        proj1_helpers.py                # helper functions
        run.py                          # to generate the predictions of test.csv
        prediction.csv                  # the predictions of test.csv
   
```
## Data preprocessing

We use functions written in `./scripts/preprocess_data.py` to preprocess the sample features. 

The preprocessing includes the following steps:

- Regroup data based on `PRI_jet_num` to find 3 groups.
- Regroup every group to two subgroups according to null values in `DER_mass_MMC`. 
- Clean outliers using IQR
- Standardize features

Note that since the labels are changed from {- 1, 1} to {0, 1} for training, we modify the `predict_labels` function in `proj1_helper1.py` accordingly.



## Methods

Six machine learning methods are implemented for this Higgs boson identification task:

- Linear Regression
- Linear regression with gradient descent
- Linear regression with stochastic gradient descent
- Ridge regression
- Logistic regression
- Regularized logistic regression

The implementations of these methods are in `./data/implementation.py`,

## Result

On [Aicrowd Leaderboard](https://www.aicrowd.com/challenges/epfl-machine-learning-higgs/leaderboards):

|Team|Categorical Accuracy  |F1-score|
|:---:|:---:|:---:|
|Bio.Chem.Env.Mater.|0.832|0.744|