import numpy as np
import matplotlib.pyplot as plt
from Hyperparameter_optimization import *
from proj1_helpers import *
from preprocess_data import *

## Load the training set
print('- Loading data from training.csv')
DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
y_train, tX_train, ids_train = load_csv_data(DATA_TRAIN_PATH)
feature_names = load_headers(DATA_TRAIN_PATH)
print('  * done')
# convert y from -1/1 to 0,1
y_train = (y_train+1)/2



## Load the test set

print('- Loading data from test.csv')
DATA_TEST_PATH = '../data/test.csv' # TODO: download test data and supply path here 
y_test, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)
feature_names = load_headers(DATA_TEST_PATH)
print('  * done')

# Train a model and see accuracy
data_pro  = data_preprocess_train(tX_train,y_train,outlier_method = 'IQR',n_out = 5)

# Train
print('- Training a ridge-regression model')
lambda_ = [6.5e-05,
 1e-05,
 0.0009,
 0.000835,
 0.00078,
 0.000615] # lambda values for each group
degrees = [10, 10, 10, 4, 10, 5]
loss_ = [] 
w_ = []
for idx_data,data in enumerate(data_pro):
    #print('┌'+'─' * 38+'┐')
    #print('│               Group: %i               │'%idx_data)
    #print('└'+'─' * 38+'┘')
    tx = data[0]
    tx = np.hstack((build_poly(tx, degrees[idx_data]),np.sin(tx)))
    y = data[1]. reshape(-1,1)
    initial_w_ = np.zeros((tx.shape[1],1))
    loss_temp, w_temp = ridge_regression(y, tx,lambda_[idx_data])
    w_.append(w_temp)
#print('\n * Overall Accuracy = %.4f'%accuracy_all)
print('  * done')

## Do prediction on test set
print('- Use trained model to predict the test.csv data')
data_test  = data_preprocess_predict(tX_test,outlier_method = 'IQR',n_out = 5)
# Using trained model to do prediction
label_ = []
data_size = sum([len(data_test[:,1][i]) for i in range(6)])
y_predict = np.zeros(data_size)
for idx_data,data in enumerate(data_test):
    tx = data[0]
    tx = np.hstack((build_poly(tx, degrees[idx_data]),np.sin(tx)))
    label_.append(predict_labels(w_[idx_data], tx,model='ridge_regression'))
    y_predict[data[1]] = predict_labels(w_[idx_data], tx,model='ridge_regression')

create_csv_submission(range(350000,350000+data_size),(y_predict-0.5)*2,'prediction.csv')

print('  * done, the results is saved in \'prediction.csv\'')