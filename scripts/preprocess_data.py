import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


def data_cleaning(Data_TRAIN_PATH):

    def out_cleaner(tx,method='mean'):
    tx_mean = np.mean(tx)
    tx_std = np.std(tx)
    out_idx = np.where(np.abs(tx-tx_mean) > 3*tx_std)[0]
    good_idx = np.where(np.abs(tx-tx_mean) < 3*tx_std)[0]
    if method == 'mean':
        tx[out_idx]= np.mean(tx[good_idx])
    elif method == 'median':
        tx[out_idx] = np.median(tx[good_idx])
    elif method == 'cloest':
        out_pos_idx = np.where(tx-t_mean > 3*tx_std)[0]
        out_neg_idx = np.where(t_mean-tx > -3*tx_std)[0]
        tx[out_pos_idx] = np.max(tx[good_idx])
        tx[out_neg_idx] = np.min(tx[good_idx])
    return (tx-tx_mean)/tx_std # standarlization

    DATA_TRAIN_PATH = '../data/train.csv' # TODO: download train data and supply path here 
    y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
    feature_names = load_headers(DATA_TRAIN_PATH)

    # convert y from -1/1 to 0,1
    y = (y+1)/2

    # delete null mass data
    idx_null_mass = np.where(tX[:,0]==-999)[0]
    tX = np.delete(tX,idx_null_mass,0)
    y = np.delete(y,idx_null_mass,0)

    # Regouping data
    ## Find the index of each group based on PRI_jet_num
    idx0 = np.where(tX[:,22]==0)[0]
    idx1 = np.where(tX[:,22]==1)[0]
    idx23 = np.where(tX[:,22]>1)[0]

    ## Regroup
    y0 = y[idx0]
    tX0 = tX[idx0,:]

    y1 = y[idx1]
    tX1 = tX[idx1,:]

    y23 = y[idx23]
    tX23 = tX[idx23,:]

    ## Clean undefined features of each group (still keep null in DER_mass_MMC)
    null_idx_0 = [ 4,  5,  6, 12, 22, 23, 24, 25, 26, 27, 28,29]
    null_idx_1 = [ 4,  5,  6, 12, 22, 26, 27, 28]
    null_idx_23 = [22]

    tX0 = np.delete(tX0,null_idx_0,1)
    tX1 = np.delete(tX1,null_idx_1,1)
    tX23 = np.delete(tX23,null_idx_23,1)

    ## Recount the number of feature of each group
    n_feature_0, n_feature_1, n_feature_23 = (tX0.shape[1], tX1.shape[1], tX23.shape[1])

    ## feature names
    feature_names_0 = np.delete(feature_names,null_idx_0,0)
    feature_names_1 = np.delete(feature_names,null_idx_1,0)
    feature_names_23 = np.delete(feature_names,null_idx_23,0)
    ## data_size of each group
    n_0 , n_1, n_23 = (len(y0), len(y1), len(y23))

    for ite,tx in enumerate(tX0.T):
        tX0[:,ite] = out_cleaner(tx)
    for ite,tx in enumerate(tX1.T):
        tX1[:,ite] = out_cleaner(tx)
    for ite,tx in enumerate(tX23.T):
        tX23[:,ite] = out_cleaner(tx)

    return [[tX0, y0], [tX1, y1], [tX23,y23]