import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *


## Clean undefined features of each group (still keep null in DER_mass_MMC)
null_idx_0 = [ 4,  5,  6, 12, 22, 23, 24, 25, 26, 27, 28,29]
null_idx_1 = [ 4,  5,  6, 12, 22, 26, 27, 28]
null_idx_23 = [22]

def jet_num_regroup(tx, y=[]):
    # Regouping data
    ## Find the index of each group based on PRI_jet_num
    idx0 = np.where(tx[:,22]==0)[0]
    idx1 = np.where(tx[:,22]==1)[0]
    idx23 = np.where(tx[:,22]>1)[0]
    tx0 = tx[idx0,:]
    tx1 = tx[idx1,:]
    tx23 = tx[idx23,:]
    tx0 = np.delete(tx0,null_idx_0,1)
    tx1 = np.delete(tx1,null_idx_1,1)
    tx23 = np.delete(tx23,null_idx_23,1)
    if y != []:
        ## Regroup
        y0 = y[idx0]
        y1 = y[idx1]
        y23 = y[idx23]
        return np.array([[tx0, y0,idx0], [tx1, y1,idx1], [tx23, y23, idx23]],dtype=object)
    else:
        # to group data of the data we need to predict, for which we don't know y 
        return np.array([[tx0, idx0], [tx1, idx1], [tx23, idx23]],dtype=object)
    '''
    ## Recount the number of feature of each group
    n_feature_0, n_feature_1, n_feature_23 = (tx0.shape[1], tx1.shape[1], tx23.shape[1])

    ## feature names
    feature_names_0 = np.delete(feature_names,null_idx_0,0)
    feature_names_1 = np.delete(feature_names,null_idx_1,0)
    feature_names_23 = np.delete(feature_names,null_idx_23,0)

    ## data_size of each group
    n_0 , n_1, n_23 = (len(y0), len(y1), len(y23))
    '''

def visualize_features(tx,y=[]):
    n_features = tx.shape[1]
    if y != []:
        # signals and backgrounds
        sig = np.where(y==1)[0]
        back = np.where(y==0)[0]
        tX_sig = tx[sig,:]
        tX_back = tx[back,:]
    f, axs = plt.subplots(int(np.ceil(tx.shape[1]/3)), 3,figsize=(30,25))
    axs = axs.ravel()
    for idx in range(n_features):
        if y != []:
            axs[idx].hist(tX_sig[:,idx],density=True, bins=50,alpha=0.5,label = 'signal')
            axs[idx].hist(tX_back[:,idx],density=True, bins=50,alpha=0.5,label = 'background')
            axs[idx].legend(loc='upper right')
        else:
             axs[idx].hist(tx[:,idx],density=True, bins=50,alpha=0.5,label = 'signal')
        #ax.set_title(feature_names[idx])

    

def mass_regroup(tx,y=[]):
    # delete null mass data
    idx_null_mass = np.where(tx[:,0]==-999)[0]
    idx_good_mass = np.where(tx[:,0]!=-999)[0]

    tx_good_mass = np.delete(tx,idx_null_mass,0)
    tx_null_mass = np.delete(tx,idx_good_mass,0)
    # delete mass feature for null mass data
    tx_null_mass = np.delete(tx_null_mass,0,1)
    if y != []:
        y_good_mass = np.delete(y,idx_null_mass,0)
        y_null_mass = np.delete(y,idx_good_mass,0)     
        return np.array([[tx_good_mass,y_good_mass,idx_good_mass],[tx_null_mass,y_null_mass,idx_null_mass]],dtype=object)
    else:
        return np.array([[tx_good_mass,idx_good_mass],[tx_null_mass,idx_null_mass]],dtype=object)
    #else:
        #print('The mode must be train or predict')

def outlier_cleaning_IQR(tx,n_IQR):
    for i in range(tx.shape[1]):
        q1_tx = np.quantile(tx[:,i], 0.25,axis=0)
        q3_tx = np.quantile(tx[:,i], 0.75,axis=0)
        iqr_tx = q3_tx - q1_tx
        lower = q1_tx - n_IQR * iqr_tx
        upper = q1_tx + n_IQR * iqr_tx
        
        #calculate mean value for each feature
        mean = np.mean(tx[:,i])
        median = np.median(tx[:,i])
        
        old_col=tx[:,i]
        arr_1=np.where(old_col < lower)
        new_col_1=np.delete(old_col,arr_1)
        arr_2=np.where(old_col > upper)
        new_col_2=np.delete(old_col,arr_2)
        new_mean=np.mean(new_col_2)
        old_col[old_col < lower]=new_mean 
        old_col[old_col > upper]=new_mean
    return tx

def outlier_cleaning_sigma(tx,n_sigma):
    for ite,ite_tx in enumerate(tx.T):
        ite_tx_mean = np.mean(ite_tx)
        ite_tx_std = np.std(ite_tx)
        outlier_idx = np.where(np.abs(ite_tx-ite_tx_mean) > n_sigma*ite_tx_std)[0]
        good_idx = np.where(np.abs(ite_tx-ite_tx_mean) < n_sigma*ite_tx_std)[0]
        tx[outlier_idx,ite]= np.mean(ite_tx[good_idx])
    return tx

def feature_normalize(tx):
    mu = np.mean(tx,axis=0)
    sigma = np.std(tx,axis=0)
    return (tx-mu)/sigma

def data_preprocess_predict(tx,outlier_method = 'IQR',n_out = 3):
    '''
    *Preprocess the unlabeled data (for which we need to use our model to predict)
    Input:  tx - features
            outlier_method - the method used to deal with outliers in features
                             'IQR' or 'sigma
            n_out - the range of good data, range_good = tx_mean +- n_out * (IQR or sigma)

    Output: data_regrouped - array of with 6 element, each element is an groupe of data set (regrouped based on jet_num as mass)

            data_regrouped = [data_groupe_1, ... , data_groupe_6]
            data_groupe_1 = [tx_1,y_1,idx_1]
            
            where id_1 is the corresponding idxes in the original dataset
    '''
    data_regrouped = []
    # firstly regroup data based on jet number
    data_grouped_jet = jet_num_regroup(tx)
    #secondly regroup data based mass
    for data in data_grouped_jet:
        data_good_mass,data_null_mass = mass_regroup(data[0])
        # find the corresponding indexes of the original dataset
        idx_good_mass_ori = data[-1][data_good_mass[-1]]
        idx_null_mass_ori = data[-1][data_null_mass[-1]]
        data_good_mass[-1],data_null_mass[-1] = [idx_good_mass_ori, idx_null_mass_ori]
        # grouped data
        data_regrouped.append(data_good_mass)
        data_regrouped.append(data_null_mass)
    data_regrouped = np.array(data_regrouped, dtype = object) 
    #At last, standardize features
    if outlier_method == 'sigma':
        for i in range(6):
            data_regrouped[i,0] = outlier_cleaning_sigma(data_regrouped[i,0],n_sigma = n_out) #clear outliers
            data_regrouped[i,0] = feature_normalize(data_regrouped[i,0])
    elif outlier_method == 'IQR':
        for i in range(6):
            data_regrouped[i,0] = outlier_cleaning_sigma(data_regrouped[i,0],n_sigma = n_out)
            data_regrouped[i,0] = feature_normalize(data_regrouped[i,0])
    return data_regrouped



def data_preprocess_train(tx,y,outlier_method = 'IQR',n_out = 3):
    '''
    *Preprocess the training data
    Input:  tx - features
            y - labels (0,1)
            outlier_method - the method used to deal with outliers in features
                             'IQR' or 'sigma
            n_out - the range of good data, range_good = tx_mean +- n_out * (IQR or sigma)

    Output: data_regrouped - array of with 6 element, each element is an groupe of data set (regrouped based on jet_num as mass)

            data_regrouped = [data_groupe_1, ... , data_groupe_6]
            data_groupe_1 = [tx_1,y_1,idx_1]
            
            where id_1 is the corresponding idxes in the original dataset
    '''
    data_regrouped = []
    # firstly regroup data based on jet number
    data_regrouped_jet = jet_num_regroup(tx,y)

    #secondly regroup data based mass
    for data in data_regrouped_jet:
        data_good_mass,data_null_mass = mass_regroup(data[0],data[1])
        # find the corresponding indexes of the original dataset
        idx_good_mass_ori = data[-1][data_good_mass[-1]]
        idx_null_mass_ori = data[-1][data_null_mass[-1]]
        data_good_mass[-1],data_null_mass[-1] = [idx_good_mass_ori, idx_null_mass_ori]
        # grouped data
        data_regrouped.append(data_good_mass)
        data_regrouped.append(data_null_mass)
        
    data_regrouped = np.array(data_regrouped, dtype = object) 
    #At last, standardize features
    if outlier_method == 'sigma':
        for i in range(6):
            data_regrouped[i,0] = outlier_cleaning_sigma(data_regrouped[i,0],n_sigma = n_out) #clear outliers
            data_regrouped[i,0] = feature_normalize(data_regrouped[i,0])
    elif outlier_method == 'IQR':
        for i in range(6):
            data_regrouped[i,0] = outlier_cleaning_sigma(data_regrouped[i,0],n_sigma = n_out)
            data_regrouped[i,0] = feature_normalize(data_regrouped[i,0])
    return data_regrouped

        