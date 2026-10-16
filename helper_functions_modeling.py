# ------------------------
#Library of Alexandria
# ------------------------

import numpy as np
import glob
import natsort
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.family':'DejaVu Sans'})
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import PowerNorm
from matplotlib.widgets import Button
import os
import os.path as op
from pathlib import Path
import scipy
from scipy import stats
from scipy.stats import zscore
import scipy.signal as signal
from scipy import stats,signal,integrate
from scipy.signal import butter, filtfilt, cheby1,hilbert
from scipy.ndimage import gaussian_filter1d
from statsmodels.regression.linear_model import yule_walker as sm_yw
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn import metrics, preprocessing
from sklearn.feature_selection import RFE
import imblearn
from imblearn import over_sampling
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ks_2samp
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import tkinter as tk
from tkinter import simpledialog
import joblib
from copy import deepcopy as cdc
import time
import datetime
today = datetime.datetime.today().strftime('%Y-%m-%d')
from datetime import datetime, timedelta
import seaborn as sns
import neuraltoolkit as ntk
import james_work as jmw
import re
import elephant
#import criticality as cr
#from criticality import pvaluenew2 as pv
#from criticality import exclude as ex
from PIL import Image
from IPython.display import display
from IPython.display import display, clear_output
from tqdm import tqdm
from scipy.stats import sem
import pickle
import time
import logging
from helper_functions_math_addons import *
from helper_functions_file_handling import *
from imblearn.over_sampling import SMOTE
from collections import Counter
from threadpoolctl import threadpool_info, threadpool_limits
from alternateMLPClassifier import AlternateMLPClassifier
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# extract sample, calculate spike matrix based on extracted data, and return the mean of difference
def extract_data(df, n_vals):
    # get spike matrix at first n isis
    m1 = []
    m2 = []
    m3 = []
    m4 = []
    m5 = []
    m6 = []
    m7 = []
    m8 = []
    m9 = []
    m10 = []
    for n in n_vals:
        colname = f'isis{n}'
        df[colname] = df['isis'].apply(sample_n_elements, n = n)

        # get the first n ifr
        colname_matrix = f'ifr_{n}'
        colname_diff = f'diff_ifr_{n}'   
        df[colname_matrix] = df[colname].apply(compute_ifr)
        df[colname_diff] = np.abs(df[colname_matrix] - df['ifr'])/df['ifr'] 
        t = df[colname_diff].mean()
        m1.append(t)

        # get the first n mean isis
        colname_matrix = f'mean_isis_{n}'
        colname_diff = f'diff_mean_isis_{n}'
        df[colname_matrix] = df[colname].apply(compute_mean)
        df[colname_diff] = np.abs(df[colname_matrix] - df['mean_isis'])/df['mean_isis'] 
        t = df[colname_diff].mean()
        m2.append(t)

        # get the first n median isis
        colname_matrix = f'median_isis_{n}'
        colname_diff = f'diff_median_isis_{n}'
        df[colname_matrix] = df[colname].apply(compute_median)
        df[colname_diff] = np.abs(df[colname_matrix] - df['median_isis'])/df['median_isis'] 
        t = df[colname_diff].mean()
        m3.append(t)

         # get the first n max isis
        colname_matrix = f'max_isis_{n}'
        colname_diff = f'diff_max_isis_{n}'
        df[colname_matrix] = df[colname].apply(compute_max)
        df[colname_diff] = np.abs(df[colname_matrix] - df['max_isis'])/df['max_isis'] 
        t = df[colname_diff].mean()
        m4.append(t)

        # get the first n min isis
        colname_matrix = f'min_isis_{n}'
        colname_diff = f'diff_min_isis_{n}'
        df[colname_matrix] = df[colname].apply(compute_min)
        df[colname_diff] = np.abs(df[colname_matrix] - df['min_isis'])/df['min_isis'] 
        t = df[colname_diff].mean()
        m5.append(t)

         # get the first n sd
        colname_matrix = f'sd_{n}'
        colname_diff = f'diff_sd_{n}'
        df[colname_matrix] = df[colname].apply(compute_std)
        df[colname_diff] = np.abs(df[colname_matrix] - df['sd'])/df['sd'] 
        t = df[colname_diff].mean()
        m6.append(t)

        # get the first n cv
        colname_matrix = f'cv_{n}'
        colname_diff = f'diff_cv_{n}'
        df[colname_matrix] = df[colname].apply(compute_cv)
        df[colname_diff] = np.abs(df[colname_matrix] - df['cv'])/df['cv'] 
        t = df[colname_diff].mean()
        m7.append(t)

        # get the first n cv2
        colname_matrix = f'cv2_{n}'
        colname_diff = f'diff_cv2_{n}'
        df[colname_matrix] = df[colname].apply(compute_cv2)
        df[colname_diff] = np.abs(df[colname_matrix] - df['cv2'])/df['cv2'] 
        t = df[colname_diff].mean()
        m8.append(t)

        # get the first n lv
        colname_matrix = f'lv_{n}'
        colname_diff = f'diff_lv_{n}'
        df[colname_matrix] = df[colname].apply(compute_lv)
        df[colname_diff] = np.abs(df[colname_matrix] - df['lv'])/df['lv'] 
        t = df[colname_diff].mean()
        m9.append(t)

        # get the first n lvr
        colname_matrix = f'lvr_{n}'
        colname_diff = f'diff_lvr_{n}'
        df[colname_matrix] = df[colname].apply(compute_lvr)
        df[colname_diff] = np.abs(df[colname_matrix] - df['lvr'])/df['lvr'] 
        t = df[colname_diff].mean()
        m10.append(t)
        
    data = {
    'ifr': m1,
    'mean_isis': m2,
    'median_isis': m3,
    'max_isis': m4,
    'min_isis': m5,
    'sd': m6,
    'cv': m7,
    'cv2': m8,
    'lv': m9,
    'lvr': m10
    }
    return data    

# normalize the training set; return the normalized data and scalers   
def scaling_x(df_train):
    total_cell = df_train['cell'].unique()
    scaled_data = []
    scalers = []
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_train.groupby(df_train.cell)
    for cells in tqdm(total_cell):
        #train = df_train.loc[df_train['cell'] == cells]
        train = grouped.get_group(cells)
        X_train = train[spike_matrix]
        y_train = train['state']
        
        # scale X_train
        scaler = preprocessing.StandardScaler()
        X_train = scaler.fit_transform(X_train)
        scalers.append({
            'scaler': scaler,
            'cell': cells
        })
        
        # convert the scaled data into data frame
        train_scaled_df = pd.DataFrame(X_train, columns=spike_matrix)
        train_scaled_df['state'] = y_train.values
        train_scaled_df['cell'] = train['cell'].values
        train_scaled_df['animal'] = train['animal'].values
        train_scaled_df['region'] = train['region'].values
        train_scaled_df['celltype'] = train['celltype'].values
        train_scaled_df['spike_num'] = train['spike_num'].values
        train_scaled_df['state_spikes'] = train['state_spikes'].values
        train_scaled_df['state_length'] = train['state_length'].values
        train_scaled_df['state_start_end'] = train['state_start_end'].values 
        scaled_data.append(train_scaled_df)
    scaled_data = pd.concat(scaled_data).reset_index(drop=True)
    return scaled_data, scalers

# normalize the testing set based on normalization of training set
def scaling_y(df_test, scaler):
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    X_test = df_test[spike_matrix]
    y_test = df_test['state']
    
    X_test = scaler.transform(X_test)
    test_scaled_df = pd.DataFrame(X_test, columns=spike_matrix)
    test_scaled_df['state'] = y_test.values
    test_scaled_df['cell'] = df_test['cell'].values
    test_scaled_df['animal'] = df_test['animal'].values
    test_scaled_df['region'] = df_test['region'].values
    test_scaled_df['celltype'] = df_test['celltype'].values
    test_scaled_df['spike_num'] = df_test['spike_num'].values
    test_scaled_df['state_spikes'] = df_test['state_spikes'].values
    test_scaled_df['state_length'] = df_test['state_length'].values
    test_scaled_df['state_start_end'] = df_test['state_start_end'].values 
    return test_scaled_df

# get coefficients for models
def get_coef(models):
    coefs = {}
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    for item in models:
        model = item['model']
        coef = model.coef_
        coefs[item['cell']] = coef
    return coefs

# average balanced accuray
def accuracy(confusion_matrix):
    s = 0
    for i in range(len(confusion_matrix)):
        s += confusion_matrix[i][i]
    average = s/len(confusion_matrix)
    
    return average

# for a list of models, perform recursive feature selection for each model
def perform_rfe(models, df_train):
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    
    feature_importance = []
    feature_names = spike_matrix
    
    for items in tqdm(models):
        model = items['model']
        cell = items['cell']
        
        df_cell = df_train.loc[df_train['cell'] == cell]
        X_train = df_cell[spike_matrix]
        y_train = df_cell['state']
        
        rfe = RFE(model, n_features_to_select = None)
        rfe = rfe.fit(X_train, y_train)
        importance = rfe.support_
        feature_importance.append(importance)
    importance_df = pd.DataFrame(feature_importance, columns=feature_names)
    return importance_df

#collapse isis from the same cell
def collapse_epoch(df_raw):
    # remove NAs from the dataset
    df = df_raw[['animal', 'block', 'clust_idx_v2', 'state', 'region','celltype', 'isis']].reset_index()
    na_indices = df.index[df.isna().any(axis=1)]
    df = df.drop(na_indices)
    # collapse all epochs by cell
    df_cell = df.groupby(['animal', 'block', 'clust_idx_v2', 'region', 'celltype', 'state']).agg(
        isis=('isis', list)).reset_index()
    df_cell['cell'] = df_cell['animal'] + '_' + df_cell['block'] + '_' + df_cell['clust_idx_v2'].astype('str')
    return df_cell


# split training and testing epochs
def split_epoch(df_raw, r = 0.75):
    df_cell = collapse_epoch(df_raw)
    
    df_train = pd.DataFrame({
    'animal': df_cell['animal'],
    'block': df_cell['block'],
    'clust_idx_v2': df_cell['clust_idx_v2'],
    'region': df_cell['region'],
    'state': df_cell['state'],
    'celltype': df_cell['celltype'],
    'cell': df_cell['cell']
    })
    df_test = pd.DataFrame({
        'animal': df_cell['animal'],
        'block': df_cell['block'],
        'clust_idx_v2': df_cell['clust_idx_v2'],
        'region': df_cell['region'],
        'state': df_cell['state'],
        'celltype': df_cell['celltype'],
        'cell': df_cell['cell']
    })
    df_train['isis'] = df_cell['isis'].apply(get_dataset, n = r)
    df_test['isis'] = df_cell['isis'].apply(get_dataset, n = 1-r, order = 'last')

    df_train = df_train.explode('isis').reset_index()
    df_test = df_test.explode('isis').reset_index()
    
    return df_train, df_test


# chop epochs so that each containing n spikes
def sub_epoch(isi, n = 200, shuffle = False):
    if len(isi) <= n:
        return isi
    else:
        if shuffle == True:
            np.random.shuffle(isi)
        num_sub = round(len(isi)/n)

        subs = []
        while num_sub>0:
            if len(isi) <= n:
                subs.append(isi)
            else:
                subs.append(get_first_n_elements(isi, n))
                isi = isi[n:]
            num_sub -= 1
        return subs

# determine if the variable is a list or a list of list
# True: list of list; False: simple list
def list_check(v):
    if any(isinstance(i, np.ndarray) for i in v):
        return True
    else:
        return False  

def custom_explode(data, colname):
    t = data[colname].apply(list_check)
    ind = []
    ind2 = []
    for i in range(len(t)):
        if t[i] == True:
            ind.append(i)
        else:
            ind2.append(i)
    mask = data.iloc[ind]
    unexplode = data.iloc[ind2]
    explode = mask.explode(colname)
    new_data = pd.concat([unexplode, explode]).reset_index(drop=True)
    return new_data

def xgb_by_cell(df_train, num_states=3, sampling_type='over', **kwargs):
    total_cell = df_train['cell'].unique()
    models = []
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_train.groupby(df_train.cell)
    for cells in tqdm(total_cell):
        train_spec = grouped.get_group(cells)
        X_train = train_spec[spike_matrix]
        y_train = train_spec['state']
        # perform random oversampling on training dataset to balance the number of sample between NREM, REM, and WAKE
        # if the training dataset does not have all states, discard it
        
        if (train_spec['state'].nunique() == num_states):
            sample_weight=[1]*num_states
            add_kwargs = {}
            try:
                if sampling_type=='over': 
                    oversample = over_sampling.RandomOverSampler()
                    X_train, y_train = oversample.fit_resample(X_train, y_train)
                elif sampling_type== 'smote':
                    sm = SMOTE()
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                elif sampling_type== 'under':
                    undersample = RandomUnderSampler()
                    X_train, y_train = undersample.fit_resample(X_train, y_train)
                elif sampling_type== 'balanced':
                    add_kwargs['sample_weight'] = compute_sample_weight(class_weight='balanced', y=y_train)
                else:
                    pass
                my_model = XGBClassifier(n_jobs=3, **kwargs)
                #my_model.set_params(base_score=0.5)
                xgb_model = my_model.fit(X_train, y_train, **add_kwargs)

                models.append({
                    'model': xgb_model,
                    'cell': cells
                })
            except Exception as e:
                print("sad, this failed", e)
    return models

# fit logistic regression for each cell, return a list of models
def logistic_by_cell(df_train, num_states=3, penalty_type = 'l2', sampling_type='over'):
    total_cell = df_train['cell'].unique()
    models = []
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_train.groupby(df_train.cell)
    for cells in tqdm(total_cell):
        train = grouped.get_group(cells)
        X_train = train[spike_matrix]
        y_train = train['state']

        # perform random oversampling on training dataset to balance the number of sample between NREM, REM, and WAKE
        # if the training dataset does not have all states, discard it
        
        if (train['state'].nunique() == num_states):
            add_kwargs = {}
            try:
                if sampling_type =='over': 
                    oversample = over_sampling.RandomOverSampler()
                    X_train, y_train = oversample.fit_resample(X_train, y_train)
                elif sampling_type==  'smote':
                    sm = SMOTE()
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                elif sampling_type==  'under':
                    undersample = RandomUnderSampler()
                    X_train, y_train = undersample.fit_resample(X_train, y_train)
                elif sampling_type==  'balanced':
                    add_kwargs['class_weight'] = 'balanced'
                else:
                    pass
                if penalty_type == 'l2':
                    clf = LogisticRegression(random_state=0, max_iter = 500, **add_kwargs).fit(X_train, y_train)
                elif penalty_type == 'l1':
                    clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter = 500, 
                                             **add_kwargs).fit(X_train, y_train)
                models.append({
                    'model': clf,
                    'cell': cells
                })
            except Exception as e:
                print("Sad,", e)
    return models

def logistic_by_grouping(df_train, num_states=3, penalty_type = 'l2', sampling_type='over', **grouping_vars):
    cond_arr = get_extended_condensed_data(df_train)
    groupings, group_labels, _, _, _, _ = get_cell_groupings(cond_arr, **grouping_vars)
    total_cell = df_train['cell'].unique()
    models = []
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    name_ind = -1
    for group in tqdm(groupings):
        name_ind+=1
        train = df_train[df_train.cell.isin(group)]
        cells = group
        #print(len(train))
        X_train = train[spike_matrix]
        y_train = train['state']

        # perform random oversampling on training dataset to balance the number of sample between NREM, REM, and WAKE
        # if the training dataset does not have all states, discard it
        
        if (train['state'].nunique() == num_states):
            add_kwargs = {}
            try:
                if sampling_type== 'over': 
                    oversample = over_sampling.RandomOverSampler()
                    X_train, y_train = oversample.fit_resample(X_train, y_train)
                elif sampling_type== 'smote':
                    sm = SMOTE()
                    X_train, y_train = sm.fit_resample(X_train, y_train)
                elif sampling_type==  'under':
                    undersample = RandomUnderSampler()
                    X_train, y_train = undersample.fit_resample(X_train, y_train)
                elif sampling_type== 'balanced':
                    add_kwargs['class_weight'] = 'balanced'
                else:
                    pass
                if penalty_type == 'l2':
                    clf = LogisticRegression(random_state=0, max_iter = 500, **add_kwargs).fit(X_train, y_train)
                elif penalty_type == 'l1':
                    clf = LogisticRegression(penalty='l1', solver='liblinear', max_iter = 500, 
                                             **add_kwargs).fit(X_train, y_train)
                models.append({
                    'model': clf,
                    'cell': cells,
                    'name':group_labels[name_ind]
                })
            except Exception as e:
                print("Sad,", e)
    return models

def mlp_by_cell(df_train, num_states=3, sampling_type='smote', **kwargs):
    total_cell = df_train['cell'].unique()
    models = []
    grouped = df_train.groupby(df_train.cell)
    for cells in tqdm(total_cell):
        train_spec = grouped.get_group(cells)
        item = mlp_individual(cells, train_spec, num_states=num_states, sampling_type=sampling_type,
                             filtered=True, **kwargs)
        if len(item):
            models.append(item)
        else: 
            if sampling_type=='smote':
                print("try with oversampling instead")
                item = mlp_individual(cells, train_spec, num_states=num_states, sampling_type='over',
                             filtered=True, **kwargs)
            else:
                print("sad, this failed")
    return models

def mlp_by_animal(df_train, num_states=3, sampling_type='smote', **kwargs):
    cond_arr = get_extended_condensed_data(df_train)
    groupings, group_labels, _, _, _, _ = get_cell_groupings(cond_arr, by_animal=True)
    models = []
    name_ind = -1
    for group in tqdm(groupings):
        name_ind+=1
        print("Looking at ", group_labels[name_ind])
        train = df_train[df_train.cell.isin(group)]
        item = mlp_individual(group, train, num_states=num_states, sampling_type=sampling_type,
                             filtered=True, **kwargs)
        if not len(item):
            print('sad this failed')
        item['name'] = group_labels[name_ind]
        models.append(item)
    return models

def mlp_individual(cell, df_train, num_states=3, sampling_type='smote', filtered=False, **kwargs):
    item = {}
    if not len(kwargs):
        kwargs = {'hidden_layer_sizes':50,
                    'alpha':5E-4,
                    'beta_1':0.8,
                    'learning_rate_init':1E-3,
                    'batch_size':60,
                    'max_iter':2000,
                    'random_state':42}
    if not filtered:
        train_spec = df_train[df_train['cell']==cell]
    else: train_spec = df_train
    arrs = []
    for indiv_cell in train_spec.cell.unique():
        with threadpool_limits(limits=4):
            this_isi = train_spec[train_spec['cell']==indiv_cell].isi_distrib
            this_isi = list(this_isi)
            this_isi = np.array(this_isi)
            arrs.append(this_isi)
    X_train = np.concatenate(arrs)
    y_train = train_spec['state']
    #print("Trying resampling")
    if (train_spec['state'].nunique() == num_states):
        try:
            with threadpool_limits(limits=4):
                match sampling_type:
                    case 'over': 
                        oversample = over_sampling.RandomOverSampler()
                        X_train, y_train = oversample.fit_resample(X_train, y_train)
                    case 'smote':
                        sm = SMOTE()
                        X_train, y_train = sm.fit_resample(X_train, y_train)
                    case 'under':
                        undersample = RandomUnderSampler()
                        X_train, y_train = undersample.fit_resample(X_train, y_train)
            my_model = AlternateMLPClassifier(**kwargs)
            #print("trying training")
            with threadpool_limits(limits=4):
                model = my_model.fit(X_train, y_train)
            item['model'] = model
            item['cell'] = cell
            item['training_length'] = len(y_train)
        except Exception as e:
            print("sad, this failed", e, flush=True)
    else:
        print("Hey this is tragic, this neuron doesn't have all three states, won't work :(")
    return item

def combined_cm(wake_sleep_models, n_rem_models, df_test):
    cm = {}
    use_nrem = pd.DataFrame(n_rem_models)
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_test.groupby(df_test.cell)
    for item in tqdm(wake_sleep_models):
        try:
            model = item['model']
            cell = item['cell']
            test = grouped.get_group(cell)
            X_test = test[spike_matrix]
            y_test = test['state']
            predictions = model.predict(X_test)
            nrem_mod = use_nrem[use_nrem['cell']==cell]
            if len(nrem_mod):
                if len(nrem_mod)>1: print("Ahh too many models")
                n_rem_predictions = nrem_mod['model'].iloc[0].predict(X_test)
            else: 
                print("Hey, no matching nrem model sorry")
                continue
            true_predictions = []
            for ind, prediction in enumerate(predictions):
                if prediction==1:
                    true_predictions.append(2)
                else:
                    true_predictions.append(int(n_rem_predictions[ind]))
            #get confusion matrix
            cm[cell] = (metrics.confusion_matrix(y_test, true_predictions,
                                              normalize = 'true'))
        except Exception as e:
            print(f"Model failed: {e}")
    return cm

def combined_cm_nrem(nrem_other_models, wake_rem_models, df_test):
    cm = {}
    use_rem = pd.DataFrame(wake_rem_models)
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_test.groupby(df_test.cell)
    for item in tqdm(nrem_other_models):
        try:
            model = item['model']
            cell = item['cell']
            test = grouped.get_group(cell)
            X_test = test[spike_matrix]
            y_test = test['state']
            my_label = y_test.unique()
            predictions = model.predict(X_test)
            rem_mod = use_rem[use_rem['cell']==cell]
            if len(rem_mod):
                if len(rem_mod)>1: print("Ahh too many models")
                rem_predictions = rem_mod['model'].iloc[0].predict(X_test)
            else: 
                print("Hey, no matching rem model sorry")
                continue
            true_predictions = []
            for ind, prediction in enumerate(predictions):
                if prediction==0:
                    true_predictions.append(0)
                else:
                    true_predictions.append(int(rem_predictions[ind]))
            #get confusion matrix
            cm[cell] = (metrics.confusion_matrix(y_test, true_predictions,
                                              normalize = 'true'))
        except Exception as e:
            print(f"Model failed: {e}")
    return cm

def get_cm(models, df_test, get_spiking=False, use_spike_matrix=True, grouping_vars={}, include_timing=True):
    cm = {}
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    grouped = df_test.groupby(df_test.cell)
    co = get_extended_condensed_data(df_test)
    alternate_groupings, group_labels,_,_,_,_ = get_cell_groupings(co, **grouping_vars)
    for item in tqdm(models):
        try:
            model = item['model']
            cell = item['cell']
            if type(cell)==str:
                test = grouped.get_group(cell)
            else:
                spec_group = alternate_groupings[group_labels.index(item['name'])]
                test = df_test[df_test.cell.isin(spec_group)]
                cell = item['name']
            if use_spike_matrix: X_test = test[spike_matrix]
            else: 
                X_test = test['isi_distrib']
                X_test = list(X_test)
                X_test = np.array(X_test)
            y_test = test['state']
            #get confusion matrix
            predictions = model.predict(X_test)
            confidence = model.predict_proba(X_test)
            if type(predictions[0])==str and type(y_test.iloc[0])!=str:
                predictions = [2 if x=='wake' else (1 if x=='rem' else 0) for x in predictions]
            spec_cm = metrics.confusion_matrix(y_test, predictions,
                                              normalize = 'true')
            if get_spiking: 
                out=(spec_cm, y_test, predictions, test['spike_num'], test['state_spikes'], confidence)
                if include_timing: out = (*out, test['state_start_end'])
            else: out = spec_cm
            cm[cell] = out
        except Exception as e:
            print(f"Model failed: {e}")
    return cm

def get_individual_cm(item, df_test, get_spiking=False, use_spike_matrix=True):
    cm = {}
    spike_matrix = ['ifr', 'mean_isis', 'median_isis', 'max_isis', 'min_isis', 'sd', 'cv', 'cv2', 'lv', 'lvr']
    try:
        model = item['model']
        cell = item['cell']
        test = df_test[df_test['cell']==cell]
        if use_spike_matrix: X_test = test[spike_matrix]
        else: 
            X_test = test['isi_distrib']
            X_test = list(X_test)
            X_test = np.array(X_test)
        y_test = test['state']
        #get confusion matrix
        predictions = model.predict(X_test)
        confidence = model.predict_proba(X_test)
        if type(predictions[0])==str and type(y_test.iloc[0])!=str:
            predictions = [2 if x=='wake' else (1 if x=='rem' else 0) for x in predictions]
        spec_cm = metrics.confusion_matrix(y_test, predictions,
                                          normalize = 'true')
        if get_spiking: out=(spec_cm, y_test, predictions, test['spike_num'], test['state_spikes'], confidence)
        else: out = spec_cm
        cm[cell] = out
    except Exception as e:
        print(f"Model failed: {e}")
    return cm

def remove_irregular_cm(spec_cm,  norm_size=3):
    """Take the provided cms and remove any of them that aren't norm_size x norm_size"""
    bad_cms = []
    swap=False
    for i, arr in spec_cm.items():
        if len(arr)>=4: 
            swap=True
            break
        if len(arr) != norm_size: 
            bad_cms.append(i)
    if not swap:
        for bad_cm in bad_cms[::-1]:
            spec_cm.pop(bad_cm)
        return spec_cm, len(bad_cms)
    use = {}
    out_bad = 0
    for key, arr in spec_cm.items():
        arr = arr[0]
        if len(arr)==norm_size:
            use[key] = arr
        else: out_bad+=1
    return use, out_bad

def get_extended_condensed_data(dataset, from_scaled_data=True):
    """Collapse the dataset into 3 rows/neuron only, with details per neuron per sleep state, including the number
    of isi chunks total per sleep state for the neuron, which is pretty much all the general data that is needed
    for performance visualizations when combined with cms"""
    if from_scaled_data:
        output = dataset.copy()[['state', 'region', 'cell', 'animal', 'celltype']].drop_duplicates(['state', 'cell']).reset_index()
        #Create a condensed cell dataframe to look at by condition
        this = output['cell'].apply(lambda x: x.split("_"))
        for i, row in enumerate(this):
            if len(row)==8: 
                this.iloc[i] = row[:-2] + ['app_ps1']
        for i, row in enumerate(this):
            if len(row)!=7: print(row)
        output['geno'] = [x[-1] for x in this]
        output['cell_no'] = [x[-2] for x in this]
        output['age'] = [x[1] for x in this]
        output['age_group'] = output['age'].apply(lambda x: 'young' if int(x)<200 else 'old')
        output['state_str'] = [str(x) for x in output['state']]
        output['cell_state'] = output['cell'] + '-' + output['state_str']
        groupings = dataset.groupby(['cell', 'state'])
        sizes = groupings.size()
        def get_state_lengths(cell_state):
            spl = cell_state.split('-')
            return sizes[spl[0], int(spl[1])]
        output['state_chunks'] = output['cell_state'].apply(get_state_lengths)
    else:
        output = dataset.copy()[['geno','age','brain_region', 'cell', 'animal', 'cell_type']].reset_index()
        output['age_group'] = output['age'].apply(lambda x: 'young' if int(x)<200 else 'old')
    return output

def get_sleep_state_conversion(dataset, conv={1:0, 2:1}):
    """Don't use this, just replace the values with the .replace pd function lol"""
    sleep_wake_train = dataset.copy()
    for key in conv:
        sleep_wake_train.loc[sleep_wake_train['state']==key, 'state'] = conv[key]
    return sleep_wake_train

def get_nrem_rem(dataset):
    """Just use the .replace function instead :("""
    return pd.concat([dataset[(dataset['state'] ==0)], dataset[(dataset['state']==1)]]).reset_index()

def get_nrem_wake(dataset):
    """Just use the .replace function instead :("""
    return pd.concat([dataset[(dataset['state'] ==0)], dataset[(dataset['state']==2)]]).reset_index()

def get_rem_wake(dataset):
    """Just use the .replace function instead :("""
    return pd.concat([dataset[(dataset['state'] ==1)], dataset[(dataset['state']==2)]]).reset_index()

def get_cell_groupings(train_df, by_region=False, by_age=False, by_geno=False, by_animal=False, by_ntype=False,
                      subset = {}, exclude = {}, exclude_ind = set(), final_filters=[], these=None):
    """Get a specific split of cells split up by the conditions you specify (region, age, genotype, etc), 
    with subsets/exclusion/exclude_ind working to
    leave out some of the options of the thing you generate, and final filters being a list of filters applied to
    the dataset before it is graphed. 'These' can specify certain groupings to look at by index, as the opposite of exclude_ind"""
    if by_animal:
        animals =  train_df['animal'].unique()
        by_geno=False
        by_age=False
    else: animals = ['all']
    if by_region:
        regions = train_df['region'].unique()
    else: regions = ["total"]
    if by_age: ages = ['young', 'old']
    else: ages =[""]
    if by_geno: genos = train_df['geno'].unique()
    else: genos = ["all"]

    if by_ntype:
        ntypes = ['RSU', 'FS']
    else: ntypes = ['']
    output_cells = []
    output_labels = []
    out_genos,out_ages, out_regions=[],[],[]
    out_nts = []
    i = 0
    for animal in animals:
        if 'animal' in subset and animal != subset['animal']: continue
        if 'animal' in exclude and animal in exclude['animal']: continue
        alla_spot = train_df
        if by_animal: 
            alla_spot = alla_spot[alla_spot['animal']==animal]
            y_old = alla_spot.iloc[0].age_group
            gen = alla_spot.iloc[0].geno
        for geno in genos:
            if by_animal:
                if 'geno' in subset and gen != subset['geno']: continue
                if 'geno' in exclude and gen in exclude['geno']: continue
            else:
                if 'geno' in subset and geno != subset['geno']: continue
                if 'geno' in exclude and geno in exclude['geno']: continue
            this_spot = alla_spot
            if by_geno: this_spot = this_spot[this_spot['geno']==geno]
            for age in ages:
                if by_animal:
                    if 'age' in subset and y_old != subset['age']: continue
                    if 'age' in exclude and y_old in exclude['age']: continue
                else:
                    if 'age' in subset and age != subset['age']: continue
                    if 'age' in exclude and age in exclude['age']: continue
                that_spot = this_spot
                if by_age: 
                    that_spot = that_spot[that_spot['age_group']==age]
                for region in regions:
                    if 'region' in subset and region != subset['region']: continue
                    if 'region' in exclude and region in exclude['region']: continue
                    this_use = that_spot
                    if by_region: this_use = this_use[this_use['region']==region]
                    for nt in ntypes:
                        if 'ntype' in subset and nt != subset['ntype']: continue
                        if 'ntype' in exclude and nt in exclude['ntype']: continue
                        if i in exclude_ind: 
                            i+=1
                            continue
                        i+=1
                        use = this_use
                        if by_ntype: use = use[use['celltype']==nt]
                        for f_filter in final_filters:
                            use = f_filter(use)
                        use = set(use['cell'].unique())
                        animal_lab = (f'{geno}{"" if not by_age else (f"-{age}")}{nt}' if not by_animal else f"{animal}({gen},{y_old}){nt}")
                        spec_reg_lab = f"{animal_lab} in {region}"
                        output_cells.append(use)
                        output_labels.append(spec_reg_lab)
                        out_genos.append((gen if by_animal else geno))
                        out_ages.append((age if not by_animal else y_old))
                        out_regions.append(region)
                        out_nts.append(nt)
    if these:
        output_cells = output_cells[these[0]:these[1]]
        output_labels = output_labels[these[0]:these[1]]
        out_genos = out_genos[these[0]:these[1]]
        out_ages = out_ages[these[0]:these[1]]
        out_regions = out_regions[these[0]:these[1]]
        out_nts = out_nts[these[0]:these[1]]
    return output_cells, output_labels, out_genos, out_ages, out_regions, out_nts

def look_at_states_per_grouping(train_df, by_region=False, by_age=False, by_geno=False, by_animal=False,by_ntype=False,
                      subset = {}, exclude = {}, exclude_ind = set(), final_filters=[], just_counts=False):
    """Count up a bunch of statistics for each of the groups specified by your splits/filters yada yada. If
    just counts is False, the output includes all the individual state_chunk counts in rough order rather than
    just the summed stuff."""
    groupings, labels, genos, ages, regions, nts = get_cell_groupings(train_df, by_region, by_age, 
                                           by_geno, by_animal,by_ntype,subset, exclude, exclude_ind, final_filters)
    train_df.index = train_df.cell
    group_counts = {}
    group_chunks = {}
    group_neurons = {}
    group_prop = {}
    for ind, group in enumerate(groupings):
        state_counts = [0,0,0]
        state_lens = [[],[],[]]
        proportions = [[],[],[]]
        this_group_neurons = {}
        for cell in group:
            this = train_df.loc[cell]
            state_layout = [0,0,0]
            for i,neuron in this.iterrows():
                state_counts[neuron['state']] += neuron['state_chunks']
                if not just_counts: 
                    state_lens[neuron['state']].append(neuron['state_chunks'])
                    state_layout[neuron['state']] = neuron['state_chunks']
            if not just_counts:
                this_group_neurons[cell] = state_layout[:]
            for j in range(3):
                proportions[j] = state_counts[j]/sum(state_counts)
        group_neurons[labels[ind]] = this_group_neurons
        group_counts[labels[ind]] = state_counts
        group_prop[labels[ind]] = proportions
        group_chunks[labels[ind]] = state_lens
    return group_counts, group_prop, group_chunks, group_neurons

def get_state_grouping_dataframe(train_df, grouping_vars={}, just_counts=False):
    df = pd.DataFrame(look_at_states_per_grouping(train_df, just_counts=just_counts, **grouping_vars))
    df = df.transpose()
    df = df.rename(columns={0:'counts', 1:'proportion', 2:'indiv_instances', 3:'by_neuron'})
    df['name'] = list(df.index)
    df['age'] = df.name.apply(lambda x: 'old' if 'old' in x else 'young')
    df['geno'] = df.name.apply(lambda x: 'app_ps1' if 'app_ps1' in x else 'wt')
    return df
    
def filter_dataset(dataset, filters=[]):
    """Filter the dataset with the given list of functions"""
    for filt in filters:
        dataset = filt(dataset)
    dataset.reset_index(inplace=True, drop=True)
    return dataset

def add_epoch_length(dataset):
    """Get the rough hour timing from the file name and convert it to seconds"""
    def get_length(epoch):
        first_num = ""
        i=5
        while epoch[i]!='_':
            first_num += epoch[i]
            i+=1
        i+=1
        second_num = epoch[i:]
        return int(second_num) - int(first_num)

    dataset['length_s'] = (dataset['epoch'].apply(get_length))*60*60
    return dataset

def add_firing_rate(df_app_dataset):
    """Add in the firing rate for the given neurons"""
    def get_num_spikes(row):
        return sum([len(i) for i in row])

    wake_spikes = (df_app_dataset['wake_isi'].apply(get_num_spikes))
    rem_spikes = (df_app_dataset['rem_isi'].apply(get_num_spikes))
    nrem_spikes = (df_app_dataset['nrem_isi'].apply(get_num_spikes))
    all_lengths = list(df_app_dataset['length_s'])
    df_app_dataset['firing_rate'] = [(wake_spikes[i]+rem_spikes[i]+nrem_spikes[i])/all_lengths[i] for i in range(len(wake_spikes))]
    return df_app_dataset

def split_train_test_epochwise(df_app_dataset, test_thresholds=(0.8, 1), filtering_threshold=5,
                              splitting_number=500, exclude_ragged=True, validation_set=False,
                              grouping=None):
    """For each neuron and sleep state layout, take out the specified portion of states defined by test_thresholds
    and then split both the remaining test data and that training data into chunks of ISIs that are *splitting_number long.
    Leave exclude_ragged True because it prevents oversampling of edge case situations.
    If you are trying to group by anything, """
    new_ext_train = {'animal': [], 'epoch': [], 'age': [], 'sex': [], 'geno': [], 'sleep_state': [], 'isi': [],
           'brain_region': [], 'cell_type': [], 'peak_latency': [], 'cell_no': [], 'spike_num':[], 'state_length':[],
                    'state_spikes':[], 'state_start_end':[]}
    new_ext_test = {'animal': [], 'epoch': [], 'age': [], 'sex': [], 'geno': [], 'sleep_state': [], 'isi': [], 
               'brain_region': [], 'cell_type': [], 'peak_latency': [], 'cell_no': [], 'spike_num':[], 'state_length':[],
                   'state_spikes':[],'state_start_end':[]}
    if validation_set:
        new_ext_val = {'animal': [], 'epoch': [], 'age': [], 'sex': [], 'geno': [], 'sleep_state': [], 'isi': [], 
               'brain_region': [], 'cell_type': [], 'peak_latency': [], 'cell_no': [], 'spike_num':[], 'state_length':[],
                   'state_spikes':[],'state_start_end':[]}
    one_to_one = ['animal', 'epoch', 'age', 'sex', 'geno', 'brain_region', 'cell_type', 'peak_latency', 'cell_no']
    conds = ['wake_isi', 'nrem_isi', 'rem_isi']
    cond_lens = ['wake_lens', 'nrem_lens', 'rem_lens']
    cond_timings =['wake_timings', 'nrem_timings', 'rem_timings']
    goes_to = ['wake', 'nrem', 'rem']
    test_start_prop, test_end_prop = test_thresholds[:2]
    if grouping is not None:
        #print("Here, grouping")
        grouped_data = get_extended_condensed_data(df_app_dataset, from_scaled_data=False)
        groupings,labels,_,_,_,_ = get_cell_groupings(grouped_data, **grouping)
        groupings = [sorted(list(group)) for group in groupings]
    if validation_set: val_start_prop, val_end_prop = test_thresholds[2:]
    for row in tqdm(df_app_dataset.index):
        current_info = df_app_dataset.loc[row]
        if grouping is not None:
            for group in groupings:
                if current_info.cell in group:
                    test_threshold = round(len(group)*test_start_prop)
                    test_end = round(len(group)*test_end_prop)
                    group_ind = group.index(current_info.cell)
                    spot_to_place = new_ext_train if (group_ind < test_threshold or group_ind > test_end) else new_ext_test        
        for i, cond in enumerate(conds):
            working_epochs = list(range(len(current_info[cond]))) 
            for ind, epoch in enumerate(current_info[cond]):
                if len(epoch) < filtering_threshold:
                    working_epochs.remove(ind)
            if grouping is None:
                test_threshold = round(len(working_epochs)*test_start_prop)
                test_end = round(len(working_epochs)*test_end_prop)
            if validation_set:
                val_thresh = round(len(working_epochs)*val_start_prop)
                val_end = round(len(working_epochs)*val_end_prop)
            for spec_pos, ind in enumerate(working_epochs):
                if grouping is None:
                    spot_to_place = new_ext_train if (spec_pos < test_threshold or spec_pos > test_end) else new_ext_test
                    if validation_set:
                        if spot_to_place==new_ext_train:
                            spot_to_place = new_ext_train if (spec_pos < val_thresh or spec_pos > val_end) else new_ext_val
                epoch = current_info[cond][ind]
                total_epoch_len = len(epoch)
                #print(f"Total len: {total_epoch_len}, --> {max(total_epoch_len//500, 1)}")
                for specific_index, block in enumerate(range(0, len(epoch), splitting_number)):
                    if block>0 and exclude_ragged and (block + splitting_number >= len(epoch)): continue
                    spot_to_place['sleep_state'].append(goes_to[i])
                    spot_to_place['state_length'].append(current_info[cond_lens[i]][ind])
                    #The original dataset records timings every 100 isis, in seconds, so these
                    #following lines account for that, and grab the appropriate timings for this isi block
                    off = int(splitting_number//100)
                    start_spot = specific_index*off
                    end_spot = min(start_spot + off, len(current_info[cond_timings[i]][ind])-1)
                    #print(start_spot, end_spot)
                    start_end = [current_info[cond_timings[i]][ind][start_spot], current_info[cond_timings[i]][ind][end_spot]]
                    spot_to_place['state_start_end'].append(start_end)
                    if (block + splitting_number >= len(epoch)):
                        spot_to_place['isi'].append(epoch[block:])
                    else:
                        spot_to_place['isi'].append(epoch[block:block+splitting_number])
                    spot_to_place['spike_num'].append(len(spot_to_place['isi'][-1]))
                    spot_to_place['state_spikes'].append(total_epoch_len)
                    for name in one_to_one:
                        spot_to_place[name].append(current_info[name])
    new_ext_train = pd.DataFrame(new_ext_train)
    new_ext_test = pd.DataFrame(new_ext_test)
    if validation_set:
        new_ext_val = pd.DataFrame(new_ext_val)
        return [new_ext_train, new_ext_test, new_ext_val]
    return [new_ext_train, new_ext_test]

def get_total_time(dataset, splitting_num):
    cum_sum=0
    ind=0
    #Mapped total layout of states
    state_rows = dataset[dataset['cell_no']==0]
    while ind < len(state_rows):
        spec_length = state_rows.state_length.iloc[ind]
        cum_sum += int(spec_length)
        chunk_count = int(max(1, state_rows.state_spikes.iloc[ind]//splitting_num))
        ind += chunk_count
    return cum_sum

def get_state_minimums(dataset, metric='proportion'):
    state_layouts = get_file_state_breakdown(dataset)
    print(state_layouts.proportion)
    overall_mins = {'time':[np.nan, np.nan, np.nan], 'count':[np.nan, np.nan, np.nan], 'proportion':[np.nan,
                                        np.nan, np.nan], }
    for column in ['time', 'count', 'proportion', ]:
        for j in range(3):
            rel_list = [x[j] for x in state_layouts[column]]
            overall_mins[column][j] = np.nanmin(rel_list)
    mins = overall_mins[metric]
    return mins

def split_train_chunkwise_timewise(df_app_dataset, broken_down_df, starting_from=0, filtering_threshold=5,
                              splitting_number=500, metric = 'proportion'):
    output = pd.DataFrame()
    mins = get_state_minimums(df_app_dataset, metric=metric)

    for animal in tqdm(broken_down_df.animal.unique()):
        for file in broken_down_df[broken_down_df.animal==animal].epoch.unique():
            total_file_len = get_total_time(broken_down_df[(broken_down_df.animal==animal)&(
                    broken_down_df.epoch==file)], splitting_number)
            this_file = pd.DataFrame()
            for state in range(3):
                spec_rows = broken_down_df[(broken_down_df.animal==animal)&(
                    broken_down_df.epoch==file)&(broken_down_df.state==state)]
                #Add one to ensure no poor penalties to the minimum fella through wack ties
                relevant_prop_limit = mins[state] * (total_file_len+1)
                for cell_no in spec_rows.cell_no.unique():
                    state_cum_sum = [0]
                    chunk_cum_sum = [0]
                    chunk_sizes = [0]
                    ind=0
                    #Mapped total layout of states
                    state_rows = spec_rows[spec_rows['cell_no']==cell_no]
                    while ind < len(state_rows):
                        spec_length = state_rows.state_length.iloc[ind]
                        state_cum_sum.append(int(state_cum_sum[-1]+ spec_length))
                        chunk_count = int(max(1, state_rows.state_spikes.iloc[ind]//splitting_number))
                        ind += chunk_count
                        chunk_cum_sum.append(ind)
                        chunk_sizes.append(chunk_count)
                    point = starting_from*state_cum_sum[-1]
                    left_barrier = np.searchsorted(state_cum_sum, point, side='left')
                    right_barrier = np.searchsorted(state_cum_sum, point+relevant_prop_limit, side='right')
    #                 #See how far over the left is
    #                 offsetl = state_cum_sum[left_barrier] - point
    #                 if right_barrier >= len(state_cum_sum):
    #                     offsetr = 0
    #                 else: offsetr = state_cum_sum[right_barrier] - point - relevant_prop_limit
    #                 testtraindiscrepancy = offsetr-offsetl
                    #Positive discrepancy = test covers more time than it should
                    chunk_cum_sum.append(chunk_cum_sum[-1]+1)
                    this_file = pd.concat([this_file, state_rows[chunk_cum_sum[left_barrier]:chunk_cum_sum[right_barrier]]])
            output = pd.concat([output, this_file])
    return output

def split_train_chunkwise_same_isi(df_app_dataset, broken_down_df, starting_from=0, filtering_threshold=5,
                              splitting_number=500):
    mins = get_state_minimums(df_app_dataset, metric='proportion')
    state_layouts = get_file_state_breakdown(dataset)
    print(mins, state_layouts)
    output = pd.DataFrame()
    for animal in tqdm(broken_down_df.animal.unique()):
        for file in broken_down_df[broken_down_df.animal==animal].epoch.unique():
            this_file = pd.DataFrame()
            for state in range(3):
                spec_rows = broken_down_df[(broken_down_df.animal==animal)&(
                    broken_down_df.epoch==file)&(broken_down_df.state==state)]
                for cell_no in spec_rows.cell_no.unique():
                    state_rows = spec_rows[spec_rows['cell_no']==cell_no]
                    relevant_prop_limit = mins[state] * (len(state_rows)+1) / (
                        state_layouts.proportion.loc[f'{animal}{file}'][state])
                    relevant_prop_limit = int(round(max(1, relevant_prop_limit)))
                    this_file = pd.concat((this_file, state_rows[:relevant_prop_limit]))
            output = pd.concat([output, this_file])
    return output

def split_train_chunkwise_same_isi_set(df_app_dataset, broken_down_df, starting_from=0, filtering_threshold=5,
                              splitting_number=500, down_lims = [25, 10, 40]):
    state_layouts = get_file_state_breakdown(df_app_dataset)
    output = pd.DataFrame()
    for animal in tqdm(broken_down_df.animal.unique()):
        for file in broken_down_df[broken_down_df.animal==animal].epoch.unique():
            this_file = pd.DataFrame()
            spec_rows = broken_down_df[(broken_down_df.animal==animal)&(
                    broken_down_df.epoch==file)]
            for cell_no in spec_rows.cell_no.unique():
                these_rows = spec_rows[spec_rows['cell_no']==cell_no]
                lens = []
                for state in range(3):
                    lens.append(len(these_rows[these_rows['state']==state]))
                if lens[0] < down_lims[0]: break
                elif lens[1] < down_lims[1]: break
                elif lens[2] < down_lims[2]: break
                for state in range(3):
                    state_rows = these_rows[these_rows['state']==state]
                    total_states = len(state_rows)
                    begin = int(starting_from*total_states)
                    end = begin+down_lims[state]
                    if end>total_states:
                        use_rows = pd.concat((state_rows[begin:], state_rows[:(end-total_states)]))
                    else: use_rows = state_rows[begin:end]
                    this_file = pd.concat((this_file, use_rows))
            output = pd.concat([output, this_file])
    return output

def resample_isis_and_add_spike_metrics(dataset, num_resample=500):
    dataset = resample_isis(dataset, num_resample=num_resample)
    print("Spike metrics beginning!")
    get_spike_matrix(dataset, f'sample_isi_{num_resample}')
    return dataset

def resample_isis(dataset, num_resample=500):
    dataset[f'sample_isi_{num_resample}'] = dataset['isi'].apply(sample_n_elements, n = num_resample)
    return dataset

def add_in_cols(dataset, rename=True):
    dataset['str_age'] = dataset['age'].apply(lambda x: str(round(x)))
    dataset['cell'] = dataset['animal'] + '_' + dataset['str_age'] + '_'+ dataset['epoch'] + '_' + dataset['cell_no'].astype('str') + '_' + dataset['geno']
    if rename:
        dataset = dataset.rename(columns={"sleep_state": "state", "brain_region": "region", "cell_type" : "celltype"})
        le = LabelEncoder()
        dataset['state'] = le.fit_transform(dataset['state'])
    return dataset

def scale_data(train, test):
    train_scaled, scalers = scaling_x(train)
    test_scaled = []
    for sc in tqdm(scalers):
        cells = sc['cell']
        test_cell = test.loc[test['cell'] == cells]
        try:
            test_scaled.append(scaling_y(test_cell, sc['scaler']))
        except Exception as e:
            print(test_cell, sc['scaler'], sc, e)
    test_scaled = pd.concat(test_scaled).reset_index(drop=True)
    return train_scaled, test_scaled

def train_multi_models(train_data, methods=['LogReg'], sampling_types=['over'], num_states=3, **kwargs):
    models = {}
    mapping = {'LogReg':logistic_by_cell, 'XGBoost':xgb_by_cell, 'LogRegGroup':logistic_by_grouping}
    for method in methods:
        for typ in sampling_types:
            try:
                models[f'{method}_{typ}'] = mapping[method](train_data, sampling_type=typ, num_states=num_states, **kwargs)
            except Exception as e:
                print("Failure", e)
                if typ=='smote':
                    print("Replacing smote with oversampling because smote failed")
                    models[f'{method}_{typ}'] = mapping[method](train_data, sampling_type='over', num_states=num_states, **kwargs)
    return models

def get_cms_multi(test_data, models, get_spiking=False, grouping_vars={}, include_timing=True):
    output_cms = {}
    for lab, model_set in models.items():
        output_cms[lab] = get_cm(model_set, test_data, get_spiking=get_spiking,grouping_vars=grouping_vars,
                                include_timing=include_timing)
    return output_cms

def load_all_isi_cms():
    test_vals = [5000, 3500, 2000, 1500, 1000, 750, 500, 350, 200, 150,100,75]
    cms = {}
    for val in test_vals:
        cms[val] = load_cm_coef(f'LogRegOver-{val}')
    return cms
def get_dataframe_isi_performance(cms):
    reses = {}
    for val, cm_master in cms.items():
        res = {'state':[], 'length':[], 'accuracy':[], 'constituents':[]}
        for cell, cm_tuple in cm_master['LogReg_over'].items():
            cm, real, predicted, indiv_spike_num, overall_spike_num = cm_tuple
            i = 0
            while i< len(real):
                group = overall_spike_num.iloc[i]
                start_ind=i
                constituents = max(1, group//val)
                #print(constituents)
                i+=constituents
                end_ind = i
                correct = 0
                #print(group, start_ind, end_ind)
                for ind in range(start_ind, end_ind):
                    if ind >= len(real): 
                        #print('hey')
                        continue
                    if real.iloc[ind] == predicted[ind]: correct+=1
                res['state'].append(real.iloc[start_ind])
                res['length'].append(group)
                res['accuracy'].append(correct/(end_ind-start_ind))
                res['constituents'].append(constituents)
        res = pd.DataFrame(res)
        res.index = res.length
        reses[val] = res
    return reses

def verify_regions(array):
    array.region = array.region.replace("ACC", "ACad")
    array.region = array.region.replace("RSC", "RSPv")
    return array

def transform_states_verify(arrs):
    le = LabelEncoder()
    for i in range(len(arrs)):
        arrs[i]['state'] = le.fit_transform(arrs[i]['state'])
        arrs[i] = verify_regions(arrs[i])
    return arrs

def pare_down_training(dataset, train_arr, split_thres, addon, splitting_number, starting_from, downlims=None):
    if addon=='downtime':
        train_maybe = split_train_chunkwise_timewise(dataset, train_arr, starting_from=starting_from, splitting_number=splitting_number)
    elif addon=='downisitime':
        train_maybe = split_train_chunkwise_same_isi(dataset, train_arr,splitting_number=splitting_number,
                                                starting_from=starting_from)
    elif 'downisiset' in addon:
        if not downlims:
            print("Hey this isn't working, include down limits for this kind of isi restriction")
            return
        train_maybe = split_train_chunkwise_same_isi_set(dataset, train_arr, down_lims = downlims,splitting_number=splitting_number,
                                                    starting_from=starting_from)
    else:
        print("Not filtering the dataset/downsampling isis, assuming this is intentional!")
        train_maybe = train_arr
    return train_maybe

def obtain_train_test_scaled_data(split_thres, addon, load_data=True, downlims=[],
                                 generate_if_failed=True, overwrite=False, save_data_after=True,
                                 starting_from=0.0, grouping=None, early_train_exit=False, splitting_number=500):

    if addon=='downisiset':
        addon = f'{addon}{str(downlims)}{str(starting_from)[2]}'
    elif addon in ['downisitime', 'downtime']:
        addon = addon + str(starting_from)[2]
    if load_data and grouping is None:
        res = load_scaled_data(split_thres, addon=addon)
        if len(res):
            train_scaled, test_scaled = transform_states_verify(res)
            print("Loaded successfully")
            return train_scaled, test_scaled
        else:
#             print("Trying to load unfiltered instead")
#             res = load_scaled_data(split_thres, addon="unfiltered")
#             if len(res):
#                 train_scaled, test_scaled = transform_states_verify(res)
#                 pare_down_training(dataset, arrs[0], split_thres, addon, splitting_number, starting_from, downlims)
            print('Not saved, tough luck')
    if generate_if_failed:
        res = generate_train_test_dataset(split_thres, addon, downlims=downlims, save_data_after=save_data_after,
                                          overwrite=overwrite, starting_from=starting_from, grouping=grouping,
                                         early_train_exit=early_train_exit, splitting_number=splitting_number)
        #Data is not directly returned since loading the pickled version defragments the data in some way--> makes model training
        #almost 4x faster for logistic regression.
        if save_data_after and grouping is None:
            return obtain_train_test_scaled_data(split_thres, addon, True, downlims, generate_if_failed, overwrite, save_data_after,
                                            grouping=grouping)
        return res
def generate_train_test_dataset(split_thres, addon, downlims=None, filt=[lambda x: x[x['firing_rate'] > 0.1]],
                               save_data_after=True, overwrite=False, early_train_exit=False, splitting_number=500,
                               validation_set=False, starting_from=0, grouping=None,
                               base_path='/media/HlabShare/Toby_work/sleep_indiv_neuron/filtered_full_animal_isi.pkl'):
    print("Loading Dataset")
    dataset = load_dataset(path=base_path)
    dataset = clean_dataset(dataset, filters=filt)
    dataset = add_in_cols(dataset, rename=False)
    print(f"Splitting into train/test {splitting_number} isi chunks")
    arrs = split_train_test_epochwise(dataset, split_thres, splitting_number=splitting_number, validation_set=validation_set,
                                     grouping=grouping)
    print("Add in some ids")
    for i in range(len(arrs)):
        arrs[i] = add_in_cols(arrs[i])
    print("minimizing stuff in the train dataset")
    train_maybe = pare_down_training(dataset, arrs[0], split_thres, addon, splitting_number, starting_from, downlims)
    if early_train_exit: return [train_maybe, *arrs[1:]]
    print("Resampling and doing spike metrics")
    train_maybe = resample_isis_and_add_spike_metrics(train_maybe, splitting_number)
    test = resample_isis_and_add_spike_metrics(arrs[1], splitting_number)
    print("Scale data")
    if grouping is None:
        train_scaled, test_scaled = scale_data(train_maybe, test)
    else:
        train_scaled, _ = scaling_x(train_maybe)
        test_scaled, _ = scaling_x(test)
    print("Fixing regions")
    train_scaled = verify_regions(train_scaled)
    test_scaled = verify_regions(test_scaled)
    if save_data_after:
        print("Save that data")
        dump_scaled_data(train_scaled, test_scaled, split_thres, addon=addon, overwrite=overwrite)
    return train_scaled, test_scaled

def clean_dataset(dataset, filters=[lambda x: x[x['firing_rate'] > 0.1]]):
    dataset = dataset.reset_index()
    print("adding column if necessary")
    if 'length_s' not in dataset:
        dataset = add_epoch_length(dataset)
    if 'firing_rate' not in dataset:
        dataset = add_firing_rate(dataset)
    print("Filtering out lame neurons")
    dataset = filter_dataset(dataset, filters=filters)
    return dataset

def accuracy_breakdown(graphic_train, cms, **kwargs):
    groupings,labels,_,_,_,_ = get_cell_groupings(graphic_train, **kwargs)
    overall_arr = []
    right_v_wrong = []
    for group in groupings:
        corr_acc = []
        wrong_acc_big = []
        right_reals = []
        wrong_reals = []
        wrong_wrongs = []
        wrong_ones = []
        wrong_one_acc = []
        wrong_acc_correct= []
        these_cells = {}
        for cell in group:
            if cell in cms:
                these_cells[cell] = cms[cell]
            else:
                print(f"{cell} don't have a model :(")
        for key, test_cm in these_cells.items():
            predictions = test_cm[2]
            real = test_cm[1]
            #print(predictions, real)
            for i, prediction in enumerate(predictions):
                if prediction==real.iloc[i]:
                    right_reals.append(real.iloc[i])
                    corr_acc.append(test_cm[5][i][prediction])
                    wrong_ones.append(2 if not prediction else prediction-1)
                    wrong_one_acc.append(test_cm[5][i][wrong_ones[-1]])
                else: 
                    wrong_reals.append(real.iloc[i])
                    wrong_wrongs.append(prediction)
                    wrong_acc_big.append(test_cm[5][i][prediction])
                    wrong_acc_correct.append(test_cm[5][i][real.iloc[i]])
        overall_arr.append([corr_acc, wrong_one_acc, wrong_acc_big, wrong_acc_correct])
        right_v_wrong.append([right_reals, wrong_ones, wrong_reals, wrong_wrongs])
    return overall_arr, labels, right_v_wrong

def order_by_conf_wrong(acc_break, right_v_wrong):
    ind_tracking = {}
    seen = set()
    duplicate = 0
    look = pd.DataFrame()
    for ind, x in enumerate(acc_break[2]):
        ind_tracking[x] = ind
    sorted_wrong = sorted(list(set(acc_break[2])))
    matching_side = []
    matching_side_wrong = []
    state_layout = []
    wrong_state_layout = []
    for x in sorted_wrong:
        matching_side.append(acc_break[3][ind_tracking[x]])
        matching_side_wrong.append(1 - (acc_break[3][ind_tracking[x]]+x))
        state_layout.append(right_v_wrong[2][ind_tracking[x]])
        wrong_state_layout.append(right_v_wrong[3][ind_tracking[x]])
    look['right_conf'] = matching_side
    look['wrong_conf'] = sorted_wrong
    look['bystander_conf'] = matching_side_wrong
    look['corr_state'] = state_layout
    look['wrong_guess_state'] = wrong_state_layout
    look['close_ratio'] = (look.right_conf-look.bystander_conf)/look.bystander_conf
    return look

def order_by_conf_right(acc_break, right_v_wrong):
    ind_tracking = {}
    seen = set()
    duplicate = 0
    look = pd.DataFrame()
    dups = []
    for ind, x in enumerate(acc_break[0]):
        if x not in ind_tracking:
            ind_tracking[x] = ind
        else: dups.append(x)
    sorted_right = sorted(list(set(acc_break[0])))
    #print(f"ignored {len(acc_break[0])-len(sorted_right)} values, probably duplicate 1s, {dups}")
    matching_side = []
    matching_side_wrong = []
    state_layout = []
    wrong_state_layout = []
    for x in sorted_right:
        matching_side.append(acc_break[1][ind_tracking[x]])
        matching_side_wrong.append(1 - (acc_break[1][ind_tracking[x]]+x))
        state_layout.append(right_v_wrong[0][ind_tracking[x]])
        #print(len(right_v_wrong[0]))
        wrong_state_layout.append(right_v_wrong[1][ind_tracking[x]])
    look['right_conf'] = sorted_right
    look['wrong_conf'] = matching_side
    look['bystander_conf'] = matching_side_wrong
    look['corr_state'] = state_layout
    look['wrong_guess_state'] = wrong_state_layout
    look['close_ratio'] = (look.right_conf-look.bystander_conf)/look.bystander_conf
    return look

def merge_accuracy_neuron(cms, condensed_dataset, save=False, name='0', num_state=3):
    """Merge together accuracy values and the test dataset to get it laid out by neuron"""
    output = condensed_dataset[condensed_dataset.state==0].drop(columns=['state_str', 'cell_state', 'state'])
    output.rename(columns = {'state_chunks':'state_chunks_nrem'}, inplace=True)
    rem, wake = [],[]
    for i, row in output.iterrows():
        cell = row.cell
        if num_state ==3: 
            w = condensed_dataset[(condensed_dataset.state==2)&(condensed_dataset.cell==cell)]
            if len(w): wa = w.iloc[0].state_chunks
            else:wa =0
            wake.append(wa)
        r = condensed_dataset[(condensed_dataset.state==1)&(condensed_dataset.cell==cell)]
        if len(r): re = r.iloc[0].state_chunks
        else: re = 0
        rem.append(re)
    if num_state==3:
        output['state_chunks_rem'] = rem
        output['state_chunks_wake'] = wake
    else:
        output.rename(columns={'state_chunks_nrem':'state_chunks_sleep'}, inplace=True)
        output['state_chunks_wake'] = rem
    accs = [[],[],[],[]]
    for i, row in output.iterrows():
        cell = row.cell
        if cell in cms:
            use_cm = cms[cell]
            if len(use_cm)>3:
                use_cm=use_cm[0]
            accs[0].append(accuracy(use_cm))
            accs[1].append(use_cm[0][0])
            accs[2].append(use_cm[1][1])
            if num_state==3:
                accs[3].append(use_cm[2][2])
        else:
            for j in range(num_state+1):
                accs[j].append(np.nan)
    output['balanced_acc'] = accs[0]
    if num_state==3:
        output['nrem_acc'] = accs[1]
        output['rem_acc'] = accs[2]
        output['wake_acc'] = accs[3]
    else:
        output['wake_acc'] = accs[2]
        output['sleep_acc'] = accs[1]
    if save:
        with open(f'/media/HlabShare/Toby_work/sleep_indiv_neuron/accuracy_dataframes/{name}', 'wb') as f:
            pickle.dump(output, f)
    return output

def get_sleeptimes(sleepnpy):
    """Get all the of the sleep states into a convenient dataframe from the file"""
    #deal with sleepybois
    sleep_states = np.zeros((2,0))
    sleeps = np.load(sleepnpy)
    timestamps = (np.nonzero(np.diff(sleeps))[0]+1)*4
    time_ind = (timestamps/4)-1
    states = sleeps[time_ind.astype(int)]
    s = np.array(states)
    t = np.stack((timestamps,s))
    sleep_states = np.concatenate((sleep_states,t), axis =1)
    start_time = np.append([0],sleep_states[0,:-1])
    stop_time = sleep_states[0,:]
    sleepstate = sleep_states[1,:]
    sleepdf = pd.DataFrame({'start_time': start_time,'stop_time': stop_time,'sleepstate': sleepstate})
    return(sleepdf)

def get_file_state_breakdown(dataset):
    """From the original dataset's length information, get a breakdown by file of how long the animal
        spent in each state overall, timewise, proportionally, and by number of bouts."""
    tooooo = ['nrem_lens', 'rem_lens', 'wake_lens']
    per_file_counts = {}
    per_file_proportions = {}
    per_file_times = {}
    for i, animal in enumerate(dataset.animal.unique()):
        for file in dataset[dataset.animal==animal].epoch.unique():
            neur = dataset[dataset.epoch==file].iloc[0]
            totals = [0,0,0]
            amounts = [0,0,0]
            for state in range(3):
                totals[state] += sum(neur[tooooo[state]])
                amounts[state]+= len(neur[tooooo[state]])
            label = f'{animal}{file}'
            per_file_counts[label] = totals
            per_file_proportions[label] = [totals[i]/sum(totals) for i in range(3)]
            per_file_times[label] = amounts
    look = pd.DataFrame([per_file_counts, per_file_times, per_file_proportions]).transpose()
    look = look.rename(columns={0:'time', 1:'count', 2:'proportion'})
    look['total_time'] = look['time'].apply(sum)
    return look

def predict_full_animal(animal, models, cms, condensed_dataset, interval=0):
    """Given models for every cell in this animal and confusion matrices with timings for predictions,
    align them all! Get a total df for merged performance per sleep state! Interval dictates if 
    sleep states should be split up into intervals of *interval* seconds for alignment purposes, so that you
    can distinguish between the merged predictions in the first 30s of a long wake bout vs the 90th-120th seconds
    of the bout. Bouts shorter than the interval are left alone. An interval of 0 makes no splitting happen"""
    cell_base_dir = '/media/HlabShare/appps1_cortical_communication_samb/appps1/data/'
    sleepnpy_base_dir = '/media/HlabShare/appps1_cortical_communication_samb/appps1/data/'
    cell_dir, sleepnpy_dir = get_dirs(animal, cell_base_dir, sleepnpy_base_dir)
    #print(sleepnpy_dir)
    valid_days = {'caf69':'', 'caf71':'0103','caf73':'0120','caf74':'0125', 'caf75':'0125',
             'caf77':'0202','caf81':'0304','caf82':'0304',
             'zbr106':'0223','zbr113':'0513','zbr114':'0426','zbr117':'','zbr118':'0522',
             'zbr119':'0604','bth90':'0323','bth95':'0611','bth100':'0624'}
    sleep_to_predictions = {1:2, 2:0, 3:1}
    sleep_states_files = {}
    all_dfs = {}
    for file in sleepnpy_dir:
        if valid_days[animal] in file:
            out = {'real':[], 'predictions':[], 'confidences':[], 'length':[], 'start':[], 'end':[],
                  'hits':[]}
            ind = os.path.basename(file)[:-10].lower()
            file_name = ind
            these_sleep_times = get_sleeptimes(file)
            these_sleep_times.sleepstate = these_sleep_times.sleepstate.replace(4, 1)
            these_sleep_times.sleepstate = these_sleep_times.sleepstate.replace(5, 1)
            these_sleep_times =these_sleep_times[these_sleep_times.sleepstate!=0]
            starts = list(these_sleep_times.start_time)
            ends = list(these_sleep_times.stop_time)
            reals = list(these_sleep_times.sleepstate)
            inserts = np.zeros(len(starts))
            if interval:
                for i, start in enumerate(starts):
                    length = ends[i]-start
                    inserts[i] = int((length-1)//interval)
                added=0
                for i, val in enumerate(inserts):
                    if val:
                        ends[i+added] = starts[i+added]+interval
                    while added<val:
                        added+=1
                        to_ind = i+added
                        starts.insert(to_ind,starts[to_ind-1]+interval)
                        ends.insert(to_ind, ends[to_ind-1]+interval)
                        reals.insert(to_ind, reals[to_ind-1])
            cells = condensed_dataset[(condensed_dataset.animal==animal)&(
                #filename matches up
                condensed_dataset.cell.str.contains(("_".join(ind.split("_")[-3:]))))].cell.unique()
            overall_predictions = np.zeros((len(starts), 3))
            confidence_predictions = np.zeros((len(starts), 3))
            timings_hits = np.zeros((len(starts)), dtype=object)
            for i, cell in enumerate(cells):
                spec_model = None
                for model in models:
                    if model['cell']==cell:
                        spec_model = model
                        break
                if not spec_model:
                    print(f'lol {cell} sucks, skipping')
                    continue
                timings = cms[cell][-1]
                predictions = cms[cell][2]
                confidences = cms[cell][5]
                for ind, timing in enumerate(timings):
                    spot = np.searchsorted(starts, timing[0], side='right')
                    overall_predictions[spot-1,predictions[ind]] += 1
                    confidence_predictions[spot-1, predictions[ind]] += confidences[ind][predictions[ind]]
                    if isinstance(timings_hits[spot-1], int):
                        timings_hits[spot-1] = [timing]
                    else: 
                        timings_hits[spot-1].append(timing)
            for index in range(len(overall_predictions)):
                out['real'].append(sleep_to_predictions[reals[index]])
                out['start'].append(starts[index])
                out['end'].append(ends[index])
                out['predictions'].append(overall_predictions[index])
                out['confidences'].append(confidence_predictions[index])
                out['length'].append(ends[index] - starts[index])
                out['hits'].append(timings_hits[index])
            out = pd.DataFrame(out)
            all_dfs[file_name] = out
    #print(all_dfs[file_name][all_dfs[file_name].length > 60])
    return all_dfs

def get_all_animal_accuracy(models, cms, graphic_train,
                            min_predictions_to_include=10, time_correction_by_file=True, time_norm_animal=True,
                           interval=0):
    """Return a combined dataframe containing all the sleep bouts for every animal and merged prediction accuracy stats, as
    well as the confusion matrices for each animal's merged prediction performance. Min_predictions_to_include determines the number
    of *combined* neuron predictions within a single sleep bout is required to be included. Should be >0. Time_correction_by_file
    effectively "smooshes" all files from an animal together so the time stamps are next to each other between files (for easier graphing),
    and time_norm_animal sets all the timing values as starting from 0 at the beginning of the first recording. Not required, just nice."""
    combined = pd.DataFrame()
    anim_cms = {}
    for animal in tqdm(graphic_train.animal.unique()):
        all_dfs = predict_full_animal(animal, models, cms, graphic_train,
                                     interval=interval)
        animal_df = pd.DataFrame()
        prev_max=0
        for key, df in all_dfs.items():
            #print(key)
            pred_b = []
            pred_c = []
            total_pred = []
            for i, row in df.iterrows():
                pred_b.append(list(row['predictions']).index(max(row['predictions'])))
                pred_c.append(list(row['confidences']).index(max(row['confidences'])))
                total_pred.append(sum(row['predictions']))
            df['pred_b'] = pred_b
            df['pred_c'] = pred_c
            df['pred_num'] = df.predictions.apply(sum)
            avg_conf = []
            for i, row in df.iterrows():
                avg_conf.append([0 if not row.predictions[ind] else row.confidences[ind]/row.predictions[ind] for ind in range(3)])
            df['avg_conf']=avg_conf
            df['file'] = key
            df = df[df.pred_num>min_predictions_to_include].reset_index()
            if(len(df)):
                if prev_max and time_correction_by_file:
                    m = min(df.start)
                    df['start'] += prev_max - m
                    df['end'] += prev_max - m
                    df.hits = df['hits'].apply(lambda x: [[j[0]+prev_max-m, j[1]+prev_max-m] for j in x])
                prev_max = max(df['end'])
                animal_df = pd.concat([animal_df, df])
        animal_df = animal_df.reset_index()
        animal_df['animal'] = animal
        animal_df['geno'] = graphic_train[graphic_train.animal==animal].iloc[0].geno
        animal_df['age'] = graphic_train[graphic_train.animal==animal].iloc[0].age_group
        animal_df['cell'] = animal_df.animal + animal_df.index.astype(str)
        if time_norm_animal:
            m = min(animal_df.start)
            animal_df['end'] -= m
            animal_df['start'] -= m
            animal_df.hits = animal_df['hits'].apply(lambda x: [[j[0]-m, j[1]-m] for j in x])
        combined = pd.concat([animal_df, combined])
        anim_cms[animal] = metrics.confusion_matrix(animal_df.real, animal_df.pred_b, normalize="true")
    return combined, anim_cms