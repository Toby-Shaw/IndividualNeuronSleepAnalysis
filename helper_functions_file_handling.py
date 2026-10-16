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
from imblearn.over_sampling import SMOTE

import time
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def extract_start_datetime(path):
    match = re.search(r'H_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_', path)
    if match:
        start_datetime_str = match.group(1)
        return datetime.strptime(start_datetime_str, '%Y-%m-%d_%H-%M-%S')
    return None

def get_spike_start_time(filepath):
    fname = os.path.basename(filepath)
    # This regex specifically looks for the first timestamp after the 'H_' prefix
    match = re.search(r'H_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})', fname)
    
    if match:
        try:
            return pd.to_datetime(match.group(1), format='%Y-%m-%d_%H-%M-%S')
        except Exception:
            # Fallback if the underscore format confuses the parser
            return pd.to_datetime(match.group(1).replace('_', ' '), fuzzy=True)
    return None

def dump_animal_data(animal_name, s_params, actual_vars=[], folder_path=None, overwrite=True):
    if not folder_path:
        folder_path = f"/media/HlabShare/Toby_work/{animal_name}_neuronal_data"
    try:
        os.mkdir(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Directory already exists, presumably overwriting")
        if not overwrite: 
            print("To avoid overwrite, exiting")
            return 0

    saving_names = ['swd2s', 'end_lengths', 'successful_paths', 'neuron_types']
    #Try to make the subdirectories
    try:
        for name in saving_names:
            os.mkdir(folder_path + f"/{name}")
    except FileExistsError:
        print(f"Directory already exists, presumably overwriting")
        if not overwrite: 
            logger.info("To avoid overwrite, exiting")
            return 0
    assert len(actual_vars), 'Provide the data!!!'
    for i in range(len(saving_names)):
        filename = f"{folder_path}/{saving_names[i]}/bin_{s_params['d2_binsz']}_o_{s_params['d2_order']}" + \
                f"_t{s_params['threshold']}"
        with open(filename, 'wb') as f:
            pickle.dump(actual_vars[i], f)
    
def load_animal_data(animal_name, s_params, folder_path=None):
    if not folder_path:
        folder_path = f"/media/HlabShare/Toby_work/{animal_name}_neuronal_data"
    saving_names = ['swd2s', 'end_lengths', 'successful_paths', 'neuron_types']
    actual_vars = {}
    try:
        for i in range(len(saving_names)):
            filename = f"{folder_path}/{saving_names[i]}/bin_{s_params['d2_binsz']}_o_{s_params['d2_order']}" + \
                f"_t{s_params['threshold']}"
            with open(filename, 'rb') as f:
                actual_vars[saving_names[i]] = pickle.load(f)
    except Exception as e:
        logger.info("Hey that data isn't saved, tough luck. It'll either try to generate it now or crash")
        logger.info(e)
    return actual_vars

def dump_scaled_data(train, test, thresholds, addon='epoch', folder_path=None, overwrite=False):
    """Store some scaled data in the specified area, naming it based on the threshold and the addon var"""
    if not folder_path:
        name = f'{str(thresholds[0])[2]}{str(thresholds[1])[2]}{addon}'
        folder_path = f"/media/HlabShare/Toby_work/sleep_indiv_neuron/scaled_sets/{name}"
    try:
        os.mkdir(folder_path)
        print(f"Directory '{folder_path}' created successfully.")
    except FileExistsError:
        print(f"Directory already exists, presumably overwriting")
        if not overwrite: 
            print("To avoid overwrite, exiting")
            return 0

    saving_names = ['test_scaled', 'train_scaled']
    saving = [test, train]
    for i, name in enumerate(saving_names):
        filename = f"{folder_path}/{name}"
        with open(filename, 'wb') as f:
            pickle.dump(saving[i], f)
            
def load_scaled_data(thresholds, addon='epoch', folder_path=None):
    """Load a train and test scaled dataset that has already been generated and stored before"""
    if not folder_path:
        name = f'{str(thresholds[0])[2]}{str(thresholds[1])[2]}{addon}'
        folder_path = f"/media/HlabShare/Toby_work/sleep_indiv_neuron/scaled_sets/{name}"
    loading_names = ['train_scaled', 'test_scaled']
    loading = []
    for name in loading_names:
        filename = f"{folder_path}/{name}"
        try:
            with open(filename, 'rb') as f:
                loading.append(pickle.load(f))
        except Exception as e:
            logger.info("Hey that data isn't saved, tough luck")
    return loading

def load_all_scaled_data(thresholds):
    files = glob.glob("/media/HlabShare/Toby_work/sleep_indiv_neuron/scaled_sets/*")
    output = {}
    for file in files:
        base = os.path.basename(file)
        with open(f'{file}/train_scaled', 'rb') as f:
            output[base] = [pickle.load(f)]
        with open(f'{file}/test_scaled', 'rb') as f:
            output[base].append(pickle.load(f))
    return output

def load_all_acc_dataframes(thresholds):
    files = glob.glob("/media/HlabShare/Toby_work/sleep_indiv_neuron/accuracy_dataframes/*")
    output = {}
    for file in files:
        base = os.path.basename(file)
        with open(file, 'rb') as f:
            output[base] = pd.read_pickle(f)
    return output

def load_dataset(path='/media/HlabShare/Toby_work/sleep_indiv_neuron/filtered_full_animal_isi.pkl', filters=[]):
    #'/media/HlabShare/Ben_work/isi/all_animal_isi_df.pkl'
    dataset = pd.read_pickle(path)
    return dataset

def save_cm_coef(models, cm, label, do_coef=False):
    begin = "/media/HlabShare/Toby_work/sleep_indiv_neuron/isi_cms"
    with open(f'{begin}/{label}.pkl', 'wb') as file:
        pickle.dump(cm, file)
    if label[0]!='X' and do_coef:
        with open(f'{begin}/coefs/{label}.pkl', 'wb') as file:
            pickle.dump(get_coef(models), file)
    return None

def load_cm_coef(label, do_coef=False):
    begin = "/media/HlabShare/Toby_work/sleep_indiv_neuron/isi_cms"
    output = []
    with open(f'{begin}/{label}.pkl', 'rb') as file:
        output.append(pickle.load(file))
    if label[0]!='X' and do_coef:
        with open(f'{begin}/coefs/{label}.pkl', 'rb') as file:
            output.append(pickle.load(file))
    if len(output)==1: return output[0]
    else: return output
    
# function to get cell_dir and sleepnpy_dir based on animal name
def get_dirs(animal_name, cell_base_dir, sleepnpy_base_dir):
    character_code = animal_name[:3]
    character_code = character_code.upper()
    number_code = animal_name[-3:]
    # add zeros in front of number_code to make the length of 5
    number_code = number_code.zfill(5)
    # '/media/HlabShare/appps1_cortical_communication_samb/appps1/data/ZBR117/0529/0_12/ZBR117_0529_0_12_singleunits.npy'
    cell_dir = natsorted(glob.glob(f'{cell_base_dir}{animal_name.lower()}/*/*_*/*_singleunits.npy'))
    sleepnpy_dir = natsorted(glob.glob(f'{sleepnpy_base_dir}{animal_name.upper()}/*/*_*/*_sleep.npy'))
    return cell_dir, sleepnpy_dir

def save_safe(cms):
    with open(f"/media/HlabShare/Toby_work/sleep_indiv_neuron/mlp_cms/baseline.pkl", "wb") as f:
        pickle.dump(cms, f)
    time.sleep(1)
    with open(f"/media/HlabShare/Toby_work/sleep_indiv_neuron/mlp_cms/backup_baseline.pkl", "wb") as f:
        pickle.dump(cms, f)
        
def load_safe():
    try:
        with open(f"/media/HlabShare/Toby_work/sleep_indiv_neuron/mlp_cms/baseline.pkl", "rb") as f:
            cms = pickle.load(f)
        print("Successfully loaded the main cm dictionary")
        return cms
    except Exception as e:
        print("Initial load failed->",e)
        print("Trying backup")
        try:
            with open(f"/media/HlabShare/Toby_work/sleep_indiv_neuron/mlp_cms/backup_baseline.pkl", "rb") as f:
                cms = pickle.load(f)
            print("secondary worked alright")
            #Overwrite bugged first file with correct second
            save_safe(cms)
            return cms
        except:
            print("Damn both suck, I guess you are screwed->", e)
            return {}