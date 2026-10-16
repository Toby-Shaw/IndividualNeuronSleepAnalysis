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
from helper_functions_math_addons import *
from helper_functions_file_handling import *
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_avalanches(data, perc=0.25, ncells=-1):

    '''
    Function that goes through an array of binned spikes and determines
        the avalanch boundaries and properties

    parameters:
        data - array of spike times. one row for each neuron
        perc - threshold for defining an avalanche,
            if network is dominated by silent periods you can use 0

    returns:
    Result - a dictionary with 2 inputs.
        'S' is the size of each avalanche (number of spikes above threshold)
        'T' is the duration (number of time bins avalanche spanned)
    '''

    # To time the function
    ttic = time.time()

    # num cells, num bins
    if ncells == -1:
        n, m = np.shape(data)
    else:
        n = ncells
        m = np.shape(data)[0]
    print("Data has {} neurons with length {}*binsize".format(n, m))

    # tic = time.time()
    if n == 1:
        network = cdc(data)
    else:
        # collapse into single array. sum the amount of activity in each bin.
        if ncells == -1:
            network = np.nansum(data, axis=0)
        else:
            network = data.copy()
        # print("network ", network)
        # print("sh network ", network.shape)

    # ckbn doubts why we need data now?
    # why we dont give network and n(number of cells) instead of data as input?
    data = None
    del data
    # toc = time.time()
    # print("bustbefor000 calc took time {} seconds".format(toc-tic))

    # tic = time.time()
    if perc > 0:
        sortN = np.sort(network)
        perc_threshold = sortN[round(m * perc)]
        print("perc_threshold ", perc_threshold)
        sortN = None
        del sortN
    else:
        perc_threshold = 0
        print("perc_threshold ", perc_threshold)

    # print("network ", network[0:30])
    zdata = cdc(network)
    # toc = time.time()
    # print("bustbefor00 calc took time {} seconds".format(toc-tic))

    # tic = time.time()
    # intervals
    zdata[zdata <= perc_threshold] = 0
    # print("zdata 0 ", zdata[0:30])
    # avalanches
    zdata[zdata > perc_threshold] = 1
    # print("zdata 1 ", zdata[0:30])
    zdata = zdata.astype(np.int8)

    # location of intervals
    zeros_loc_zdata = np.where(zdata == 0)[0]
    # print("zeros_loc_zdata ", zeros_loc_zdata[0:30])
    zeros_to_delete = \
        zeros_loc_zdata[np.where(np.diff(zeros_loc_zdata) == 1)[0]]
    # print("zeros_to_delete ", zeros_to_delete[0:30])
    # toc = time.time()
    # print("bustbefor01 calc took time {} seconds".format(toc-tic))
    zeros_loc_zdata = None
    del zeros_loc_zdata

    # tic = time.time()
    # cuts out irrelevant 0s result=>, series of 1s and a single 0 separation
    # z1data = np.delete(zdata, Z[np.where(np.diff(Z) == 1)[0]])
    # 1 0 0 1 :  12  5  6  9 (if perc_threshold = 7)  =>  1 0  1 : 12  6  9
    # in short in a series of zeros last zero index is not deleted
    z1data = np.delete(zdata, zeros_to_delete)
    # print("zdata d ", zdata[0:30])
    # print("z1data ", z1data[0:30])
    # print("network ava ", network[0:30])
    # avalanches = np.delete(network, Z[np.where(np.diff(Z) == 1)[0]])
    avalanches = np.delete(network, zeros_to_delete)
    # print("avalanches ", avalanches[0:30])
    # use a single 0 to separate network activities in each avalanche
    avalanches[z1data == 0] = 0
    # print("avalanches ", avalanches[0:30])

    # location of the intervals
    zeros_loc_z1data = np.where(z1data == 0)[0]
    # print("zeros_loc_z1data ", zeros_loc_z1data[0:30])
    # toc = time.time()
    # print("bustbefor02 calc took time {} seconds".format(toc-tic))

    # Now calculate S and T based on avalanches and z1data
    burst = []
    # tic = time.time()
    for i in np.arange(0, np.size(zeros_loc_z1data) - 1):
        tmp_av = avalanches[zeros_loc_z1data[i] + 1:zeros_loc_z1data[i + 1]]
        tmp_burst = np.sum(tmp_av) - (perc_threshold * len(tmp_av))
        if tmp_burst <= 0:
            raise RuntimeError('Burst value {}, zero/negative'
                               .format(tmp_burst))
        burst.append(tmp_burst)
    # print("burst ", burst[0:30])
    # toc = time.time()
    # print("bust calc took time {} seconds".format(toc-tic))
    tmp_av = None
    del tmp_av

    # tic = time.time()
    # AVduration
    T = np.diff(zeros_loc_z1data) - 1
    # print("T ", T[0:30])
    # Duration should be positive
    # print("T < 0 ", T[T < 0])
    T = T[T > 0]
    
    z2data = zdata[0:-1]
    z2data = np.insert(z2data,0,0)
    location = np.where(np.logical_and(zdata == 1, z2data == 0))[0]
    location = location[1:-1]

    Result = {
        'S': np.asarray(burst),
        'T': T,
        'loc': location  	
    }

    # toc = time.time()
    # print("bustaft calc took time {} seconds".format(toc-tic))

    ttoc = time.time()
    print("Time took in get_avalanches is {} seconds".format(ttoc-ttic),
          flush=True)

    # # print variable and sizes
    # local_vars = list(locals().items())
    # for var, obj in local_vars:
    #    print("Variable ", var, " size ", sys.getsizeof(obj), " id ", id(obj))

    return Result

def n_getspikes_old(neuron_list, start=False, end=False, lonoff=1):

    '''
    Extracts spiketimes to a list from neuron_list
    Unless otherwise specified start, end are in seconds

    n_getspikes(neuron_list, start=False, end=False)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    lonoff : Apply on off times (default on, 1)

    Returns
    -------
    spiketimes_allcells : List of all spiketimes

    Raises
    ------
    ValueError if neuron list is empty

    Examples
    --------
    n_getspikes(neuron_list, start=False, end=False)

    '''

    logger.info('Extracting spiketimes to a list from neuron_list')
    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.info('start and end is %s and %s', start, end)

    if start < neuron_list[0].start_time:
        raise ValueError('start is less than neuron_list[0].start_time')
    if end > neuron_list[0].end_time:
        raise ValueError('end is greater than neuron_list[0].end_time')

    # Create empty list
    spiketimes_allcells = []

    # Loop through and get spike times
    for idx, neuron_l in enumerate(neuron_list):
        logger.debug('Getting spiketimes for cell %s', str(idx))

        # get spiketimes for each cell and append
        if lonoff:
            spiketimes = neuron_l.spike_time_sec_onoff
        else:
            spiketimes = neuron_l.spike_time_sec
        idx_l = \
            np.where(np.logical_and(spiketimes >= start, spiketimes <= end))[0]
        spiketimes = spiketimes[idx_l]
        spiketimes_allcells.append(spiketimes)

    return spiketimes_allcells

def n_getspikes(neuron_list, start=False, end=False, lonoff=1):

    '''
    Extracts spiketimes to a list from neuron_list
    Unless otherwise specified start, end are in seconds

    n_getspikes(neuron_list, start=False, end=False)

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    lonoff : Apply on off times (default on, 1)

    Returns
    -------
    spiketimes_allcells : List of all spiketimes

    Raises
    ------
    ValueError if neuron list is empty

    Examples
    --------
    n_getspikes(neuron_list, start=False, end=False)

    '''

    logger.info('Extracting spiketimes to a list from neuron_list')
    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')

    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.info('start and end is %s and %s', start, end)

    if start < neuron_list[0].start_time:
        raise ValueError('start is less than neuron_list[0].start_time')
    if end > neuron_list[0].end_time:
        raise ValueError('end is greater than neuron_list[0].end_time')

    # Create empty list
    spiketimes_allcells = []

    # Loop through and get spike times
    for idx, neuron_l in enumerate(neuron_list):
        #logger.debug('Getting spiketimes for cell %s', str(idx))

        # get spiketimes for each cell and append
        if lonoff:
            spiketimes = neuron_l.spike_time_sec_onoff
        else:
            spiketimes = neuron_l.spike_time_sec
        alt_idx_first = np.searchsorted(spiketimes, start)
        alt_idx_last = np.searchsorted(spiketimes, end, side='right')
        spiketimes_allcells.append(spiketimes[alt_idx_first: alt_idx_last])

    return spiketimes_allcells
def n_spiketimes_to_spikewords(neuron_list, binsz=0.02,
                               start=False, end=False,
                               binarize=0, lonoff=1):
    '''
    This function converts spiketimes to spikewords

    Parameters
    ----------
    neuron_list : List of neurons from (usually output from ksout)
    binsz : Bin size (default 0.02 (20 ms))
    start : Start time (default self.start_time)
    end : End time (default self.end_time)
    binarize : Get counts (default 0) in bins,
    if binarize is 1,binarize to 0 and 1
    lonoff : Apply on off times (default on, 1)

    Returns
    -------
    hzcount : count per bins
    spikewords_array : array of spikewords row x column (time bins x cells)

    '''

    # Constants
    conv_mills = 1000.0

    logger.info('Converting spiketime to spikewords')

    # check neuron_list is not empty
    if (len(neuron_list) == 0):
        raise ValueError('Neuron list is empty')
    # binarize is only 0 or 1
    if (binarize not in [0, 1]):
        raise ValueError('Binarize takes only values 0 or 1')

    # Get time
    if start is False:
        start = neuron_list[0].start_time
    if end is False:
        end = neuron_list[0].end_time
    logger.debug('start and end is %s and %s', start, end)

    # Get spiketime list
    spiketimes = n_getspikes(neuron_list, start=start, end=end,
                             lonoff=lonoff)

    # convert time to milli seconds
    start = start * conv_mills
    end = end * conv_mills
    binsz = binsz * conv_mills

    # startime in bins
    binrange = np.arange(start, (end + binsz), binsz)
    n_cells = len(spiketimes)

    # initialize array
    spikewords_array = np.zeros([n_cells, binrange.shape[0]-1])

    # loop over cells and find counts/binarize
    for i in range(n_cells):

        # spiketimes in seconds to ms
        spiketimes_cell = np.asarray(spiketimes[i]) * conv_mills
        counts, bins = np.histogram(spiketimes_cell, bins=binrange)

        # binarize the counts
        if binarize == 1:
            counts[counts > 0] = 1
        spikewords_array[i, :] = counts

    if binarize == 1:
        if (spikewords_array < 128).all():
            return (spikewords_array.astype(np.int8))
        else:
            return (spikewords_array.astype(np.int32))
    elif binarize == 0:
        if (spikewords_array < 128).all():
            return (spikewords_array.astype(np.int8))
        elif (spikewords_array < 2147483647).all():
            return (spikewords_array.astype(np.int32))
        else:
            return (spikewords_array.astype(np.int64))