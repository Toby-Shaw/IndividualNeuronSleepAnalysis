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
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# -------------------------
# --- Helper Functions ---
# -------------------------

colors = ['#8CDADE','#DE908C','#DE8CDA','#8CDE90']
colors_rgb = np.array([(140, 218, 222), (222, 144, 140), (222, 140, 218), (140, 222, 144)], dtype=np.float64)
darker_colors = ['#2EA4A8','#A8322E','#A82EA4','#2EA832']
darker_colors_rgb = np.array([(46, 164, 168), (168, 50, 46), (168, 46, 164), (46, 168, 50)], dtype=np.float64)
lighter_colors = ['#B8E9EA','#EAB9B8','#EAB8E9','#B8EAB9']
lighter_colors_rgb = np.array([(184, 233, 234), (234, 185, 184), (234, 184, 233), (184, 234, 185)], dtype=np.float64)
colors_rgb /= 255
darker_colors_rgb /= 255
lighter_colors_rgb /= 255

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


def get_beta_event_mask(data, sf, low=12, high=30, threshold_factor=6):
    """
    Returns a boolean mask of the same length as data, 
    True where beta power > 6x Median.
    """
    # 1. Bandpass filter 12-30Hz
    nyq = 0.5 * sf
    b, a = butter(3, [low/nyq, high/nyq], btype='band')
    filt_data = filtfilt(b, a, data)
    
    # 2. Get instantaneous power envelope via Hilbert
    amplitude_envelope = np.abs(hilbert(filt_data))
    power_envelope = amplitude_envelope**2
    
    # 3. Calculate Threshold (FOM)
    fom_threshold = threshold_factor * np.median(power_envelope)
    
    # 4. Return boolean mask
    return power_envelope > fom_threshold

def scaling_plots(Result, burst, burstMin, burstMax, alpha, T, tMin, tMax,
                  beta, TT, Sm, sigma, fit_sigma, pltname, saveloc, p_val_b, p_val_t):
    # burst PDF
    burstMax = int(burstMax)
    burstMin = int(burstMin)
    fig1, ax1 = plt.subplots(nrows = 1, ncols = 3, figsize = [10, 6])
    pdf = np.histogram(burst, bins = np.arange(1, np.max(burst) + 2))[0]
    p = pdf / np.sum(pdf)
    ax1[0].plot(np.arange(1, np.max(burst) + 1), p, marker = 'o',
                markersize = 5, fillstyle = 'none', mew = .5,
                linestyle = 'none', color = 'darkorchid', alpha = 0.75)
    ax1[0].plot(np.arange(1, np.max(burst) + 1)[burstMin:burstMax],
                p[burstMin:burstMax], marker = 'o', markersize = 5,
                fillstyle = 'full', linestyle = 'none', color = 'darkorchid',
                alpha = 0.75)
    ax1[0].set_yscale('log')
    ax1[0].set_xscale('log')

    x = np.arange(burstMin, burstMax + 1)
    y = (np.size(np.where(burst == burstMin + 6)[0]) / np.power(burstMin + 6, -alpha)) *\
        np.power(x, -alpha)
    y = y / np.sum(pdf)

    ax1[0].plot(x, y, color = '#c5c9c7')
    ax1[0].set_xlabel('AVsize')
    ax1[0].set_ylabel('PDF(S)')
    ax1[0].set_title('AVsize PDF, ' + str(np.round(alpha, 3)))
    if p_val_b is not None:
        ax1[0].text(10, .1, f'p_val = {p_val_b}')

    # time pdf
    tdf = np.histogram(T, bins = np.arange(1, np.max(T) + 2))[0]
    t = tdf / np.sum(tdf)
    ax1[1].plot(np.arange(1, np.max(T) + 1), t, marker = 'o',
                markersize = 5, fillstyle = 'none', mew = .5,
                linestyle = 'None', color = 'mediumseagreen', alpha = 0.75)
    ax1[1].plot(np.arange(1, np.max(T) + 1)[tMin:tMax], t[tMin:tMax],
                marker = 'o', markersize = 5, fillstyle = 'full',
                linestyle = 'none', color = 'mediumseagreen', alpha = 0.75)
    ax1[1].set_yscale('log')
    ax1[1].set_xscale('log')
    sns.despine()

    x = np.arange(tMin, tMax + 1)
    y = np.size(np.where(T == tMin)) / (np.power(tMin, -beta)) *\
        np.power(x, -beta)
    y = y / np.sum(tdf)
    ax1[1].plot(x, y, color = '#c5c9c7')
    ax1[1].set_xlabel('AVduration')
    ax1[1].set_ylabel('PDF(D)')
    ax1[1].set_title('AVdura PDF, ' + str(np.round(beta, 3)))
    if p_val_t is not None:
        ax1[1].text(10, .1, f'p_val = {p_val_t}')


    # figure out how to plot shuffled data

    # scaling relation
    i = np.where(TT == tMax)  # just getting the last value we use, getting rid of the hard codes
    ax1[2].plot(TT, ((np.power(TT, sigma) / np.power(TT[7], sigma)) * Sm[7]),
                label = 'pre', color = '#4b006e')
    ax1[2].plot(TT, (np.power(TT, fit_sigma[0]) / np.power(TT[7], fit_sigma[0]) * Sm[7]),
                'b', label = 'fit', linestyle = '--', color = '#826d8c')
    ax1[2].plot(TT, Sm, 'o', color = '#fb7d07', markersize = 5, mew = .5,
                fillstyle = 'none', alpha = 0.75)

    locs = np.where(np.logical_and(TT < tMax, TT > tMin))[0]

    ax1[2].plot(TT[locs], Sm[locs], 'o', markersize = 5, mew = .5,
                color = '#fb7d07', fillstyle = 'full', alpha = 1)
    ax1[2].set_xscale('log')
    ax1[2].set_yscale('log')
    ax1[2].set_ylabel('<S>')
    ax1[2].set_xlabel('Duration')
    ax1[2].set_title('Difference = ' + str(np.round(Result['df'], 3)))

    plt.tight_layout()
    plt.legend()
    # plt.savefig(saveloc + "/" + pltname + 'scaling_relations' + '.svg', format='svg')
    savefigpath = op.join(saveloc , pltname + 'scaling_relations' + '.svg')
    print("savefigpath ", savefigpath)
    plt.savefig(savefigpath, format='svg')

    return fig1


def AV_analysis(burst, T, flag = 1, bm = 20, tm = 10, nfactor_bm=0, nfactor_tm=0,
                nfactor_bm_tail=0.8, nfactor_tm_tail=1.0, none_fact=40,
                max_time=7200, verbose = True, exclude = False, exclude_burst = 50, 
                exclude_time = 20, exclude_diff_b=20, exclude_diff_t=10, plot=True, pltname='', saveloc=''):
    #print('VERBOSE: ', verbose)

    if bm is None:
        if verbose:
            print('none_fact ', none_fact)
        bm = int(np.max(burst)/none_fact)

    Result = {}
    burstMax, burstMin, alpha = \
        ex.EXCLUDE(burst[burst < np.power(np.max(burst), nfactor_bm_tail)], bm,
                   nfactor=nfactor_bm, verbose = verbose)
    idx_burst = \
        np.where(np.logical_and(burst <= burstMax, burst >= burstMin))[0]
    # print("burst[idx_burst] ", burst[idx_burst])
    # print("idx_burst ", idx_burst)
    if verbose:
        print("alpha ", alpha)
        print("burst min: ", burstMin)
        print("burst max:", burstMax, flush=True)

    Result['burst'] = burst
    Result['alpha'] = alpha
    Result['xmin'] = burstMin
    Result['xmax'] = burstMax

    Result['P_burst'] = None
    Result['EX_b'] = False
    if exclude:
        if burstMin > exclude_burst or (burstMax-burstMin)<exclude_diff_b:
            print(f'This block excluded for burst: xmin {burstMin} diff: {burstMax-burstMin}')
            Result['EX_b'] = True
    
    if flag == 2 and not Result['EX_b']:
        if verbose:
            print("About to do the p val test for burst")
        # pvalue test
        Result['P_burst'], ks, hax_burst, ptest_bmin = \
            pv.pvaluenew(burst[idx_burst], alpha, burstMin, nfactor=nfactor_bm,
                         max_time=max_time, verbose = verbose)

    if tm is None:
        if verbose:
            print('none_fact ', none_fact)
        tm = int(np.max(T)/none_fact)

    # print("tMax, tMin, beta = ex.EXCLUDE(T, tm)")
    # ckbn tMax, tMin, beta = ex.EXCLUDE(T, tm, nfactor=nfactor_tm)
    tMax, tMin, beta = \
        ex.EXCLUDE(T[T < np.power(np.max(T), nfactor_tm_tail)], tm,
                   nfactor=nfactor_tm, verbose = verbose)
    idx_time = np.where(np.logical_and(T >= tMin, T <= tMax))[0]

    if verbose:
        print("beta ", beta)
        print(f'time min: {tMin}')
        print(f'time max: {tMax}', flush=True)

    Result['T'] = T
    Result['beta'] = beta
    Result['tmin'] = tMin
    Result['tmax'] = tMax

    Result['P_t'] = None
    Result['EX_t'] = False
    if exclude:
        if tMin > exclude_time or (tMax-tMin)<exclude_diff_t:
            print(f'This block excluded for time: tmin {tMin} diff: {tMax-tMin}')
            Result['EX_t'] = True

    if flag == 2 and not Result['EX_t'] and not Result['EX_b']:
        if verbose:
            print("About to do the p val test for time")
        # pvalue for time
        Result['P_t'], ks, hax_time, ptest_tmin = \
            pv.pvaluenew(T[idx_time], beta, tMin, nfactor=nfactor_tm,
                         max_time=max_time, verbose=verbose)

    TT = np.arange(1, np.max(T) + 1)
    Sm = []
    for i in np.arange(0, np.size(TT)):
        Sm.append(np.mean(burst[np.where(T == TT[i])[0]]))
    Sm = np.asarray(Sm)
    Loc = np.where(Sm > 0)[0]
    TT = TT[Loc]
    Sm = Sm[Loc]

    # ckbndnt
    fit_sigma = \
        np.polyfit(np.log(TT[np.where(np.logical_and(TT > tMin,
                                                     TT < tMax))[0]]),
                   np.log(Sm[np.where(np.logical_and(TT > tMin,
                                                     TT < tMax))[0]]), 1)
    sigma = (beta - 1) / (alpha - 1)
    if verbose:
        print("fit_sigma ", fit_sigma)
        print("sigma ", sigma, flush=True)


    Result['pre'] = sigma
    Result['fit'] = fit_sigma
    Result['df'] = np.abs(sigma - fit_sigma[0])
    Result['TT'] = TT
    Result['Sm'] = Sm
    Result['burst_cdf'] = None
    Result['time_cdf'] = None
    Result['scaling_relation_plot'] = None 

    if plot:
        fig1 = scaling_plots(Result, burst, burstMin, burstMax, alpha, T,
                             tMin, tMax, beta, TT, Sm, sigma, fit_sigma,
                             pltname, saveloc, Result['P_burst'], Result['P_t'])
        if flag == 2 and not Result['EX_t'] and not Result['EX_b']:
            hax_burst.axes[0].set_xlabel('Size (S)', fontsize=16)
            hax_burst.axes[0].set_ylabel('Prob(size < S)', fontsize=16)
            # hax_burst.savefig(saveloc + "/" + pltname + 'pvalue_burst' + '.svg', format='svg')
            savefigpathb = op.join(saveloc,  pltname + 'pvalue_burst' + '.svg')
            print("savefigpathb ", savefigpathb)
            hax_burst.savefig(savefigpathb, format='svg')

            hax_time.axes[0].set_xlabel('Duration (D)', fontsize=16)
            hax_time.axes[0].set_ylabel('Prob(size < D)', fontsize=16)
            # hax_time.savefig(saveloc + "/" + pltname + 'pvalue_time' +  '.svg', format='svg')
            savefigpatht = op.join(saveloc, pltname + 'pvalue_time' + '.svg')
            print("savefigpatht ", savefigpatht)
            hax_burst.savefig(savefigpatht, format='svg')
            Result['burst_cdf'] = hax_burst
            Result['time_cdf'] = hax_time
        Result['scaling_relation_plot'] = fig1
        plt.close('all')

    return Result



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

def higuchi_fractal_dimension_flexible(x, k_min=40, k_max=140):
    """
    Calculates HFD within the 40ms - 140ms scaling regime found during diagnostic sweep.
    """
    n = len(x)
    x = np.array(x, dtype=float)
    
    # Baseline for quiet windows
    if np.sum(x) < 10 or np.all(x == x[0]):
        return 1.0
        
    lk = np.empty(k_max)
    for k in range(1, k_max + 1):
        lm = []
        for m in range(k):
            sub_series = x[m::k]
            if len(sub_series) < 2: continue
            ll = np.sum(np.abs(np.diff(sub_series)))
            n_max = (n - 1) // k
            norm = (n - 1) / (n_max * k)
            lm.append((ll * norm) / k)
        lk[k-1] = np.mean(lm) if lm else 1e-10

    ln_lk = np.log(lk)
    ln_k = np.log(1. / np.arange(1, k_max + 1))
    
    # Fit within specified Goldilocks indices
    start_idx = k_min - 1
    end_idx = k_max - 1
    fit_x = ln_k[start_idx:end_idx]
    fit_y = ln_lk[start_idx:end_idx]
    
    mask = np.isfinite(fit_y)
    if np.sum(mask) < 2: return 1.0
    
    slope, _ = np.polyfit(fit_x[mask], fit_y[mask], 1)
    return slope

def sweep_k_ranges_with_timescales(pop_data, max_search_k=300, step=10):
    """
    Seeks the exact k-range where HFD stays within the 'Goldilocks' zone (1.2 - 1.8).
    Returns (k_lower_bound, k_upper_bound).
    """
    lower_k = np.nan
    upper_k = np.nan
    
    # 1. FIND THE LOWER BOUND
    for k in range(10, max_search_k - 10, step):
        val = higuchi_fractal_dimension_flexible(pop_data, k_min=k, k_max=k+10)
        
        if 1.2 < val < 1.8:
            # We found the zone! Now back up by 2s to find the start
            for fine_k in range(k, k - step - 1, -2):
                if fine_k < 2: break
                f_val = higuchi_fractal_dimension_flexible(pop_data, k_min=fine_k, k_max=fine_k+10)
                if f_val <= 1.2:
                    lower_k = fine_k + 2 # The last k that was still in range
                    break
            if np.isnan(lower_k): lower_k = k # Fallback if fine search misses
            break
    
    if np.isnan(lower_k):
        return np.nan, np.nan # Never found the zone

    # 2. FIND THE UPPER BOUND
    # Start from where the lower_k was found and move forward
    for k in range(int(lower_k), max_search_k - 10, step):
        val = higuchi_fractal_dimension_flexible(pop_data, k_min=k, k_max=k+10)
        
        if val >= 1.8:
            # We hit noise! Back up by 2s to find the exit
            for fine_k in range(k, k - step - 1, -2):
                f_val = higuchi_fractal_dimension_flexible(pop_data, k_min=fine_k, k_max=fine_k+10)
                if f_val < 1.8:
                    upper_k = fine_k
                    break
            if np.isnan(upper_k): upper_k = k
            break # Exit immediately once noise is handled
            
    return lower_k, upper_k

# --- HELPER FUNCTIONS FOR D2 ---
def autocov(sig, lags):
    sig = np.asarray(sig)
    mean_sig = np.mean(sig)
    N = len(sig)
    if max(lags) >= N: return np.zeros(len(lags))
    acv = np.array([np.mean(sig[:N - lag_i] * sig[lag_i:]) for lag_i in lags]) - mean_sig ** 2
    return acv

def yule_walker(sig, order):
    lags = np.arange(start=0, stop=order+1, step=1)
    acv = autocov(sig, lags)
    Gamma = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            Gamma[i, j] = acv[np.abs(i-j)]
    try:
        return np.dot(np.linalg.inv(Gamma), np.transpose(acv[1:]))
    except:
        return np.zeros(order)

def g_helper(kernel, w_list):
    kernel_w = np.zeros(np.shape(w_list), dtype=complex)
    for t in range(len(kernel)):
        kernel_w += kernel[t] * np.exp(-1j * w_list * (t + 1))
    return np.abs(1 - kernel_w)**2

def KLr(kernel, kernelc):
    w_list = np.linspace(start=-np.pi, stop=np.pi, num=2000) # Reduced num for speed in loop
    g = g_helper(kernel, w_list)
    gc = g_helper(kernelc, w_list)
    sdC = np.sqrt(np.trapz(1/(2*np.pi)*gc/g, w_list))
    f = 1/(4*np.pi)*(np.log(g*(sdC**2)/gc)+gc/(g*(sdC**2))-1)
    return np.trapz(f, w_list)

def d2_calculation(sig, order):
    """Integrated d2 and d2_KLr calculation"""
    try:
        kernel = yule_walker(sig, order)
        # d2_old calculation
        d2_old = (1 - np.sum(kernel)) / np.sqrt(order)
        
        # d2_new (KLr) calculation
        if order > 1:
            def cost(kernelc_m):
                return KLr(kernel, np.append(kernelc_m, 1 - np.sum(kernelc_m)))
            x0 = (kernel + (np.ones(np.shape(kernel)) / np.sqrt(order)) * (1 - np.sum(kernel)) / np.sqrt(order))[:-1]
            res = minimize(cost, x0, method='BFGS', options={'gtol': 1e-3, 'disp': False})
            d2_new = np.log2(np.e) * res.fun
        else:
            d2_new = np.log2(np.e) * KLr(kernel, np.ones(1))
            
        return float(d2_old), float(d2_new)
    except:
        return np.nan, np.nan

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from datetime import timedelta
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.diagnostic import acorr_ljungbox

# --- 1. THE GRAD STUDENT'S ROBUST D2 FUNCTION ---
def d2_yw(x, binsz, p, diag_lags=20):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)

    if n <= (p + 10):
        return None

    results = {}
    x0 = x - np.mean(x)

    try:
        # mle method is usually the most stable for neural data
        phi, sigma = yule_walker(x0, order=p, method="mle")
        phi = np.asarray(phi, dtype=float)
        
        results["phi"] = phi
        results["sigma"] = sigma

        A = np.zeros((p, p))
        A[0, :] = phi
        if p > 1:
            A[1:, :-1] = np.eye(p - 1)

        eigvals = np.linalg.eigvals(A)
        rho = np.max(np.abs(eigvals))
        results["rho"] = rho

        if (not np.isfinite(rho)) or rho <= 0:
            tau_rho = np.nan
        elif rho >= 1:
            tau_rho = np.inf
        else:
            tau_rho = -binsz / np.log(rho)

        results["tau"] = tau_rho
        d2 = (1.0 - np.sum(phi)) / np.sqrt(p)
        results["d2"] = d2

        Xlag = np.vstack([x0[p-k:-k] for k in range(1, p + 1)]).T
        resid = x0[p:] - Xlag.dot(phi)

        n_eff = len(resid)
        sig2 = np.mean(resid**2)
        k_params = p + 1
        bic = n_eff * (np.log(sig2) + 1.0) + k_params * np.log(n_eff)
        results["bic"] = bic
        
        return results
    except Exception as e:
        # This will help us see if statsmodels is throwing an error
        return {"error": str(e)}


    
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

def check_for_sufficient_neuron_counts(sw_data, threshold):
    bad_inds = []
    for i, neuron in enumerate(sw_data):
        if len(neuron) < threshold:
            #logger.info(f"Tsk, file {i} is poor quality, only {len(neuron)} neuron(s)")
            bad_inds.append(i)
    return bad_inds
        
def load_specific_animal_spiking_data(animal_name, s_params, files_to_try=[], load_data_attempt=True,
                                     save_loaded_data_if_not_present=True, overwrite=False,
                                     absolute_quality_threshold=10, rsu_filter=True):
    #Load spikes, binarize all relevant neurons, store them
    print(f"Loading and binning all the spikes you want to look at")
    initial_time = time.time()
    load_data_success = False
    if load_data_attempt:
        loaded_vars = load_animal_data(animal_name, s_params)
        if len(loaded_vars): 
            load_data_success = True
            sw_d2s, end_lengths, neuron_types = loaded_vars['swd2s'], loaded_vars['end_lengths'], loaded_vars['neuron_types']
            successful_paths = loaded_vars['successful_paths']
            print(f"Loading from calculated bins successfully, this includes data from {len(successful_paths)} files")
            
            assert len(sw_d2s)==len(end_lengths)==len(successful_paths), "Misaligned files, no can do :("

    if not load_data_success:
        logger.info("Grabbing all the spikes from the individual files, this will take ~10 minutes")
        sw_d2s = []
        end_lengths = []
        neuron_types = []
        #Save which paths have sufficient high quality neurons
        successful_paths = []
        assert len(files_to_try), "Need file paths to load from if this hasn't already been generated :("
        for path in files_to_try:
            print(path)
            try:
                fileStart = get_spike_start_time(path)
                cells = np.load(path, allow_pickle=True)
                good_cells = [c for c in cells if hasattr(c, 'quality') and c.quality < 3]
                neuron_type = [neuron.cell_type for neuron in good_cells]
                if not good_cells:
                    logger.info("No good cells :(")
                    continue
                if len(good_cells) < absolute_quality_threshold:
                    logger.info(f"This file has {len(good_cells)} neuron(s), not sufficient, skipping")
                    continue
                successful_paths.append(path)
                # Pull count data (non-binarized)
                end_lengths.append(good_cells[0].end_time)
                sw_d2 = n_spiketimes_to_spikewords(good_cells, binsz=s_params['d2_binsz'])
                sw_d2s.append(sw_d2)
                neuron_types.append(neuron_type)
            except Exception as e:
                print(f"Failed on file {path}: {e}")

        if save_loaded_data_if_not_present: 
            dump_animal_data(animal_name, s_params, [sw_d2s, end_lengths, successful_paths, neuron_types], overwrite=overwrite)
            logger.info('Successfully saved newly loaded data in easy-to-access format')

    final_time = time.time()
    logger.info(f"Took about {final_time-initial_time} seconds to load data from {len(successful_paths)} files")
    logger.info(f"Post-processing time for quality checks and data adjustments")
    initial_time = time.time()
    results = {}
    #Post processing to filter for RSUs if desired, and binarize population level data if specified
    ret_sw_d2s = []
    total_lens = []
    filtered_neuron_types = (neuron_types if not s_params['rsu_filter'] else [])
    if s_params['binarized']: logger.info("Binarizing...")
    for i in range(len(sw_d2s)):
        filter_ind = [x for x in range(len(neuron_types[i])) if neuron_types[i][x]=="RSU"]
        relevant = sw_d2s[i]
        total_lens.append(len(relevant))
        if s_params['rsu_filter']:
            relevant = relevant[filter_ind]
            addon = []
            for ind in filter_ind:
                addon.append(neuron_types[i][ind])
            filtered_neuron_types.append(addon)
        if s_params['binarized']:
            copied = np.zeros(relevant.shape)
            for ind, spec_neuron in enumerate(relevant):
                copied[ind,:] = spec_neuron >= 1
            ret_sw_d2s.append(copied)
        else: 
            ret_sw_d2s.append(relevant)
    pop_rate_d2s = calculate_pop_rate(ret_sw_d2s)
    
    if s_params['rsu_filter']: logger.info(f"Progression of neuron counts thanks to RSU filtering: {[f'{total_lens[i]}->{len(ret_sw_d2s[i])}' for i in range(len(total_lens))]}")
    logger.info("Double checking cleanliness of data and thresholds")
                
    bad_inds = check_for_sufficient_neuron_counts(ret_sw_d2s, absolute_quality_threshold)
    for bad_ind in bad_inds[::-1]:
        ret_sw_d2s.pop(bad_ind)
        end_lengths.pop(bad_ind)
        successful_paths.pop(bad_ind)
        filtered_neuron_types.pop(bad_ind)
        pop_rate_d2s.pop(bad_ind)
                
    logger.info(f"{len(files_to_try) - len(successful_paths)} file(s) had poor data/insufficient neuron counts and were removed")
    labels = ['swd2s', 'popd2s', 'end_lengths', 'successful_paths', 'neuron_types']
    values = [ret_sw_d2s, pop_rate_d2s, end_lengths, successful_paths, filtered_neuron_types]
    for i in range(len(labels)):
        results[labels[i]] = values[i] 
    final_time = time.time()
    logger.info(f"Post-processing took {final_time-initial_time} seconds")
    return results   

def calculate_pop_rate(sw_d2s):
    pop_rate_d2s = []
    for sw_d2 in sw_d2s:
        pop_rate_d2 = np.sum(sw_d2, axis=0)
        pop_rate_d2s.append(pop_rate_d2)
    return pop_rate_d2s

def collect_sleep_state_timings(total_sleep_df, include_mas_as_nrem = True):
    #In format: index/measurement count of point when a state changed --> that state
    #Measurements are 4 seconds apart
    state_starts = {}
    prev_state = None
    for i, state in enumerate(total_sleep_df['sleep_state']):
        if state!=prev_state:
            state_starts[i] = state
        prev_state = state
    if include_mas_as_nrem:
        #Remove microarousals/count them as NREM, since they "interrupt" NREM 1-min intervals for d2. Test keeping them later
        kill_list = []
        for key in state_starts:
            if state_starts[key] == 4.0:
                kill_list.append(key)
        for key in kill_list:
            del state_starts[key]
    #Timestamps for every state change
    sleep_change_timestamps = total_sleep_df['datetime'].iloc[list(state_starts.keys())]
    state_time = [x.dayofyear*60*60*24+x.hour*60*60+x.minute*60+x.second for x in sleep_change_timestamps]
    state_starts_list = list(state_starts.values())
    state_start_end = [[state_time[i], state_time[i+1]] for i in range(len(state_time)-1)]
    state_length = np.diff(state_time)
    return sleep_change_timestamps, state_starts_list, state_start_end, state_length

def collect_spike_data_timings(successful_paths, end_lengths):
    #List of start and end times for spiking data from each file, pairs corresponding to each file
    start_timestamps = []
    end_timestamps = []
    start_times_seconds = []
    end_times_seconds = []
    for i, path in enumerate(successful_paths):
        start = get_spike_start_time(path)
        end = start + pd.to_timedelta(end_lengths[i], 'second')
        start_time_seconds = start.dayofyear*60*60*24+start.hour*60*60+start.minute*60+start.second
        end_time_seconds = start_time_seconds + end_lengths[i]
        start_timestamps.append(start)
        end_timestamps.append(end)
        start_times_seconds.append(start_time_seconds)
        end_times_seconds.append(end_time_seconds)
    return start_timestamps, end_timestamps, start_times_seconds, end_times_seconds

def calculate_criticality_sleep_windows(paths_to_files, window_size, order, pop_rate_d2s, neuron_counts, bin_size,
                         s_e_timestamps, sleep_s_e_timestamps, show_output=False, sw_d2s=None):
    #assert len(sw_d2s)==len(pop_rate_d2s), "Need aligned sums and individual neuron counts"
    initial_time = time.time()
    win_size = window_size
    dfs = []
    # For every file
    for ind, block in enumerate(paths_to_files): 

        #data to be saved[]
        state_length_list = []
        bin_size_list = []
        num_units_list = []
        d2_list = []
        cur_state_list = []
        time_in_cur_state_list = []
        prev_state_list = []
        next_state_list = []
        t_since_last_aw_start_list = []
        t_since_last_aw_end_list = []
        t_since_last_w_start_list = []
        t_since_last_w_end_list = []
        t_since_last_nrem_start_list = []
        t_since_last_nrem_end_list = []
        t_since_last_s_start_list = []
        t_since_last_s_end_list = []
        nonzeros = []
        win_stds = []
        statuses = []
        datetimes = []
        firing_rates = []
        fano_factors = []
        mean_pop_counts = []
        var_pop_counts = []
        n_spikes_totals = []
        n_neurons_list = []

        # filter spikeobject for high quality RSUs
        start_timestamps, end_timestamps, start_times_seconds, end_times_seconds = s_e_timestamps
        state_starts_list, state_start_end, state_length = sleep_s_e_timestamps
        start_time = start_timestamps[ind]
        end_time_of_file = end_timestamps[ind]
        start_time_seconds = start_times_seconds[ind]
        end_time_seconds = end_times_seconds[ind]
        logger.info(f"Looking at file with electrode data from {start_time} : {end_time_of_file}")

        last_aw_idx = None
        last_nrem_idx = None
        last_w_idx = None
        last_s_idx = None

        #Going through each individual state
        for i, start_end in enumerate(state_start_end):
            cur_state = state_starts_list[i]
            state_dur = start_end[1] - start_end[0]
            right_end = start_end[1]
            left_end = start_end[0]
            #If this state is after these neurons recorded
            if start_end[0] > end_time_seconds: 
                continue
            #If this state is fully before these neurons recorded
            elif start_end[1] < start_time_seconds: 
                continue
            #If this state gets cut in the middle by a neuron record end
            if start_end[1] > end_time_seconds:
                state_dur = end_time_seconds - start_end[0]
                right_end = end_time_seconds
            #If this state started before this recording but goes through it
            if start_end[0] < start_time_seconds and start_end[1] > start_time_seconds:
                state_dur = start_end[1] - start_time_seconds
                left_end = start_time_seconds
            num_wins = int(state_dur/win_size)
            if num_wins==0: 
                logger.debug(f"Hit a short state from {start_end[0]-start_time_seconds} to {start_end[1]-start_time_seconds}, skipping")
                continue

            #Slice the total data to get all available electrode data in this specific state
            bin_ind_start = int((left_end-start_time_seconds)/bin_size)
            bin_time_start = start_time + pd.to_timedelta(left_end-start_time_seconds, 'second')
            bin_ind_end = int((right_end-start_time_seconds)/bin_size)

            pop_rate_d2 = pop_rate_d2s[ind][bin_ind_start:bin_ind_end]
            if sw_d2s: sw_d2 = sw_d2s[ind][:,bin_ind_start:bin_ind_end]
            bins_per_window = int(win_size/bin_size)
            logger.debug(f"Looking at state {cur_state} from {start_end[0]-start_time_seconds}-{start_end[1]-start_time_seconds}" +
                 f"\n\tIt has electrode data from {left_end-start_time_seconds}-{(right_end-start_time_seconds)}" +
                  f", with {num_wins} criticality window(s)")
            #Go through the windows and calculate your criticality
            for win in range(num_wins):
                #Index for each criticality window
                start_idx = win*bins_per_window
                end_idx = (win+1)*bins_per_window
                pop_sum = pop_rate_d2[start_idx: end_idx]

                res = d2_yw(pop_sum + 1e-10 * np.random.randn(len(pop_sum)), 
                    binsz=bin_size, p=order)
                if not res: res = {}
                res['d2'] = res.get('d2', np.nan)
                res['tau'] = res.get('tau', np.nan)
                res['rho'] = res.get('rho', np.nan)
                res['bic'] = res.get('bic', np.nan)
                
                d2_list.append(res)
                    
                #Calculate additional info/factors
                duration = win_size
                n_neurons = neuron_counts[ind]
                if sw_d2s: 
                    n_spikes_total = np.sum(sw_d2[:,start_idx:end_idx])
                    firing_rate_hz = n_spikes_total / (n_neurons * duration)
                else:
                    n_spikes_total = np.sum(pop_rate_d2[start_idx: end_idx])
                    firing_rate_hz = n_spikes_total / (n_neurons * duration)
                # Fano factor of population counts across bins
                mean_count = np.mean(pop_rate_d2[start_idx: end_idx])
                var_count = np.var(pop_rate_d2[start_idx: end_idx], ddof=1) if len(pop_rate_d2[start_idx: end_idx]) > 1 else np.nan

                if mean_count > 0:
                    fano_factor = var_count / mean_count
                else:
                    fano_factor = np.nan

                # CV of ISI, pooled across neurons
                #cvs = []

                #for neuron in neurons:
                #    spk = get_spikes_in_window(neuron, start_t, end_t)
                #    isi = np.diff(spk)

                #    if len(isi) >= 2 and np.mean(isi) > 0:
                #        cvs.append(np.std(isi, ddof=1) / np.mean(isi))

                #cv_isi = np.nanmean(cvs) if len(cvs) > 0 else np.nan
                firing_rates.append(firing_rate_hz)
                fano_factors.append(fano_factor)
                mean_pop_counts.append(mean_count)
                var_pop_counts.append(var_count)
                n_spikes_totals.append(n_spikes_total)
                n_neurons_list.append(n_neurons)
                
                
               
                
                nonzeros.append(np.count_nonzero(pop_sum))
                win_stds.append(np.std(pop_sum))
                datetimes.append(bin_time_start + pd.to_timedelta(win*win_size, 'second'))
                status = "Success"
                if res is None:  status = "Window too small"
                elif "error" in res: status = f"Math Error: {res['error']}"
                elif np.isnan(res.get('d2', np.nan)): status = "Result is NaN"
                statuses.append(status)
                #cur state
                cur_state_list.append(cur_state)
                time_in_cur_state = win*win_size
                time_in_cur_state_list.append(time_in_cur_state)

                #prev state
                if i>=1:
                    prev_state_list.append(state_starts_list[i-1])
                else:
                    prev_state_list.append(np.nan)

                if i<len(state_start_end)-1:
                    next_state_list.append(state_starts_list[i+1])
                else:
                    next_state_list.append(np.nan)

                #last aw
                if last_aw_idx is not None:
                    last_aw = state_start_end[last_aw_idx]
                    t_since_last_aw_start = start_end[0]-last_aw[0]+time_in_cur_state
                    t_since_last_aw_end = start_end[0]-last_aw[1]+time_in_cur_state
                    t_since_last_aw_start_list.append(t_since_last_aw_start)
                    t_since_last_aw_end_list.append(t_since_last_aw_end)
                else:
                    t_since_last_aw_start_list.append(np.nan)
                    t_since_last_aw_end_list.append(np.nan)

                #last wake
                if last_w_idx is not None:
                    last_w = state_start_end[last_w_idx]
                    t_since_last_w_start = start_end[0]-last_w[0]+time_in_cur_state
                    t_since_last_w_end = start_end[0]-last_w[1]+time_in_cur_state
                    t_since_last_w_start_list.append(t_since_last_w_start)
                    t_since_last_w_end_list.append(t_since_last_w_end)
                else:
                    t_since_last_w_start_list.append(np.nan)
                    t_since_last_w_end_list.append(np.nan)

                #last nrem
                if last_nrem_idx is not None:
                    last_nrem = state_start_end[last_nrem_idx]
                    t_since_last_nrem_start = start_end[0]-last_nrem[0]+time_in_cur_state
                    t_since_last_nrem_end = start_end[0]-last_nrem[1]+time_in_cur_state
                    t_since_last_nrem_start_list.append(t_since_last_nrem_start)
                    t_since_last_nrem_end_list.append(t_since_last_nrem_end)
                else:
                    t_since_last_nrem_start_list.append(np.nan)
                    t_since_last_nrem_end_list.append(np.nan)

                #last sleep
                if last_s_idx is not None:
                    last_s = state_start_end[last_s_idx]
                    t_since_last_s_start = start_end[0]-last_s[0]+time_in_cur_state
                    t_since_last_s_end = start_end[0]-last_s[1]+time_in_cur_state
                    t_since_last_s_start_list.append(t_since_last_s_start)
                    t_since_last_s_end_list.append(t_since_last_s_end)
                else:
                    t_since_last_s_start_list.append(np.nan)
                    t_since_last_s_end_list.append(np.nan)

                state_length_list.append(state_length[i])
                num_units_list.append(neuron_counts[ind])

            if cur_state==1:
                last_aw_idx = i
            if cur_state==2:
                last_nrem_idx = i
            if cur_state==1 or cur_state==5:
                last_w_idx = i
            if cur_state==2 or cur_state==3:
                last_s_idx = i

        #make pandas df for the block and save

        df = pd.DataFrame()
        df['state'] = cur_state_list
        df['state_length'] = state_length_list
        df['num_units'] = num_units_list

        df['d2'] = [x['d2'] for x in d2_list]
        df['datetime'] = datetimes
        df['tau'] = [x['tau'] for x in d2_list]
        df['rho'] = [x['rho'] for x in d2_list]
        df['bic'] = [x['bic'] for x in d2_list]
        df['firing_rate'] = firing_rates
        df['fano_factor'] = fano_factors
        df['mean_pop_counts'] = mean_pop_counts
        df['var_pop_counts'] = var_pop_counts
        df['n_spikes_total'] = n_spikes_totals
        df['n_neurons'] = n_neurons_list
        df['status'] = statuses
        df['std'] = win_stds
        df['nonzero_bins'] = nonzeros
        df['sleep_state'] = cur_state_list
        df['time_since_cur_state'] = time_in_cur_state_list
        df['prev_state']= prev_state_list
        df['next_state'] = next_state_list
        df['t_since_last_aw_start'] = t_since_last_aw_start_list 
        df['t_since_last_aw_end'] = t_since_last_aw_end_list 
        df['t_since_last_wake_start'] = t_since_last_w_start_list 
        df['t_since_last_wake_end'] = t_since_last_w_end_list 
        df['t_since_last_nrem_start'] = t_since_last_nrem_start_list 
        df['t_since_last_nrem_end'] = t_since_last_nrem_end_list 
        df['t_since_last_sleep_start']= t_since_last_s_start_list 
        df['t_since_last_sleep_end'] = t_since_last_s_end_list
        dfs.append(df)

    final_time = time.time()
    logger.info(f"Took about {final_time-initial_time:.1f} seconds, successfully calculated")
    d2_df = pd.concat(dfs)
    # Check correlations with failure
    if "Result is NaN" in d2_df['status'].values:
        low_activity = d2_df[d2_df['status'] == "Result is NaN"]['nonzero_bins'].mean()
        logger.info(f"Avg nonzero bins in NaN windows: {low_activity:.5f} / {window_size/bin_size}")
    d2_df = d2_df[d2_df['d2']!=np.nan]
    logger.info(f"Failed windows: {len(d2_df[d2_df['status']=='Failure'])}")
    return d2_df

def calculate_criticality(paths_to_files, window_size, order, pop_rate_d2s, neuron_counts, bin_size, s_e_timestamps,
                         sw_d2s = None):
    initial_time = time.time()
    win_size = window_size
    dfs = []
    start_timestamps, end_timestamps, start_times_seconds, end_times_seconds = s_e_timestamps
    # For every file
    for ind, block in enumerate(paths_to_files): 

        num_units_list = []
        d2_list = []

        nonzeros = []
        win_stds = []
        statuses = []
        datetimes = []
        firing_rates = []
        fano_factors = []
        mean_pop_counts = []
        var_pop_counts = []
        n_spikes_totals = []
        n_neurons_list = []

        start_time = start_timestamps[ind]
        end_time_of_file = end_timestamps[ind]
        start_time_seconds = start_times_seconds[ind]
        end_time_seconds = end_times_seconds[ind]
        logger.info(f"Looking at file with electrode data from {start_time} : {end_time_of_file}")
        
        start_i = 0
        bins_per_window = int(win_size/bin_size)
        num_windows = int((end_time_seconds-start_time_seconds)//window_size)
        for window in range(num_windows):
            win_start = window*bins_per_window
            win_end = win_start + bins_per_window
            pop_rate_d2 = pop_rate_d2s[ind][win_start:win_end]
            if sw_d2s: sw_d2 = sw_d2s[ind][:,win_start:win_end]
            res = d2_yw(pop_rate_d2 + 1e-10 * np.random.randn(len(pop_rate_d2)), 
                    binsz=bin_size, p=order)
            if not res: res = {}
            res['d2'] = res.get('d2', np.nan)
            res['tau'] = res.get('tau', np.nan)
            res['rho'] = res.get('rho', np.nan)
            res['bic'] = res.get('bic', np.nan)

            d2_list.append(res)

            #Calculate additional info/factors
            duration = win_size
            n_neurons = neuron_counts[ind]
            if sw_d2s: 
                n_spikes_total = np.sum(sw_d2)
                firing_rate_hz = n_spikes_total / (n_neurons * duration)
            else:
                n_spikes_total = np.sum(pop_rate_d2)
                firing_rate_hz = n_spikes_total / (n_neurons * duration)

            # Fano factor of population counts across bins
            mean_count = np.mean(pop_rate_d2)
            var_count = np.var(pop_rate_d2, ddof=1) if len(pop_rate_d2) > 1 else np.nan

            if mean_count > 0:
                fano_factor = var_count / mean_count
            else:
                fano_factor = np.nan

            firing_rates.append(firing_rate_hz)
            fano_factors.append(fano_factor)
            mean_pop_counts.append(mean_count)
            var_pop_counts.append(var_count)
            n_spikes_totals.append(n_spikes_total)
            n_neurons_list.append(n_neurons)

            nonzeros.append(np.count_nonzero(pop_rate_d2))
            win_stds.append(np.std(pop_rate_d2))
            datetimes.append(start_time + pd.to_timedelta(window*window_size, 'second'))
            status = "Success"
            if res is None:  status = "Window too small"
            elif "error" in res: status = f"Math Error: {res['error']}"
            elif np.isnan(res.get('d2', np.nan)): status = "Result is NaN"
            statuses.append(status)
            num_units_list.append(neuron_counts[ind])
        
        #make pandas df for the block and save
        df = pd.DataFrame()
        df['num_units'] = num_units_list

        df['d2'] = [x['d2'] for x in d2_list]
        df['datetime'] = datetimes
        df['tau'] = [x['tau'] for x in d2_list]
        df['rho'] = [x['rho'] for x in d2_list]
        df['bic'] = [x['bic'] for x in d2_list]
        df['firing_rate'] = firing_rates
        df['fano_factor'] = fano_factors
        df['mean_pop_counts'] = mean_pop_counts
        df['var_pop_counts'] = var_pop_counts
        df['n_spikes_total'] = n_spikes_totals
        df['n_neurons'] = n_neurons_list
        df['status'] = statuses
        df['std'] = win_stds
        df['nonzero_bins'] = nonzeros
        dfs.append(df)

    final_time = time.time()
    logger.info(f"Took about {final_time-initial_time:.1f} seconds, successfully calculated")
    d2_df = pd.concat(dfs)
    # Check correlations with failure
    if "Result is NaN" in d2_df['status'].values:
        low_activity = d2_df[d2_df['status'] == "Result is NaN"]['nonzero_bins'].mean()
        logger.info(f"Avg nonzero bins in NaN windows: {low_activity:.5f} / {window_size/bin_size}")
    d2_df = d2_df[d2_df['d2']!=np.nan]
    logger.info(f"Failed windows: {len(d2_df[d2_df['status']=='Failure'])}")
    return d2_df

def split_into_conditions(by_state, master_sleep_df):
    condition_changes = []
    prev = None
    for ind, x in enumerate(master_sleep_df['exp_phase']):
        if x!=prev:
            condition_changes.append(ind)
            prev = x
    condition_changes.append(len(master_sleep_df['exp_phase'])-1)
    conditions = ['baseline', 'SDCI', 'post']

    baseline, sdci, postsdci = [],[],[]
    for condition_change in range(len(condition_changes[:-1])):
        data = []
        for i, state in enumerate(by_state):
            #18:47:05.069722
            start_point = master_sleep_df['datetime'][condition_changes[condition_change]]
            end_point = master_sleep_df['datetime'][condition_changes[condition_change+1]]
            good_inds = []
            for ind, data_point in enumerate(state['datetime']):
                if start_point < data_point < end_point:
                    good_inds.append(ind)
            data.append(pd.DataFrame([state.iloc[ind] for ind in good_inds]))
        if not condition_change:
            baseline = data
        elif condition_change==1:
            sdci = data
        else:
            postsdci = data
    return baseline, sdci, postsdci

def datetime_to_zt(ma):
    zt = ma.hour*60*60 + ma.minute*60 + ma.second - 7*60*60
    if zt < 0: zt += 24*60*60
    return zt
    
    
# get spike metrics
def get_spike_matrix(df, colname):
    tqdm.pandas()
    df['ifr'] = df[colname].progress_apply(compute_ifr)
    df['mean_isis'] = df[colname].progress_apply(compute_mean)
    df['median_isis'] = df[colname].progress_apply(compute_median)
    df['max_isis'] = df[colname].progress_apply(compute_max)
    df['min_isis'] = df[colname].progress_apply(compute_min)
    df['sd'] = df[colname].progress_apply(compute_std)
    df['cv'] = df[colname].progress_apply(compute_cv)
    df['cv2'] = df[colname].progress_apply(compute_cv2)
    df['lv'] = df[colname].progress_apply(compute_lv)
    df['lvr'] = df[colname].progress_apply(compute_lvr)
    
# sample with replacement of n elements from the list
def sample_n_elements(lst, n):
    sample = np.random.choice(lst, size = n, replace = True)
    return(sample)

#functions
def compute_cv2(x, with_nan=False):
    if len(x) == 0:
        return np.nan
    return elephant.statistics.cv2(x, with_nan)
    
def compute_lv(x, with_nan=False):
    if len(x) == 0:
        return np.nan
    return elephant.statistics.lv(x, with_nan)

def compute_lvr(x, with_nan=False):
    if len(x) == 0:
        return np.nan
    #it assumes milliseconds
    return elephant.statistics.lvr(x*1000, with_nan)

def compute_std(x):
    if len(x) == 0:
        return np.nan
    return np.std(x)

def compute_mean(x):
    if len(x) == 0:
        return np.nan
    return np.mean(x)    

def compute_ifr(x):
    if len(x) == 0:
        return np.nan
    return np.mean(np.reciprocal(x))
    
def compute_cv(x, with_nan=False):
    if len(x) == 0:
        return np.nan
    return np.std(x)/np.mean(x)
    
def compute_median(x):
    if len(x) == 0:
        return np.nan
    return np.median(x)
    
def compute_min(x):
    if len(x) == 0:
        return np.nan
    return np.min(x)

def compute_max(x):
    if len(x) == 0:
        return np.nan
    return np.max(x)
    
#These take longer to run 
def fit_gamma(x):
    try:
        alpha,loc,beta = stats.gamma.fit(x)
    except: #sometimes a gamma distribution is inappropriate and fails to fit
        alpha,loc,beta = None,None,None
    return alpha,beta

def compute_power_spectrum(all_isis, method):
    all_spiketimes = (np.round(np.add.accumulate(all_isis),3)*1000).astype(int)
    all_spiketimes = np.insert(all_spiketimes, 0, 0)
    spiketrain = np.zeros(all_spiketimes[-1]+1)
    spiketrain[all_spiketimes] = 1
    x = spiketrain
    if method == 'periodogram':
        PSD = signal.periodogram(x,1000)
    elif method == 'welch':
        PSD = signal.welch(x, 1000, nperseg = 512)
    f,Pxx = PSD[0],PSD[1]
    #Pxx = Pxx/np.max(Pxx) #max normalize it
    return f, Pxx

def extract_band(psd_bins, psd_vals, band, method='periodogram'):
    band_idxs = np.where((psd_bins>band[0])&(psd_bins<band[1]))[0]
    if method == 'periodogram':
        mean_psd = np.mean(psd_vals[band_idxs])
    elif method == 'welch':
        mean_psd = (integrate.simpson(psd_vals[band_idxs], dx = psd_bins[1]-psd_bins[0]))/(band[1]-band[0]) 
    else:
        raise ValueError('Invalid method')
    return mean_psd

def freq_ratios(x, method='periodogram', normalize=True):
    psd_bins, psd_vals = compute_power_spectrum(x, method)

    delta_psd = extract_band(psd_bins,psd_vals,(0,4),method=method)
    theta_psd = extract_band(psd_bins,psd_vals,(4,8),method=method)
    alpha_psd = extract_band(psd_bins,psd_vals,(8,12),method=method)
    beta_psd = extract_band(psd_bins,psd_vals,(12,30),method=method)
    gamma_psd = extract_band(psd_bins,psd_vals,(30,50),method=method)
    psd_sum = delta_psd+theta_psd+alpha_psd+beta_psd+gamma_psd
    if normalize is True:
        delta_rat = delta_psd/psd_sum
        theta_rat = theta_psd/psd_sum
        alpha_rat = alpha_psd/psd_sum
        beta_rat = beta_psd/psd_sum
        gamma_rat = gamma_psd/psd_sum
    else:
        delta_rat = delta_psd
        theta_rat = theta_psd
        alpha_rat = alpha_psd
        beta_rat = beta_psd
        gamma_rat = gamma_psd
    return delta_rat,theta_rat,alpha_rat,beta_rat,gamma_rat

# turn a list of list into a single list
def flattern_list(a):
    new = []
    for items in a:
        for sub_items in items:
            new.append(sub_items)
    return new

# get first or last n elements of a list
def get_first_n_elements(lst: list, n, type = 'first'):
    if len(lst) < n:
        return []
    else:
        if type == 'first':
            return lst[:n]
        elif type == 'last':
            return lst[-n:]
        else:
            raise ValueError('Invalid type')

# sample with replacement of n elements from the list
def sample_n_elements(lst, n):
    sample = np.random.choice(lst, size = n, replace = True)
    return(sample)

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
                    'learning_rate_init':1E-4,
                    'batch_size':60,
                    'max_iter':400,
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
    print("Trying resampling")
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
            print("trying training")
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

def load_dataset(path='/media/HlabShare/Toby_work/sleep_indiv_neuron/filtered_full_animal_isi.pkl', filters=[]):
    #'/media/HlabShare/Ben_work/isi/all_animal_isi_df.pkl'
    dataset = pd.read_pickle(path)
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
        grouped_data = get_extended_condensed_data(df_app_dataset, False)
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

import time
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