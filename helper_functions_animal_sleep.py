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