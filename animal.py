# ------------------------
#Library of Alexandria
# ------------------------

from scipy.stats import sem
    
import pickle
import time
import pandas as pd
import os
from tqdm import tqdm
from datetime import timedelta
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.diagnostic import acorr_ljungbox
import numpy as np
import glob
import natsort
from natsort import natsorted
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import PowerNorm
from matplotlib.widgets import Button
import sahara_work as saw
import os
import os.path as op
from pathlib import Path
import musclebeachtools as mbt
import scipy
from scipy import stats
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
from james_work import jnm_sleep_scoring_pipeline
import yasa
from yasa import bandpower
import re
import criticality as cr
from criticality import pvaluenew2 as pv
from criticality import exclude as ex
from helper_functions import split_into_conditions, calculate_criticality, collect_spike_data_timings, calculate_criticality_sleep_windows
import logging
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


class Animal:
    def __init__(self, name, master_sleep_df, neuron_counts, intervention_start, intervention_end, job_params, neuron_types, loaded_file_names, end_lengths):
        self.name = name
        self.job_params = job_params
        self.neuron_counts = neuron_counts
        self.master_sleep_df = master_sleep_df
        self.intervention_start = intervention_start
        self.intervention_end = intervention_end
        self.loaded_file_names = loaded_file_names
        self.end_lengths = end_lengths
        self.neuron_types = neuron_types

        #self.figure_out_indiv_neuron_firing_rates()
    
    def do_criticality(self, window_size, order, bin_size, pop_rate_d2s,
                             sleep_s_e_timestamps, show_output=False, sw_d2s=None):
        
        self.d2_df = self.calculate_criticality_sleep_windows(window_size, order, bin_size, pop_rate_d2s,
                             sleep_s_e_timestamps, show_output=False, sw_d2s=None)
        self.by_state = self.post_processing_info()
        self.split_into_conds()
        self.days, self.unique_dates, self.cond_yes_no = self.split_by_date(self.d2_df, self.by_state)
        
        
    def post_processing_info(self):
        d2_df = self.d2_df
        intervention_start = self.intervention_start
        d2_state_sleep_changes = {}
        prev_state = None
        for i, state in enumerate(d2_df['sleep_state']):
            if state!=prev_state:
                d2_state_sleep_changes[i] = state
            prev_state = state
        d2_list_changes = list(d2_state_sleep_changes.keys())
        d2_list_changes.append(len(d2_df['sleep_state']))
        #Collect the specific data that is in each state from the total time series
        by_state = []
        for i, col_state in enumerate([1,2,3,5]):
            by_state.append(d2_df[d2_df['sleep_state']==col_state])
            by_state[i] = by_state[i].dropna()
            by_state[i] = pd.DataFrame(by_state[i])
        print([len(i) for i in by_state])
        self.d2_state_sleep_changes = d2_state_sleep_changes
        self.d2_list_changes = d2_list_changes
        return by_state

    def split_by_date(self, dataframe, enum_arr): 
        #Find every unique date that these recordings cover
        d2_df = dataframe
        intervention_start = self.intervention_start
        unique_dates = set()
        for dt in d2_df['datetime']:
            unique_dates.add((dt.month, dt.day))
        year = d2_df['datetime'].iloc[0].year
        unique_dates = natsorted(list(unique_dates))
        #Split up by day
        #year = str(intervention_start.year)
        #print(intervention_start)
        days = []
        cond_yes_no = [False]*len(unique_dates)
        for index, date in enumerate(unique_dates):
            filtered_inp = []
            for i, state in enumerate(enum_arr):
                start_point = pd.to_datetime(f'{year}-{date[0]}-{date[1]} 07:00:00')
                if start_point.day >= intervention_start.day: 
                    cond_yes_no[index] = True
                end_point = start_point + pd.to_timedelta(1, 'day')
                left = np.searchsorted(state['datetime'], start_point)
                right = np.searchsorted(state['datetime'], end_point, side='right')
                addon = state.iloc[left:right]
                filtered_inp.append(addon)
            #print(len(filtered_inp))
            #print([len(x) for x in filtered_inp])
            days.append(filtered_inp)
        return days, unique_dates, cond_yes_no
        
    def get_post_processed_arrays(self):
        return self.d2_state_sleep_changes, self.d2_list_changes, self.by_state, self.unique_dates, self.cond_yes_no, self.days
    
    def split_into_conds(self):
        self.baseline, self.sdci, self.postsdci = split_into_conditions(self.by_state, self.master_sleep_df)
        #print(baseline, sdci, postsdci)
        #Combine some I guess
        self.awake = pd.concat((self.by_state[0], self.by_state[3]))
        self.asleep = pd.concat((self.by_state[1], self.by_state[2]))
        self.awake_base = pd.concat([self.baseline[0], self.baseline[3]])
        self.asleep_base = pd.concat([self.baseline[1], self.baseline[2]])
        if len(self.sdci):
            self.awake_sdci = pd.concat([self.sdci[0], self.sdci[3]])
            self.asleep_sdci = pd.concat([self.sdci[1], self.sdci[2]])
        
    def figure_out_indiv_neuron_firing_rates(self, sw_d2s):
        totals = [len(sw_d2) for sw_d2 in sw_d2s]
        self.rsu = [file.count("RSU") for file in self.neuron_types]
        self.fs = [file.count("FS") for file in self.neuron_types]
        self.ratio_rsu = [self.rsu[i]/totals[i] for i in range(len(totals))]
        self.firing_rates_total = []
        self.firing_rates_total_fs = []
        self.firing_rates_total_rsu = []
        for i, sw_d2 in enumerate(sw_d2s):
            total_time = self.job_params['d2_binsz'] * len(sw_d2[0])
            neuron_type_spec = self.neuron_types[i]
            #print(neuron_type_spec)
            firing_rates = []
            firing_rates_rsu = []
            firing_rates_fs = []
            for ind, neuron in enumerate(sw_d2):
                total_spikes = np.sum(neuron)
                fr = total_spikes/total_time
                firing_rates.append(fr)
                if neuron_type_spec[ind]=='RSU': firing_rates_rsu.append(fr)
                if neuron_type_spec[ind]=='FS': firing_rates_fs.append(fr)
            self.firing_rates_total.append(firing_rates)
            self.firing_rates_total_fs.extend(firing_rates_fs)
            self.firing_rates_total_rsu.extend(firing_rates_rsu)
        self.mean_firing_rates = np.mean(self.firing_rates_total)
        self.mean_firing_rates_rsu = np.mean(self.firing_rates_total_rsu)
        self.mean_firing_rates_fs = np.mean(self.firing_rates_total_fs)
        
    def calculate_normal_window_criticality(self, window_size, order, bin_size, pop_rate_d2s, sw_d2s=None):
        s_e_lengths = collect_spike_data_timings(self.loaded_file_names, self.end_lengths)
        self.base_d2_df = calculate_criticality(self.loaded_file_names, window_size, 
                                                    order, pop_rate_d2s, self.neuron_counts, bin_size, s_e_lengths, sw_d2s=sw_d2s)
        return self.base_d2_df
    
    def calculate_criticality_sleep_windows(self, window_size, order, bin_size, pop_rate_d2s,
                             sleep_s_e_timestamps, show_output=False, sw_d2s=None):
        s_e_timestamps = collect_spike_data_timings(self.loaded_file_names, self.end_lengths)
        return calculate_criticality_sleep_windows(self.loaded_file_names, window_size, order, 
                                  pop_rate_d2s, self.neuron_counts, bin_size,
                                 s_e_timestamps, sleep_s_e_timestamps, show_output=show_output, sw_d2s=sw_d2s)
       