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
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({'font.family':'DejaVu Sans'})
from matplotlib.widgets import SpanSelector
import matplotlib.dates as mdates
import matplotlib.patches as patches
from matplotlib.colors import PowerNorm
from matplotlib.widgets import Button
import matplotlib.lines as line2d
import os
import os.path as op
from pathlib import Path
import scipy
from scipy import stats
from scipy.stats import zscore, ttest_1samp
from scipy.stats import ttest_ind
import scipy.signal as signal
from scipy.signal import butter, filtfilt, cheby1,hilbert
from scipy.ndimage import gaussian_filter1d
from statsmodels.regression.linear_model import yule_walker as sm_yw
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
from PIL import Image
from IPython.display import display
from IPython.display import display, clear_output
from tqdm import tqdm
from helper_functions_animal_sleep import split_into_conditions
from helper_functions_modeling import get_cell_groupings, remove_irregular_cm, accuracy, split_train_test_epochwise, accuracy_breakdown, order_by_conf_wrong, order_by_conf_right
from CustomConfusionMatrix import ConfusionMatrixDisp
from sklearn import metrics, preprocessing
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
spec_colors = np.concatenate((colors_rgb,colors_rgb,colors_rgb,colors_rgb, colors_rgb))
ext_colors = []
i = 0.45
for ind,arr in enumerate(spec_colors):
    if not ind%len(spec_colors): i+=0.1
    ext_colors.append([*arr,i])

def line_graph_state_back(d2_df, d2_list_changes, d2_state_sleep_changes, start=None, end=None, 
                          color_coding =['w', lighter_colors[0], lighter_colors[1], colors[2], 'w', colors[3], 'w'],
                         xtick_count=35):
    fig, axes = plt.subplots(1,1, figsize=(12, 5))
    #light blue = active wake
    #pink/red = nrem
    #purple = rem
    #green = quiet wake
    #white = microarousal or noise (probably noise)
    d2_vals = list(d2_df['d2'])
    #print(d2_vals)
    if not start: start = 0
    if not end: end = len(d2_vals)-1
    cs = color_coding
    start_ind = np.searchsorted(d2_list_changes, start, side='left')-1
    start_ind = max(0, start_ind)
    end_ind = np.searchsorted(d2_list_changes, end, side='right')
    for i in range(start_ind, end_ind):
        axes.axvspan(d2_list_changes[i], d2_list_changes[i+1], color=cs[int(d2_state_sleep_changes[d2_list_changes[i]])])
    plt.plot(range(start, end), d2_vals[start:end], linewidth=0.5)
    dist = end-start
    plt.xlim(start-dist/100, end+dist/100)
    plt.xticks(range(start, end, dist//xtick_count), [f"{x.day}.{x.hour}" for x in d2_df['datetime']][start:end:dist//xtick_count], rotation=45, ha='center')

def bar_chart_with_jitter(input_arr_by_state, dot_alpha=0.4):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    avgs = [np.average(list(arr['d2'])) for arr in input_arr_by_state]
    plt.bar(range(4), avgs, color=colors)
    std_err = [sem(arr['d2']) for arr in input_arr_by_state if len(arr)]
    plt.xticks(ticks = range(4), labels = ['AW', "NREM", "REM", "QW"])
    plt.xlabel("Sleep State")
    plt.ylabel("d2")
    jitter_noise = np.random.normal(scale=0.03, size=sum([len(x) for x in input_arr_by_state]))
    cum_sum = len(input_arr_by_state[0])
    prev_sum = 0
    for i in range(4):
        #print([*lighter_colors_rgb[i], 0.5])
        plt.scatter([i+ j for j in (jitter_noise[prev_sum:cum_sum])], list(input_arr_by_state[i]['d2']), 
                                    color = (*lighter_colors_rgb[i], dot_alpha), 
                                    edgecolors=(*darker_colors_rgb[i], dot_alpha))
        if i<3: cum_sum += len(input_arr_by_state[i+1])
        prev_sum += len(input_arr_by_state[i])
    for i in range(4):
        plt.errorbar(i,avgs[i],yerr=std_err[i],color = darker_colors[i],elinewidth=2,capsize=5)
        
def box_whisker(input_arr_by_state, *args, box_colors=[colors], median_colors=[lighter_colors],
                box_linewidth=3, med_width=2, gap_between_groups = 1.49, gap_between_cols = 1,
                box_widths = 0.8, ylim = None, output=True, labels=None, inner_d2=False, ax=None, 
                color_by_group=True, rotat=0, **kwargs):
    #Split every given array along its base axis into different groups, and group separately!
    #Essentially takes all arrays, and reorganizes them, so that things that were ordered [1a,1b,1c,1d], [2a,2b,2c,2d]...
    #become [1a,2a...],[1b,2b...]... in box_whisker form. Kwargs are passed into the plt.boxplot call
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    else: fig = ax.figure
    num_arrs = 1 + len(args)
    total_gap = gap_between_groups + gap_between_cols*(num_arrs-1)
    number_groups = len(input_arr_by_state)
    bits = []
    total_poses = []
    for i, arr in enumerate((input_arr_by_state, *args)):
        if(not sum([len(x) for x in arr])):
            num_arrs -= 1
            total_gap = gap_between_groups + gap_between_cols*(num_arrs-1)
            continue
        if output: print(f"Datapoints per: {', '.join([str(len(x)) for x in arr])}")
        positions_for_this = []
        x = 1+i*gap_between_cols
        while x < 1+i*gap_between_cols + total_gap*number_groups:
            positions_for_this.append(x)
            x += total_gap
        total_poses.append(positions_for_this)
        x_s = []
        for x in arr:
            if inner_d2:
                if len(x):
                    x_s.append(x['d2'])
                else: x_s.append([])
            else: 
                x_s.append(x)
        #print([len(x) for x in x_s])
        bits.append(ax.boxplot(x_s, positions=positions_for_this[:number_groups], 
                               widths=box_widths, **kwargs))
        boxplot = bits[-1]
        #print(boxplot)
        spec_color = box_colors[i%len(box_colors)]
        spec_med_color = median_colors[i%len(median_colors)]
        if not color_by_group:
            spec_color = [box_colors[i%len(box_colors)][i] for j in range(len(boxplot['boxes']))]
            spec_med_color = [median_colors[i%len(median_colors)][i] for j in range(len(boxplot['boxes']))]
        for ind, box in enumerate(boxplot["boxes"]):
            #print(spec_color)
            box.set_c(spec_color[ind])
            box.set_linewidth(box_linewidth)
        for ind, mean in enumerate(boxplot['medians']):
            mean.set_c(spec_med_color[ind])
            mean.set_linewidth(med_width)
    center_poses = []
    x=(gap_between_cols*((num_arrs+1)/2))
    while x<1+i*gap_between_cols + total_gap*number_groups:
        center_poses.append(x)
        x+=total_gap
    plt.xticks(ticks = center_poses[:number_groups], labels = labels if labels else ((['AW', "NREM", "REM", "QW"] if number_groups==4 else (
        ['Awake', "Sleep"] if number_groups==2 else [str(y) for y in range(number_groups)]))), rotation=rotat)
    plt.ylabel("d2")
    if ylim: plt.ylim(*ylim)
    plt.xlabel("Sleep State")
    return bits, total_poses
    
def plot_across_days_and_states(input_days_arr, cond_yes_no, spec_states=None, splits_per=1):
    fig, axes = plt.subplots(1,1, figsize=(12, 5))
    spec_colors = colors[:]
    spec_colors.append('0.6')
    spec_colors.append('0.8')
    use_cond = []
    color_ind = 0
    for i in cond_yes_no:
        for spl in range(splits_per):
            use_cond.append(i)
    #print(use_cond)
    lines = []
    for big_ind, input_days in enumerate(input_days_arr):
        #print(input_days)
        if not spec_states: num_states = len(input_days[0])
        else: num_states = len(spec_states)
        avg_points_per_state = []
        std_per_state = []
        x_points = []
        for i in range(num_states):
            x_points.append([])
        for i, day in enumerate(input_days):
            #Find every day that there is data for this state
            avgs, std_err = [], []
            offset = 0
            for ind, arr in enumerate(day):
                if spec_states and ind not in spec_states: 
                    offset+=1
                    continue
                if not arr.empty:
                    initial_time = arr['datetime'].iloc[0]
                    time_delta = pd.to_timedelta(24/splits_per, 'hour')
                    for gone in range(splits_per):
                        
                        x_points[ind-offset].append(i*splits_per + gone)
                        use = arr[(initial_time+time_delta*gone) < arr['datetime']]
                        use = use[use['datetime'] < initial_time+time_delta*(gone+1)]
                        use.sort_values('datetime', inplace=True)
                        avg = np.average(list(use['d2']))
                        error = sem(list(use['d2']))
                        avgs.append(avg)
                        std_err.append(error)
                else: 
                    avgs.extend([np.nan]*splits_per)
                    std_err.extend([np.nan]*splits_per)
            for ind in range(splits_per):
                corr = [(i)*splits_per+ind for i in range(num_states)]
                #print(corr)
                avg_points_per_state.append([avgs[i] for i in corr])
                std_per_state.append([std_err[i] for i in corr])

        avg_points_per_state = np.array(avg_points_per_state, dtype=object)
        #print(avg_points_per_state)
        std_per_state = np.array(std_per_state, dtype=object)
        for i in range(num_states):
            #print(x_points[i])
            lines.append(plt.plot(x_points[i], [x[i] for x in avg_points_per_state[x_points[i]]], color=spec_colors[(i+color_ind)%(len(spec_colors))], linewidth = 3)[0])
            for day in x_points[i]:
                axes.axvspan(day, day+1, color=(*lighter_colors_rgb[0], 0.5) if use_cond[day] else (*lighter_colors_rgb[1], 0.5))
                plt.errorbar(day,avg_points_per_state[day][i],yerr=std_per_state[day][i],color = spec_colors[(i+color_ind)%(len(spec_colors))],elinewidth=2,capsize=5)
        color_ind += num_states
    plt.xlabel("Days")
    plt.ylabel("d2")
    return lines

def plot_across_days_and_states_normalized(input_days_arr, cond_yes_no, baseline_value, spec_states=None):
    fig, axes = plt.subplots(1,1, figsize=(12, 5))
    spec_colors = colors[:]
    spec_colors.append('0.6')
    spec_colors.append('0.8')
    lines = []
    color_ind = 0
    for big_ind, input_days in enumerate(input_days_arr):
        if not spec_states: num_states = len(input_days[0])
        else: num_states = len(spec_states)
       
        avg_points_per_state = []
        std_per_state = []
        x_points = []
        for i in range(num_states):
            x_points.append([])
        for i, day in enumerate(input_days):
            #Find every day that there is data for this state
            avgs, std_err = [], []
            offset = 0
            for ind, arr in enumerate(day):
                if spec_states and ind not in spec_states: 
                    offset+=1
                    continue
                if not arr.empty:
                    values = np.array(arr.sort_values('datetime')['d2']/baseline_value)
                    values -= 1
                    x_points[ind-offset].append(i)
                    avgs.append(np.average(values))
                    std_err.append(sem(values))
                else: 
                    avgs.append(0)
                    std_err.append(0)
            avg_points_per_state.append(avgs)
            std_per_state.append(std_err)
        avg_points_per_state = np.array(avg_points_per_state, dtype=object)
        std_per_state = np.array(std_per_state, dtype=object)
        for i in range(num_states):
            lines.append(plt.plot(x_points[i], [x[i] for x in avg_points_per_state[x_points[i]]], color=spec_colors[(i+color_ind)%(len(spec_colors))], linewidth = 3)[0])
            for day in x_points[i]:
                axes.axvspan(day, day+1, color=(*lighter_colors_rgb[0], 0.5) if cond_yes_no[day] else (*lighter_colors_rgb[1], 0.5))
                plt.errorbar(day,avg_points_per_state[day][i],yerr=std_per_state[day][i],color = spec_colors[(i+color_ind)%(len(spec_colors))],elinewidth=2,capsize=5)
        color_ind += num_states
    plt.xlabel("Days")
    plt.ylabel("Normalized difference from baseline all neurons d2")
    plt.plot(range(len(input_days_arr[0])), [0]*len(input_days_arr[0]), color='gray', ls='--')
    return lines


def plot_all_cms_from_animal(overall_cm, train_df, animal):
    cells = train_df[train_df['animal']==animal]['cell'].unique()
    i=1
    for cell in cells:
        i+=1
        use = overall_cm[cell]
        if len(use)>4:
            use=use[0]
        if len(use) == 3:
            state =  ['NREM','REM', 'WAKE']
        else: 
            state = ['SLEEP', 'WAKE']
        cm_xgb_display = metrics.ConfusionMatrixDisplay(
                        confusion_matrix = use, display_labels=state) 
        cm_xgb_display.plot()
        
def get_mean_cms_visual(overall_cm, train_df, cm_label='Unknown', by_region=False, by_age=False, by_geno=False,
                        by_animal=False, by_ntype=False, exclude={}, exclude_ind = set(), final_filters=[],
                       norm_size=3, subset = {}, spec_label=['SLEEP', 'WAKE'], savefig=False, addon_suff='nrw', addon_text=True,
                       se=False, sd=False, alt_se=False, color_scheme='Blues', parent_file_p='mean_cms_', file_format='svg',
                       subplot_kwargs={}, text_kw={}, prev_group_by_anim=False, these=None):
    """Get mean confusion matrices based on the conditions you go by, split as defined by get_cell_groupings() in
    helper_functions.py. Addon_text provides helpful information about the number/proportion of sleep states each cm is working with.
    Savefig will save the final image to several places depending on conditions, using the 'addon_suff' to determine which folder
    it lands in. Norm_size determines the number of columns, se/sd/alt_se are different methods of displaying variance in 
    the cms, with se being by neuron, sd being pretty large, and alt_se being sd/sqrt(num_anims) as a more accurate measure."""
    groupings, labels, genos, ages, regions, nts = get_cell_groupings(train_df, by_region, by_age, 
                                           by_geno, by_animal, by_ntype, subset, exclude, exclude_ind, final_filters,these=these)
    #print(groupings, labels)
    #print(overall_cm)
    for i, group in enumerate(groupings):
        geno, age, region, nt = genos[i], ages[i], regions[i], nts[i]
        use = train_df[[x in group for x in train_df['cell']]]
        #print(use)
        num_anims = len(use.animal.unique())
        if num_anims == 1: num_anims=len(use.cell.unique())
        fig, ax = plt.subplots(1,1,**subplot_kwargs)
        state_lens = np.zeros(norm_size)
        total_data_p = 0
        if addon_text:
            for spec_state in range(norm_size):
                for ind, cell in use[use['state']==spec_state].iterrows():
                    state_lens[spec_state] += cell['state_chunks']
                    total_data_p += cell['state_chunks']
            if norm_size==3: nr, r, wk = state_lens
            else: nr, wk=state_lens 
        spec_cells = group
        spec_cm = {}
        for cell, cm in overall_cm.items():
            #print(cell, cm)
            if len(cm)>3: cm= cm[0]
            if not prev_group_by_anim:
                if cell in spec_cells:
                    spec_cm[cell] = cm
            else:
                if not by_geno: cell += "all"
                if len(groupings)==1 or (genos[i] in cell and ages[i] in cell and regions[i] in cell and nts[i] in cell):
                    spec_cm[cell] = cm

        spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=norm_size)
        all_cms = np.array(list(spec_cm.values()))
        mean_xgb_cm = np.mean(all_cms, axis=0)
        #print(mean_xgb_cm)
        if len(mean_xgb_cm) == 3:
            state =  ['NREM','REM', 'WAKE']
        else: 
            if spec_label: 
                state = spec_label
            else: 
                state = ['NREM', 'REM']
        cm_xgb_display = ConfusionMatrixDisp(
                        confusion_matrix = mean_xgb_cm, display_labels=state) 
        sds=[]
        if se: sds = np.round(sem(all_cms, axis=0), 3)
        if sd: sds = np.round(np.std(all_cms, axis=0), 2)
        if alt_se: 
            sds = np.round(np.std(all_cms,axis=0)/np.sqrt(num_anims), 2)
        cm_xgb_display.plot(ax=ax, im_kw = {'vmin':0, 'vmax':1}, sds=sds, cmap=color_scheme, text_kw=text_kw)
        spec_reg_lab = labels[i]
        ba_addon = f" with ba={round(accuracy(mean_xgb_cm),2)}"
        spec_reg_lab = f"{cm_label}: {spec_reg_lab}"
        fig.suptitle(f"{spec_reg_lab}{ba_addon}")
        if addon_text:
            base = f'{len(spec_cells)} neurons\n|\nv\n{total_data_p} ' + \
                 f'data points:\n w: {wk} ({round(wk/total_data_p, 2)})'
            if norm_size==3:
                base += f'\n nr: {nr} ({round(nr/total_data_p, 2)})' + f'\n r: {r} ({round(r/total_data_p, 2)})'
            else: base += f'\n s: {nr} ({round(nr/total_data_p, 2)})'
            ax.text(4+((norm_size-3))*1.5, 1.25+(0.5*(norm_size-3)), base, fontsize=10, ha='center')
        plt.plot()
        if savefig:
            begin = '/media/HlabShare/Toby_work/sleep_indiv_neuron/Figures/'
                    
            fig.savefig(f'{begin}{parent_file_p}{addon_suff}/{geno}/' + \
                        f'{spec_reg_lab.replace(" ", "")}{ba_addon.replace(" ","")}.{file_format}',bbox_inches="tight")
            if by_age:
                fig.savefig(f'{begin}{parent_file_p}{addon_suff}/{age}/' + \
                        f'{spec_reg_lab.replace(" ", "")}{ba_addon.replace(" ","")}.{file_format}',bbox_inches="tight")
            if by_region:
                fig.savefig(f'{begin}{parent_file_p}{addon_suff}/{region}/' + \
                        f'{spec_reg_lab.replace(" ", "")}{ba_addon.replace(" ","")}.{file_format}',bbox_inches="tight")
        i+=1
        
def get_cm_distrib_hist(overall_cm, train_df, cm_label='Unknown', by_region=False, by_age=False, by_geno=False,
                        by_animal=False, by_ntype=False,
                        subset = {}, exclude={}, plot_eval = None, bin_count = 15, final_filters=[],
                        norm_size=3, exclude_ind=set(), these=None, group_labels=[],
                       histtype='bar', kde=False, prev_group_by_anim=False, use_colors = ext_colors,
                        **kwargs):
    """Get a histogram representation of the performance of all the individual neuronal models,
    split up by the conditions you specify as defined in get_cell_groupings() in helper_functions.py"""
    groupings, group_labels, genos, ages, regions, nts = get_cell_groupings(train_df, by_region, by_age, 
                                       by_geno, by_animal, by_ntype, subset, exclude, exclude_ind, final_filters,
                                                                       these)
    max_y = 0
    for i, group in enumerate(groupings):
        use = train_df[[x in group for x in train_df['cell']]]
        spec_cells = group
        spec_cm = {}
        for cell, cm in overall_cm.items():
            if len(cm)>3: cm= cm[0]
            if not prev_group_by_anim:
                if cell in spec_cells:
                    spec_cm[cell] = cm
            else:
                if not by_geno: cell += "all"
                if len(groupings)==1 or (genos[i] in cell and ages[i] in cell and regions[i] in cell and nts[i] in cell):
                    spec_cm[cell] = cm
        spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=norm_size)
        if not plot_eval: plot_eval=[accuracy]
        for ind, ev in enumerate(plot_eval):
            vals = [ev(cm) for cm in spec_cm.values()]
            if ind>0: group_labels.insert( i+1, group_labels[i])
            counts, bins = np.histogram(vals, bins=bin_count, range=(0,1), density=True)
            counts *= 1/bin_count
            if not kde:
                _,_,_ = plt.hist(bins[:-1], bins, weights=counts, color=use_colors[i*len(plot_eval)+ind], histtype=histtype, **kwargs)
                bin_dist = bins[1]-bins[0]
                max_y = max(max_y, max(counts))
            else: 
                sns.kdeplot(vals, color=use_colors[i*len(plot_eval)+ind], **kwargs)
                yt = plt.yticks()
                plt.xlim(0,1)
                
    plt.title(cm_label + ("" if not final_filters else f" {[x.__name__ for x in final_filters]}"))
    use_labs = []
    for lab in group_labels:
        addon = lab
        if 'total' in lab:
            addon = lab.split('in total')[0]
        elif 'all' in lab:
            addon = lab.split('all in')[1]
        use_labs.append(addon)
        
    plt.legend(use_labs)
    plt.xlabel('Accuracy')
    
    if not kde:
        plt.vlines(1/norm_size, 0, max_y, color='r', ls='--')
        #plt.yticks(np.arange(0, int(max_y), step=round(max_y/15,1)), [str(round(x*bin_dist, 3)) for x in np.arange(0, int(max_y), #step=round(max_y/15,1))])
        plt.ylabel('Proportion of neuron models')
    else:
        plt.vlines(1/norm_size, 0, yt[0][-1], color='r', ls='--')
        plt.ylabel('Density of neuron models')
        
    
def see_diff_performances(arrs, train_df, labs=None): 
    """Across a range of 8 different groupings, see how each model in arrs performs, grouping
    by model first and then by condition, allowing for a broad look over the condition set"""
    total_accuracies = []
    for i in range(8):
        by_geno, by_age, by_region = (i>=4), ((i)%4 >= 2), bool(i%2)
        groupings, group_labels, genos, ages, regions, nts = get_cell_groupings(train_df, by_region, by_age, 
                                        by_geno)
        this_arrs_acc = []
        for arr in arrs:
            accuracies = []
            for group in groupings:
                print(group_labels[0])
                spec_cells = group
                spec_cm = {}
                for cell, cm in arr.items():
                    if cell in spec_cells:
                        spec_cm[cell] = cm
                spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=3)
                mean_xgb_cm = np.mean(np.array(list(spec_cm.values())), axis=0)
                accuracies.append(accuracy(mean_xgb_cm))
            this_arrs_acc.append(accuracies)
        total_accuracies.append(this_arrs_acc)
    these = [*colors, '0.3', '0.5', '0.7', '0.8']
    use_colors = []
    for i in range(8):
        use_colors.append([])
    for i in range(8):
        for j in range(len(use_colors)):
            use_colors[i].append(these[i])
    boxplots, poses = box_whisker(*total_accuracies, inner_d2=False, 
                                  box_colors=use_colors, median_colors=use_colors, labels=labs,
                                 gap_between_cols=1.2)
    labeling_points = [boxplot['medians'][0] for boxplot in boxplots]
    plt.legend(labeling_points, ['total','region', 'age', 'region+age','geno','geno+reg','geno+age','geno+age+reg'],
              loc = 'upper right')
    
def see_diff_performances_specific(arrs, train_df, by_geno=False, by_age=False, by_region=False, by_animal=False, by_ntype=False,
                                   labs=None,
                                  **kwargs):    
    """Compare the different models for the specific condition set split up"""
    fig, ax = plt.subplots(1,1, figsize=(12,8))
    total_accuracies = []
    groupings, group_labels, genos, ages, regions, nts = get_cell_groupings(train_df, by_region, by_age, 
                                    by_geno, by_animal, by_ntype)
    for group in groupings:
        accuracies = []
        for arr in arrs:
            spec_cells = group
            spec_cm = {}
            for cell, cm in arr.items():
                if cell in spec_cells:
                    spec_cm[cell] = cm
            spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=3)
            spec_accuracies = [accuracy(cm) for cm in spec_cm.values()]
            #mean_xgb_cm = np.mean(np.array(list(spec_cm.values())), axis=0)
            accuracies.append(spec_accuracies)
        total_accuracies.append(accuracies)
    #use_colors = list(colors_rgb[:])
    #use_colors.extend(lighter_colors_rbg)
    these = [*colors, '0.3', '0.5', '0.7', '0.8']
    use_colors = []
    for i in range(len(arrs)):
        use_colors.append([])
    for i in range(min(len(these), len(use_colors))):
        for j in range(len(use_colors)):
            use_colors[i].append(these[i])
    boxplots, poses = box_whisker(*total_accuracies, inner_d2=False, 
                                  box_colors=use_colors, median_colors=use_colors, labels=labs,
                                 gap_between_cols=0.8, gap_between_groups=1.5, box_widths=0.5, ax=ax,
                                  **kwargs)
    #print(boxplots)
    labeling_points = [boxplot['medians'][0] for boxplot in boxplots]
    plt.legend(labeling_points, group_labels,
              loc = 'upper right')
    
def diff_breakdown_box(cms, train_df, norm_size=3, labs = ['BA', "NREM", "REM", "WAKE"], grouping_vars={},**kwargs):
    groupings, group_labels, _, _, _, _ = get_cell_groupings(train_df, **grouping_vars)
    outputs = []
    all_stats = {}
    for ind, group in enumerate(groupings):
        spec_cells = group
        spec_cm = {}
        for cell, cm in cms.items():
            if cell in spec_cells:
                spec_cm[cell] = cm
        spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=norm_size)
        vals = [[accuracy(cm) for cm in spec_cm.values()]]
        for i in range(norm_size):
                vals.append([cm[i][i] for cm in spec_cm.values()])
        outputs.append(vals)
        stats = []
        for i in range(norm_size+1):
            stats.append(ttest_1samp(vals[i], 1/norm_size, alternative='greater'))
        all_stats[group_labels[ind]] = stats
    if len(outputs)==1:
        return all_stats, box_whisker(outputs[0], labels=labs, **kwargs)
    else:
        return all_stats, box_whisker(*outputs, labels=labs, **kwargs)
    
def diff_breakdown_box_region(cms, train_df, norm_size=3, grouping_vars={},**kwargs):
    groupings, group_labels, genos, _, region_names, _ = get_cell_groupings(train_df, by_region=True, by_geno=True, by_age=True,
                                                                  subset={'age':'old'})
    #print(genos, regions)
    all_stats = {}
    regions = [[],[]]
    mapping = {"wt":0, "app_ps1":1}
    for geno in range(2):
        for region in range(4):
            spec_cells = groupings[region+geno*4]
            spec_cm = {}
            for cell, cm in cms.items():
                if cell in spec_cells:
                    spec_cm[cell] = cm
            spec_cm, num_rem = remove_irregular_cm(spec_cm, norm_size=norm_size)
            vals = [accuracy(cm) for cm in spec_cm.values()]
            
            regions[geno].append(vals)
            stats = []
            #stats.append(ttest_1samp(vals, 1/norm_size, alternative='greater'))
            if geno:
                stats.append(ttest_ind(vals, regions[0][region], equal_var=False))
            all_stats[group_labels[region+geno*4]] = stats
    return all_stats, box_whisker(*regions, labels=region_names[:4], **kwargs)
    
def ecf_plot(dataset, split=False, subplot_kwargs={}):
    fig, axes = plt.subplots(1,1, **subplot_kwargs)
    if not split:
        train, test = split_train_test_epochwise(dataset, (1.0, 1.0), splitting_number=10000000000)   
    else: train = dataset
    #train = train.loc[(train['sleep_state']!='wake') | (train['state_length']>30)]
    n_state = train['sleep_state'].unique()
    #print(n_state)
    for i in ['nrem', 'rem', 'wake']:
        subset = train.loc[train['sleep_state'] == i]
    #     plt.hist(subset['spike_num'], bins = 1000, alpha=0.5, label=i, density = True)
        sns.ecdfplot(subset['spike_num'], label=i)
    plt.legend(title='State')
    plt.xscale('log')
    plt.axvline(x=200, color='red', linestyle='--')
    plt.ylabel('cdf')
    
def plot_isi_performances_across_chunk_sizes(cms):
    reses = get_dataframe_isi_performance(cms)
    for val, res in reses.items():
        groups_state = res.groupby('state')
        fig, axes = plt.subplots(2,1, figsize=(6, 4), dpi=300)
        up_until = 10000
        up_until = int(up_until//100)
        for state in (*sorted(res['state'].unique()), 3):
            if state <3: groups = groups_state.get_group(state).groupby(lambda x: x//100)
            else: groups = res.groupby(lambda x: x//100)
            x_s, y_s = [],[]
            lens = []
            for group in range(0,min(max(groups.groups), up_until)):
                if group in groups.groups:
                    x_s.append(group)
                    lens.append(groups.size()[group])
                    y_s.append(np.average(groups.get_group(group).accuracy))
                else:
                    x_s.append(np.nan)
                    lens.append(0)
                    y_s.append(np.nan)
            plt.subplot(211)
            plt.plot(x_s[:up_until], y_s[:up_until])
            plt.xticks(x_s[:up_until:up_until//10], [x*100 for x in x_s[:up_until:up_until//10]])

            plt.subplot(212)
            axes[1].semilogy(lens[:up_until])
            plt.xticks(x_s[:up_until:up_until//10], [x*100 for x in x_s[:up_until:up_until//10]])

        plt.subplot(211)
        plt.title(f'{val}')
        plt.legend(('nrem', 'rem', 'wake', 'balanced_acc'))
        #plt.vlines([x*5 for x in range(1, up_until//5)], 0.3,1, color=(1,0,0,0.2), linestyle='--')
        plt.ylabel('mean accuracy')
        plt.xlabel('Total ISIs in state')
        plt.subplot(212)
        plt.legend(('nrem', 'rem', 'wake', 'combined'))
        plt.ylabel('Number of states')
        plt.xlabel('Number of ISIs')
        
def generate_accuracy_plot_across_isis(cms, line_placement=500):
    nrem, rem, wake = [],[],[]
    for cm in reversed(cms):
        #print(cm, cms[cm].keys())
        use, _ = remove_irregular_cm(cms[cm]['LogReg_over'])
        nrem.append(np.average([x[0][0] for x in use.values()]))
        rem.append(np.average([x[1][1] for x in use.values()]))
        wake.append(np.average([x[2][2] for x in use.values()]))
    x_s = [cm for cm in reversed(cms)]
    plt.plot(x_s, nrem)
    plt.plot(x_s, rem)
    plt.plot(x_s, wake)
    plt.plot(x_s, np.average([nrem, rem, wake], axis=0))
    plt.vlines(line_placement, 0.4, 0.8, color='r', linestyle='--')
    plt.legend(("nrem", 'rem', 'wake', 'balanced accuracy'), loc='lower right')
    plt.ylabel('accuracy')
    plt.xlabel('isi chunk length')
    plt.title("Accuracy performance across ISI chunk sizes")
    plt.xticks([x*500 for x in range(11)])
    
def wrong_confidence_breakdown_by_state_ratio(graphic_train, cms, subplot_kwargs={}):
    acc_break, labels,right_v_wrong = accuracy_breakdown(graphic_train, cms,
                                                        by_geno=True, by_age=True)
    plots = {}
    for i, label in enumerate(labels):
        look = order_by_conf_wrong(acc_break[i], right_v_wrong[i])
        num_to_state = {0:'nrem',1:'rem',2:'wake'}
        for right in range(3):
            s_look = look[look.corr_state==right]
            for wrong in range(3):
                if right==wrong: continue
                spec_look = s_look[s_look.wrong_guess_state==wrong]
                length = len(spec_look)
                if length < 50: continue
                smoothed_right = signal.savgol_filter(spec_look.right_conf, 50, 1)
                smoothed_bys = signal.savgol_filter(spec_look.bystander_conf, 50, 1)
                plots[(i, right, wrong)] = (list(spec_look.wrong_conf), (smoothed_right-smoothed_bys)/smoothed_bys)
    for right in range(3):
        for wrong in range(3):
            if right==wrong: continue
            fig,ax = plt.subplots(1,1,**subplot_kwargs)
            for key in plots:
                if key[1]==right and key[2]==wrong:
                    plt.plot(*plots[key])
                    final_key = key
            plt.title(f"Real:{num_to_state[right]} - Guessed:{num_to_state[wrong]}")
            plt.xlim(0.33, 0.7)
            plt.ylim(-1, 5)
            plt.legend(labels)
            plt.ylabel("Correct Confidence/Bystander Confidence")
            plt.xlabel("Confidence of Incorrect Guess")
            plt.show()
            
def right_confidence_breakdown_by_state(graphic_train, cms, subplot_kwargs={}):
    acc_break, labels,right_v_wrong = accuracy_breakdown(graphic_train, cms,
                                                        by_geno=False, by_age=False)
    for i, label in enumerate(labels):
        look = order_by_conf_right(acc_break[i], right_v_wrong[i])
        #display(look)
        num_to_state = {0:'nrem',1:'rem',2:'wake'}
        for right in range(3):
            fig,ax = plt.subplots(1,1,**subplot_kwargs)
            spec_look = look[look.corr_state==right]
            length = len(spec_look)
            if length < 50: continue
            plt.plot(list(spec_look.right_conf))
            plt.plot(signal.savgol_filter(spec_look.wrong_conf, 50, 1))
            plt.plot(signal.savgol_filter(spec_look.bystander_conf, 50, 1))
            plt.ylabel('Confidence')
            plt.legend(['Right Confidence', 'Wrong Confidence', 'Wrong Confidence'])
            plt.title(f"{label} - Real:{num_to_state[right]}")
            plt.hlines(0.5, 0, len(spec_look.wrong_conf), color='r', linestyle='--')
            plt.show()
            
def right_confidence_breakdown_by_state_ratio(graphic_train, cms,subplot_kwargs={}):    
    acc_break, labels,right_v_wrong = accuracy_breakdown(graphic_train, cms,
                                                        by_geno=True, by_age=True)
    plots = {}
    addons = []
    for i, label in enumerate(labels):
        look = order_by_conf_right(acc_break[i], right_v_wrong[i])
        num_to_state = {0:'nrem',1:'rem',2:'wake'}
        for right in range(3):
            spec_look = look[look.corr_state==right]
            length = len(spec_look)
            if length < 50: continue
            #print(spec_look.right_conf)
            #display(spec_look)
            addons.append(spec_look.wrong_guess_state.iloc[0])
            smoothed_wrong = signal.savgol_filter(spec_look.wrong_conf, 50, 1)
            smoothed_bys = signal.savgol_filter(spec_look.bystander_conf, 50, 1)
            plots[(i, right)] = (list(spec_look.right_conf), (smoothed_wrong-smoothed_bys)/smoothed_bys)

    #print(addons)
    for right in range(3):
        fig,ax = plt.subplots(1,1,**subplot_kwargs)
        for key in plots:
            if key[1]==right:
                plt.plot(*plots[key])
                final_key = key
        bys_remaining = [0,1,2]
        bys_remaining.remove(right)
        bys_remaining.remove(addons[right])
        plt.title(f"Real:{num_to_state[right]} {num_to_state[addons[right]]}/{num_to_state[bys_remaining[0]]}")
        plt.xlim(0.33, 0.7)
        plt.ylim(-1, 5)
        plt.legend(labels)
        plt.ylabel("Ratio of Confidences")
        plt.xlabel("Confidence of Correct Guess")
        plt.show()
        
def wrong_confidence_breakdown_by_state(graphic_train, cms):
    acc_break, labels,right_v_wrong = accuracy_breakdown(graphic_train, cms,
                                                        by_geno=True, by_age=True)
    for i, label in enumerate(labels):
        look = order_by_conf_wrong(acc_break[i], right_v_wrong[i])
        num_to_state = {0:'nrem',1:'rem',2:'wake'}
        for right in range(3):
            s_look = look[look.corr_state==right]
            for wrong in range(3):
                spec_look = s_look[s_look.wrong_guess_state==wrong]
                length = len(spec_look)
                if length < 50: continue
                plt.plot(list(spec_look.wrong_conf))
                plt.plot(signal.savgol_filter(spec_look.right_conf, 50, 1))
                plt.plot(signal.savgol_filter(spec_look.bystander_conf, 50, 1))
                plt.ylabel('Confidence')
                plt.legend(['Incorrect Confidence', 'Right Confidence', 'Bystander Confidence'])
                plt.title(f"{label} - Real:{num_to_state[right]} - Guessed:{num_to_state[wrong]}")
                plt.hlines(0.5, 0, len(spec_look.wrong_conf), color='r', linestyle='--')
                plt.show()
                
def plot_coefs(coefs_df):
    abs_coef = coefs_df[spike_matrix].abs()
    m_coef = abs_coef.mean()
    des = m_coef.sort_values(ascending = False)
    des.plot(kind='bar', width=0.8, figsize=(6, 4))
    plt.xticks(rotation=45)
    plt.ylabel('mean of absolute coefficient')
    plt.show()
    # box plot
    melted_df = coefs_df.drop(columns='cell').melt(id_vars=['state'], var_name='feature', value_name='coefficient')
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=melted_df, x='feature', y='coefficient', hue = 'state', showfliers=False)
    plt.xticks(rotation=45)  # Rotate x-axis labels if necessary
    plt.show()
    
def look_3d_chunk_by_cond(graphic_train, **kwargs):
    by_state = look_at_states_per_grouping(graphic_train, just_counts=False, **kwargs)
    lens = by_state[3]
    #print(lens)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    x_s, y_s, z_s = [],[],[]
    for group, neurons in lens.items():
        x,y,z = [],[],[]
        for neuron in neurons.values():
            x.append(neuron[0])
            y.append(neuron[1])
            z.append(neuron[2])
        ax.scatter(x, y, z)
        x_s.append(x)
        y_s.append(y)
        z_s.append(z)
    ax.set_xlabel('NREM Chunks')
    ax.set_ylabel('REM Chunks')
    ax.set_zlabel('WAKE Chunks')
    ax.view_init(elev=20., azim=-70, roll=0)
    plt.legend(lens.keys())
    plt.show()