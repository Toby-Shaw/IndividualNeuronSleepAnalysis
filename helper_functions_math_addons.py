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
