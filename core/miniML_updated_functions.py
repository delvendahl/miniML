from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import scipy as sc
from collections import namedtuple



def get_segment_stats(breakpoints: list, data: np.ndarray):
    values, slopes, variances = [], [], []
    for i, p2 in enumerate(breakpoints):
        p1 = breakpoints[i - 1] if i else 0
        p1 += 1
        p2 -= 1
        values.append(np.median(data[p1:p2]))
        variances.append(np.std(data[p1:p2]))
        slopes.append(np.polynomial.polynomial.Polynomial.fit(np.arange(p1,p2), data[p1:p2], 1).convert().coef[1])

    return np.array(values), np.array(variances), np.array(slopes)



def get_steepest_rise_position(data: np.ndarray, filter_win: int=20):
    win = sc.signal.windows.hann(filter_win)
    filtered_data = sc.signal.convolve(data, win, mode='same') / sum(win)

    return np.argmax(np.gradient(filtered_data))



def baseline_score(positions: np.ndarray, median_values: np.ndarray, slope_values: np.ndarray, 
                   variance_values: np.ndarray, steepest_rise: int, weights: list=[0.5, 0.35, 0.1, 0.05], verbose: int=0) -> float:
    rank_median = np.array(median_values).argsort().argsort()
    rank_slope = np.abs(slope_values).argsort().argsort()
    rank_var = np.array(variance_values).argsort().argsort()
    
    relative_positions = np.array(positions, dtype=float) - (steepest_rise + 3) # Add samples because steepest rise position is sometimes too far left due to filtering
    bkps_after_event = relative_positions > 0
    relative_positions[bkps_after_event] = np.nan
    rank_position = np.abs(relative_positions).argsort().argsort()
    rank_position[bkps_after_event] += 10 # penalize positions after steepest rise
    
    if verbose:
        print("median values", median_values, rank_median)
        print("slopes", slope_values, rank_slope)
        print("variances", variance_values, rank_var)
        print("position", positions, rank_position)
        
    arr = np.stack([rank_position, rank_median, rank_slope, rank_var])
    return weights @ arr



def get_event_baseline_v2(data: np.ndarray, bsl_duration: int, event_num: int, relative_event_position: int, positions: np.ndarray):
    """
    Calculate the baseline and baseline variance for an event in the given data.
    """
    previous_peak_present = False
    previous_peak_position = int(positions[event_num]) - int(positions[event_num - 1])
    if previous_peak_position < relative_event_position and event_num != 0:
        previous_peak_present = True

    if previous_peak_present:
        min_size = np.min([bsl_duration // 5, 3]) # ensure min_size >=3
        penalty = 0.5
        search_start = previous_peak_position
    else:
        min_size = bsl_duration
        penalty = 2
        search_start = 0

    search_end = int(relative_event_position * 1.75)
    if search_end > data.shape[0]:
        search_end = data.shape[0]

    model = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(data[search_start:search_end])
    result = model.predict(pen=penalty)

    if previous_peak_present:
        result = [pos + previous_peak_position for pos in result]
    
    values, variances, slopes = get_segment_stats(result, data)

    # print(result)
    # print("rise_pos", relative_event_position)

    score = baseline_score(result, values, slopes, variances, relative_event_position, weights=[0.55, 0.35, 0.05, 0.05], verbose=0)
    # print(score)
    bsl_ix = np.argmin(score)

    if previous_peak_present:
        peak1_position = np.argmax(data[previous_peak_position:relative_event_position]) + previous_peak_position
        if result[bsl_ix] < peak1_position: # check if baseline is before previous peak position
            bsl_ix = int(np.argwhere(np.argsort(score) == 1)[0][0])

    bsl_start = result[bsl_ix - 1] if bsl_ix else 0 
    bsl_end = result[bsl_ix]

    spacer = (bsl_end - bsl_start) // 10 if (bsl_end - bsl_start) // 10 > 3 else 0
    bsl_start += spacer
    bsl_end -= spacer
    # print(slopes[bsl_ix])

    # if slope is more negative than cutoff, split the baseline window in half and use the latter half
    slope_cutoff = np.var(data[-int(data.shape[0] * 0.1):]) * -0.1
    if slopes[bsl_ix] < slope_cutoff:
        # print("unstable baseline detected")
        half_window = (bsl_end - bsl_start)//2
        bsl_start += half_window

    # check if baseline is above event position, if so use minimum value in trace for baseline calculation
    if np.median(data[bsl_start : bsl_end]) >= data[relative_event_position]:
        # print('Baseline above event position. Using minimum for baseline calculation.')
        min_search_start = peak1_position if previous_peak_present else 0
        min_position = np.argmin(np.trace[min_search_start:relative_event_position]) + min_search_start

        bsl_start = min_position - 3
        bsl_end = min_position + 3

    # rpt.display(data, result)
    # plt.plot(np.arange(bsl_start+1, bsl_end-1), data[bsl_start+1: bsl_end-1], c='darkorange')
    # plt.axhline(baseline, linestyle='--', c= 'gray')
    # plt.plot(relative_event_position, data[relative_event_position], 'ko')
    # plt.show()

    bsl_result = namedtuple('BaselineResult', ['value', 'var', 'start', 'end', 'duration'])

    return bsl_result(value=np.median(data[bsl_start : bsl_end]),
                      var=np.var(data[bsl_start: bsl_end]),
                      start=bsl_start,
                      end=bsl_end,
                      duration=bsl_end - bsl_start)



def get_event_baseline_new(data: np.ndarray, bsl_duration: int, event_num: int, add_points,
                           peak_position: int, positions: np.ndarray, debug: bool=False):
    """
    Calculate the baseline and baseline variance for an event in the given data.

    Parameters:
    - data (np.ndarray): The input data (i.e. the event snippet).
    - bsl_duration (int): The duration (in points) to consider for baseline calculation.
    - event_num (int): The index of the event.
    - add_points (int): The number of additional points to consider.
    - peak_position (int): The position of the peak relative to start of the event snippet.
    - positions (np.ndarray): The absolute positions of the events in the main trace.
    - debug (bool): If True, enables debug mode with additional plots.

    Returns:
    - baseline (float): The calculated baseline.
    - bsl_var (float): The calculated baseline variance.
    - bsl_start (int): The starting index for baseline calculation.
    - bsl_end (int): The ending index for baseline calculation.
    """

    previous_peak_present = False
    if int(positions[event_num]) - int(positions[event_num - 1]) < add_points and event_num != 0:
        previous_peak_present = True

    bsl_limit_factor = 1.5
    search_end = int(add_points * 2)
    peak_win_start = add_points // 2
    win = sc.signal.windows.hann(25)

    penalty = 10
    trace_start = 0
    
    if previous_peak_present:
        if debug:
            print("previous peak in trace detected")
        penalty = 5
        trace_start = int(positions[event_num]) - int(positions[event_num - 1])
        if trace_start > peak_win_start:
            trace_start = peak_win_start
    # check if beginning of baseline is above peak 
    elif np.sum(data[0:peak_win_start] > data[peak_position]) > (peak_win_start / 2):
        if debug:
            print("baseline indicating previous peak")
        penalty = 5
        trace_start = int(peak_win_start / 2)

    model = rpt.KernelCPD(kernel="rbf", min_size=2).fit(data[trace_start:search_end])
    result = model.predict(n_bkps=2) 
    result = [val + trace_start for val in result]
    
    filtered_data = sc.signal.convolve(data, win, mode='same') / sum(win)
    gradient = np.gradient(filtered_data)
    ev_position = np.argmax(gradient[50:300]) + 50
    cutoff = ev_position - bsl_limit_factor * bsl_duration
    if debug:
        print(cutoff)

    if result[0] < cutoff:
        if result[1] > peak_position or result[1] < cutoff:
            result2 = model.predict(pen=penalty)
            result2 = [val + trace_start for val in result2]
            onset = result2[np.where(np.array(result2) - peak_position < 0)[0][-1]]
            if debug: 
                rpt.display(data, result2)
                print("re-analysis")
        else:
            onset = result[1]
            if debug: 
                rpt.display(data, result)
    else:
        onset = result[0]
        if debug: 
            rpt.display(data, result)

    if debug: 
        plt.axvline(result[0], linestyle=':', c='k')
        plt.axvline(ev_position, linestyle=':', c='g')
    bsl_end = onset - 10
    if bsl_end > bsl_duration:
        bsl_snippet = data[bsl_end - bsl_duration: bsl_end]
        fit = np.polynomial.polynomial.Polynomial.fit(np.arange(bsl_snippet.shape[0]), bsl_snippet, 1)
        if debug:
            print(fit.convert().coef)
        if fit.convert().coef[1] < -0.12 and bsl_duration > 20:
            bsl_duration = bsl_duration // 2
            bsl_snippet = data[bsl_end - bsl_duration: bsl_end]
            fit = np.polynomial.polynomial.Polynomial.fit(np.arange(bsl_snippet.shape[0]), bsl_snippet, 1)
            if debug:
                print(fit.convert().coef)
            if fit.convert().coef[1] < -0.12 and bsl_duration > 20:
                bsl_duration = bsl_duration // 2
                bsl_snippet = data[bsl_end - bsl_duration: bsl_end]
                fit = np.polynomial.polynomial.Polynomial.fit(np.arange(bsl_snippet.shape[0]), bsl_snippet, 1)
                if debug:
                    print(fit.convert().coef)
        if debug: 
            plt.plot(np.arange(bsl_end - bsl_duration, bsl_end), data[bsl_end - bsl_duration: bsl_end], c='darkorange')
            plt.axhline(np.median(data[bsl_end - bsl_duration: bsl_end]), linestyle='--', c='gray')
    if debug:
        plt.show()

    bsl_result = namedtuple('BaselineResult', ['value', 'var', 'start', 'end', 'duration'])

    return bsl_result(value=np.median(data[bsl_end - bsl_duration: bsl_end]),
                      var=np.var(data[bsl_end - bsl_duration: bsl_end]),
                      start=bsl_end - bsl_duration,
                      end=bsl_end,
                      duration=bsl_duration)
