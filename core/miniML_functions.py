from __future__ import annotations
import numpy as np


# - - - - - - - - - - - - - - - - - - - - - - -
# functions for evaluation of individual events
def get_event_peak(data:np.ndarray, event_num, add_points, window_size, diffs):
    """
    A function that calculates the peak position of an event in a given dataset.

    Parameters:
    - data: The data containing the event.
    - event_num: The index of the event in the data.
    - add_points: The number of points to add to the event index.
    - window_size: The size of the window to consider when calculating the peak position.
    - diffs: The differences between the events in the data.

    Returns:
    - peak_position: The index of the peak position in the data.
    """

    if diffs[event_num] < window_size:
        right_window_limit = int(diffs[event_num]/2)
    
    else:
        right_window_limit = int(data.shape[0]/6)

    peak_position = np.argmax(data[add_points:add_points+right_window_limit]) + add_points

    return peak_position


def get_event_baseline(data, event_num, add_points, diffs, peak_positions, positions):
    """
    Calculate the baseline and baseline variance for an event in the given data.

    Parameters:
    - data (np.ndarray): The input data.
    - event_num (int): The index of the event.
    - add_points (int): The number of additional points to consider.
    - diffs (np.ndarray): The differences between consecutive peak positions.
    - peak_positions (np.ndarray): The positions of the peaks.
    - positions (np.ndarray): The positions of the events.

    Returns:
    - baseline (float): The calculated baseline.
    - bsl_var (float): The calculated baseline variance.
    """

    previous_peak_in_trace = int(peak_positions[event_num-1] + positions[event_num-1] - add_points)
    steepest_rise_in_trace = int(positions[event_num])
    if (steepest_rise_in_trace - previous_peak_in_trace) <= (add_points*1.2) and event_num != 0:
        dd = diffs[event_num-1] - (peak_positions[event_num-1] - add_points) # get distance between previous peak and steepest rise (== search window for onset)
        if add_points-dd < 0:
            min_position = np.argmin(data[0:add_points])
        else:
            min_position = np.argmin(data[add_points-dd:add_points]) + (add_points-dd)

        bsl_duration=int(data.shape[0] * 0.005)
        bsl_start = (min_position - bsl_duration) if (min_position - bsl_duration) > 0 else 0
        bsl_end = (min_position + bsl_duration)
    else:
        bsl_duration = int(data.shape[0] * 0.1)
        bsl_end = (add_points - (peak_positions[event_num] - add_points)*2)
        if bsl_end < bsl_duration:
            bsl_end = bsl_duration
        bsl_start = (bsl_end - bsl_duration) if (bsl_end - bsl_duration) > 0 else 0

    baseline, bsl_var = np.mean(data[bsl_start:bsl_end]), np.std(data[bsl_start:bsl_end])

    if baseline >= data[peak_positions[event_num]]:
        min_position = np.argmin(data[int(add_points/2):add_points]) + int(add_points/2)

        bsl_duration=int(data.shape[0] * 0.005)
        bsl_start = (min_position - bsl_duration) if (min_position - bsl_duration) > 0 else 0
        bsl_end = (min_position + bsl_duration)

    baseline, bsl_var = np.mean(data[bsl_start:bsl_end]), np.std(data[bsl_start:bsl_end])

    return baseline, bsl_var


def get_event_onset(data, peak_position, baseline, baseline_var):
    """
    Calculate the position of the event onset relative to the peak position.

    Parameters:
        data (numpy.ndarray): The input data array.
        peak_position (int): The position of the peak in the data array.
        baseline (float): The baseline value.
        baseline_var (float): The variance of the baseline.

    Returns:
        int: The position of the event onset relative to the peak position.
    """
    
    var_factor: float=0.25

    bsl_thresh = baseline + var_factor * baseline_var
    arr = data[0:peak_position]
    below_threshold = arr[::-1] < bsl_thresh
    try:
        level_crossing = np.argmax(below_threshold)
    except ValueError:
        level_crossing = int(peak_position/2)

    onset_position = peak_position - level_crossing
    if onset_position >= peak_position:
        level_crossing = int(peak_position/2)
        onset_position = peak_position - level_crossing

    return onset_position


def get_event_risetime(data, peak_position: int, onset_position: int):
    """
    Get the 10-90% risetime of an event.

    Parameters:
    - data: A list or array-like object containing the event data.
    - peak_position (int): An integer representing the index of the peak position in the event data.
    - onset_position (int): An integer representing the index of the onset position in the event data.

    Returns:
    - risetime: A float representing the 10-90% risetime of the event.
    - min_position_rise: An integer representing the index of the minimum position in the risetime range.
    - max_position_rise: An integer representing the index of the maximum position in the risetime range.
    """

    min_percentage = 10
    max_percentage = 90
    if not (0 <= min_percentage < max_percentage) and (min_percentage < max_percentage <= 100):
        raise ValueError('Invalid risetime parameters.')
    
    rise_data = data[onset_position:peak_position]
    amplitude = data[peak_position] - data[onset_position]
    min_level = data[onset_position] + amplitude * min_percentage / 100
    max_level = data[onset_position] + amplitude * max_percentage / 100
    rise_min_threshold = rise_data[::-1] < min_level
    rise_max_threshold = rise_data[::-1] < max_level

    try:
        rise_min_level_crossing = np.argmax(rise_min_threshold)
        rise_max_level_crossing = np.argmax(rise_max_threshold)
        min_position_rise = peak_position - rise_min_level_crossing
        max_position_rise = peak_position - rise_max_level_crossing
    except ValueError:
        min_position_rise = onset_position
        max_position_rise = peak_position

    if max_position_rise <= min_position_rise or min_position_rise==onset_position or max_position_rise==peak_position:
        min_position_rise = onset_position
        max_position_rise = peak_position
        risetime = (max_position_rise - min_position_rise) * 0.8
    else:
        risetime = max_position_rise - min_position_rise
        
    return risetime, min_position_rise, max_position_rise


def get_event_halfdecay_time(data, peak_position, baseline):
    '''Calculate halfdecay time (in points) in a stretch of data.'''
    level = baseline + (data[peak_position] - baseline) / 2
    halfdecay_level = np.argmax(data[peak_position:] < level)
    
    halfdecay_position = int(peak_position + halfdecay_level)
    halfdecay_time = halfdecay_level
    
    return halfdecay_position, halfdecay_time


def get_event_charge(data, start_point, end_point, baseline, sampling):
    '''Calculate charge in a give trace between start and endpoint'''
    integrate_array = (data[start_point:end_point]) - baseline
    charge = np.trapz(integrate_array, dx=sampling)

    return charge
