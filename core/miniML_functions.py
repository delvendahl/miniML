from __future__ import annotations
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - -
# functions for evaluation of individual events
def get_event_peak(data, event_num, add_points, window_size, diffs):
    if diffs[event_num] < window_size:
        right_window_limit = int(diffs[event_num]/2)
    
    else:
        right_window_limit = int(data.shape[0]/6)

    peak_position = np.argmax(data[add_points:add_points+right_window_limit]) + add_points

    return peak_position

def get_event_baseline(data, event_num, add_points, diffs, peak_positions, positions):
    # Get baseline and onset
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
    var_factor: float=0.25

    bsl_thresh = baseline + var_factor * baseline_var
    arr = data[0:peak_position]
    below_threshold = arr[::-1] < bsl_thresh # arr[::-1] inverts the event
    try:
        level_crossing = np.argmax(below_threshold)
    except ValueError:
        level_crossing = int(peak_position/2)

    onset_position = peak_position - level_crossing
    if onset_position >= peak_position:
        level_crossing = int(peak_position/2)
        onset_position = peak_position - level_crossing

    return onset_position


def get_event_risetime(data, peak_position, onset_position):
    # get 10-90% risetime
    min_perc, max_perc=10, 90
    if not (0 <= min_perc < max_perc) and (min_perc < max_perc <= 100):
        raise ValueError('Invalid risetime parameters.')
    
    rise_data = data[onset_position:peak_position]
    amp = data[peak_position] - data[onset_position]
    min_level = data[onset_position] + amp * min_perc/100
    max_level = data[onset_position]+ amp * max_perc/100
    rise_min_threshold = rise_data[::-1] < min_level # arr[::-1] inverts the event
    rise_max_threshold = rise_data[::-1] < max_level # arr[::-1] inverts the event
    try:
        rise_min_level_crossing = np.argmax(rise_min_threshold)
        rise_max_level_crossing = np.argmax(rise_max_threshold)
        min_position_rise = peak_position - rise_min_level_crossing
        max_position_rise = peak_position - rise_max_level_crossing
    except ValueError:
        min_position_rise = onset_position
        max_position_rise = peak_position

    if max_position_rise <= min_position_rise:
        min_position_rise = onset_position
        max_position_rise = peak_position
        risetime = (max_position_rise - min_position_rise)*0.8
    elif min_position_rise==onset_position or max_position_rise==peak_position:
        min_position_rise = onset_position
        max_position_rise = peak_position
        risetime = (max_position_rise - min_position_rise)*0.8

    else:
        risetime, _ = max_position_rise - min_position_rise, (min_position_rise , max_position_rise)
        
    return risetime, min_position_rise, max_position_rise


def get_event_halfdecay_time(data, peak_position, baseline):
    '''Calculate halfdecay time (in points) in a stretch of data '''
    level = baseline + (data[peak_position] - baseline) / 2
    halfdecay_level = data[peak_position:] < level
    halfdecay_level =  np.argmax(halfdecay_level)
    halfdecay_position, halfdecay_time = int(peak_position + halfdecay_level), halfdecay_level

    return halfdecay_position, halfdecay_time


def get_event_charge(trace_data, start_point, end_point, baseline, sampling):
    '''Calculate charge in a give trace between start and endpoint'''
    integrate_array = (trace_data[start_point:end_point]) - baseline
    charge = np.trapz(integrate_array, dx=sampling)
    return charge