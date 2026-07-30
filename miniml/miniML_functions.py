from __future__ import annotations
from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import ruptures as rpt
import scipy as sc


# - - - - - - - - - - - - - - - - - - - - - - -
# functions for evaluation of individual events
# - - - - - - - - - - - - - - - - - - - - - - -

def get_event_peak(data: np.ndarray, event_num: int, add_points: int, window_size: int, diffs: np.ndarray) -> int:
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
        right_window_limit = int(data.shape[0]/5)

    peak_position = np.argmax(data[add_points:add_points+right_window_limit]) + add_points

    return peak_position



def legacy_get_event_baseline(data: np.ndarray, duration: int, event_num: int, add_points, diffs: np.ndarray, 
                              peak_positions: np.ndarray, positions: np.ndarray):
    """
    Calculate the baseline and baseline variance for an event in the given data.

    Parameters:
    - data (np.ndarray): The input data.
    - duration (int): The duration (in points) to consider for baseline calculation.
    - event_num (int): The index of the event.
    - add_points (int): The number of additional points to consider.
    - diffs (np.ndarray): The differences between consecutive peak positions.
    - peak_positions (np.ndarray): The positions of the peaks relative to start of the event snippets.
    - positions (np.ndarray): The absolute positions of the events in the main trace.

    Returns:
    - baseline (float): The calculated baseline.
    - bsl_var (float): The calculated baseline variance.
    - bsl_start (int): The starting index for baseline calculation.
    - bsl_end (int): The ending index for baseline calculation.
    """
    previous_peak_in_trace = int(peak_positions[event_num-1] + positions[event_num-1] - add_points)
    steepest_rise_in_trace = int(positions[event_num])

    if (steepest_rise_in_trace - previous_peak_in_trace) <= (add_points * 1.2) and event_num != 0:
        dd = diffs[event_num-1] - (peak_positions[event_num-1] - add_points) # get distance between previous peak and steepest rise (= search window for onset)
        if (add_points - dd) < 0:
            min_position = np.argmin(data[0:add_points])
        else:
            min_position = np.argmin(data[add_points-dd:add_points]) + (add_points - dd)

        bsl_duration = int(duration / 10) # make baseline duration shorter if previous event is close
        bsl_start = (min_position - bsl_duration // 2) if (min_position - bsl_duration // 2) > 0 else 0
        bsl_end = (min_position + bsl_duration // 2)
    else:
        bsl_duration = duration
        bsl_end = (add_points - (peak_positions[event_num] - add_points) * 3)
        if bsl_end < bsl_duration:
            bsl_end = bsl_duration
        bsl_start = (bsl_end - bsl_duration) if (bsl_end - bsl_duration) > 0 else 0

    if np.mean(data[bsl_start:bsl_end]) >= data[peak_positions[event_num]]:
        min_position = np.argmin(data[int(add_points/2):add_points]) + int(add_points/2)

        bsl_duration = int(duration / 4)
        bsl_start = (min_position - bsl_duration // 2) if (min_position - bsl_duration // 2) > 0 else 0
        bsl_end = (min_position + bsl_duration // 2)

    baseline, bsl_var = np.mean(data[bsl_start:bsl_end]), np.std(data[bsl_start:bsl_end])
    if np.isnan(baseline):
        raise ValueError('Baseline could not be determined. Will lead to downstream issues.')

    bsl_result = namedtuple('BaselineResult', ['value', 'var', 'start', 'end', 'duration'])

    return bsl_result(value=baseline,
                      var=bsl_var,
                      start=bsl_start,
                      end=bsl_end,
                      duration=bsl_duration)



def get_event_onset(data: np.ndarray, peak_position: int, baseline: float, baseline_var: float) -> int:
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



def get_event_risetime(data: np.ndarray, sampling_rate: int, baseline: float, min_percentage: float = 10, max_percentage: float = 90,
                       amplitude: float = None) -> tuple[float, float, float, float, float]:
    """
    Get the risetime of an event (default, 10-90%). Data will automatically be resampled to 200 kHz (by linear interpolation) sampling rate for better accuracy.

    Parameters:
    - data: A list or array-like object containing the rise data.
    - sampling_rate (int): Sampling rate in Hz
    - baseline (float): Baseline value.
    - min_percentage (float): A float representing the minimum percentage for the risetime range. Defaults to 10%.
    - max_percentage (float): A float representing the maximum percentage for the risetime range. Defaults to 90%.
    - amplitude (float): Amplitude of the event. If not given, it is set to the difference between peak and baseline.

    Returns:
    - risetime: A float representing the risetime of the event.
    - min_position_rise: A float representing the time point of the minimum position in the risetime range.
    - max_value_rise: A float representing the value of the resampled data at min_position_rise
    - max_position_rise: An float representing the time point of the maximum position in the risetime range.
    - max_value_rise: A float representing the value of the resampled data at max_position_rise
    """

    if min_percentage > max_percentage or min_percentage <= 0 or max_percentage >= 100:
        raise ValueError('Invalid risetime parameters.')

    amplitude = data[-1] - baseline if not amplitude else amplitude

    target_sampling_rate = 200_000 # Hz
    target_sampling = 1/target_sampling_rate
    current_sampling = 1/sampling_rate

    time_ax_original = np.arange(0, data.shape[0]) * current_sampling
    resampled_time_ax = np.arange(0, time_ax_original[-1] + target_sampling, target_sampling)

    rise_data = np.interp(resampled_time_ax, time_ax_original, data)

    min_level = baseline + (amplitude * min_percentage / 100)
    max_level = baseline + (amplitude * max_percentage / 100)
    rise_min_threshold = rise_data[::-1] < min_level
    rise_max_threshold = rise_data[::-1] < max_level
    # This should always be possible... If this breaks, take check in again.
    # try:
        # rise_min_level_crossing = np.argmax(rise_min_threshold)
        # rise_max_level_crossing = np.argmax(rise_max_threshold)
        # min_position_rise = rise_data.shape[0] - rise_min_level_crossing
        # max_position_rise = rise_data.shape[0] - rise_max_level_crossing
    # except ValueError:
    #     min_position_rise = 0 # bsl_start_position
    #     max_position_rise = rise_data.shape[0] - 1 # peak_position
    rise_min_level_crossing = np.argmax(rise_min_threshold)
    rise_max_level_crossing = np.argmax(rise_max_threshold)
    min_position_rise = rise_data.shape[0] - rise_min_level_crossing
    max_position_rise = rise_data.shape[0] - rise_max_level_crossing
    if max_position_rise <= min_position_rise or min_position_rise==0 or max_position_rise >= rise_data.shape[0] - 1:
        min_position_rise = 0
        max_position_rise = rise_data.shape[0] - 1
        risetime = (max_position_rise - min_position_rise) * 0.8
    else:
        risetime = max_position_rise - min_position_rise

    risetime *= 1/target_sampling_rate
    
    min_value_rise = rise_data[min_position_rise]
    min_position_rise *= 1/target_sampling_rate

    max_value_rise = rise_data[max_position_rise]
    max_position_rise *= 1/target_sampling_rate

    return risetime, min_position_rise, min_value_rise, max_position_rise, max_value_rise



def get_event_halfdecay_time(data: np.ndarray, peak_position: int, baseline: float) -> tuple[int, int]:
    """"
    Calculate halfdecay time (in points) in a stretch of data.
    
    Parameters
    ----------
    data: np.ndarray
        The data to calculate the halfdecay time from.
    peak_position: int
        The position of the peak in the data.
    baseline: float 
        The baseline of the data.
    
    Returns 
    ----------
    halfdecay_position: int
        The position of the halfdecay in the data.
    halfdecay_time: int
        The interval between peak and halfdecay position.
    """

    level = baseline + (data[peak_position] - baseline) / 2
    halfdecay_time = np.argmax(data[peak_position:] < level)
    halfdecay_position = int(peak_position + halfdecay_time)
    
    return halfdecay_position, halfdecay_time



def get_event_charge(data: np.ndarray, start_point: int, end_point: int, baseline: float, sampling: float) -> float:
    """
    Calculate charge in a given trace between start and endpoint
    
    Parameters
    ----------
    data: np.ndarray
        The data to calculate the charge from.
    start_point: int
        The start point in the trace.
    end_point: int
        The end point in the trace.
    baseline: float 
        The baseline of the event.
    sampling: float
        The sampling interval of the data.
    
    Returns 
    ----------
    charge: float
        The charge in the trace for the given start and end point, calculated vs. the provided baseline value.
    """

    integrate_array = (data[start_point:end_point]) - baseline
    try:
        charge = np.trapezoid(integrate_array, dx=sampling)
    except AttributeError:
        charge = np.trapz(integrate_array, dx=sampling)

    return charge



def get_event_halfwidth(event_data: np.ndarray, peak_index: int, baseline: float, amplitude: float, 
                        sampling_rate: float) -> tuple[float, float, float]:
    """
    Calculates the half-width, rise time to half-amplitude, and decay time to half-amplitude of an event.

    Parameters:
    - event_data: A 1D numpy array representing the snippet of data for a single event.
    - peak_index: The integer index of the event's peak within event_data.
    - baseline: The calculated baseline value for the event.
    - amplitude: The peak-to-baseline amplitude of the event (absolute value).
    - sampling_rate: The sampling rate of the data in Hz.

    Returns:
    - A tuple (half_width, t_rise_half, t_decay_half) in seconds.
    - Returns (np.nan, np.nan, np.nan) if calculation is not possible.
    """
    
    if peak_index < 0 or peak_index >= len(event_data) or amplitude <= 0 or sampling_rate <= 0:
        return np.nan, np.nan, np.nan

    half_amp_level = baseline + amplitude / 2.0
    sampling_interval = 1.0 / sampling_rate
    t_rise_half = np.nan
    t_decay_half = np.nan

    # Find rising phase 50% crossing
    # Search from start up to peak_index
    rising_phase_data = event_data[:peak_index + 1]
    # Points strictly below half_amp_level
    points_below_half_amp_rise = np.where(rising_phase_data < half_amp_level)[0]
    # Points at or above half_amp_level
    points_at_or_above_half_amp_rise = np.where(rising_phase_data >= half_amp_level)[0]

    if len(points_below_half_amp_rise) == 0 or len(points_at_or_above_half_amp_rise) == 0:
        # Data starts at or above half-amp or never crosses it on the rising phase
        pass # t_rise_half remains np.nan
    else:
        # Last point strictly below half_amp_level
        idx1_rise = points_below_half_amp_rise[-1]
        # First point at or above half_amp_level (must be after idx1_rise)
        valid_crossings_rise = points_at_or_above_half_amp_rise[points_at_or_above_half_amp_rise > idx1_rise]
        if len(valid_crossings_rise) == 0:
             pass # Should not happen if points_below and points_at_or_above are both non-empty and peak is above half-amp
        else:
            idx2_rise = valid_crossings_rise[0]

            if idx2_rise == idx1_rise + 1: # Ensure points are adjacent
                val1_rise = event_data[idx1_rise]
                val2_rise = event_data[idx2_rise]
                time1_rise = idx1_rise * sampling_interval
                time2_rise = idx2_rise * sampling_interval

                if val2_rise == val1_rise: # Avoid division by zero if data is flat
                    t_rise_half = time1_rise if half_amp_level <= val1_rise else time2_rise
                else:
                    t_rise_half = time1_rise + (time2_rise - time1_rise) * (half_amp_level - val1_rise) / (val2_rise - val1_rise)
            else: # No adjacent points found for interpolation (e.g. peak is first point above)
                if event_data[peak_index] >= half_amp_level and len(points_below_half_amp_rise) > 0:
                     # if peak itself is the first point at or above, and there are points below
                    idx1_rise = points_below_half_amp_rise[-1]
                    idx2_rise = peak_index
                    if idx2_rise == idx1_rise +1 : # if peak is adjacent to the point below
                        val1_rise = event_data[idx1_rise]
                        val2_rise = event_data[idx2_rise]
                        time1_rise = idx1_rise * sampling_interval
                        time2_rise = idx2_rise * sampling_interval
                        if val2_rise == val1_rise:
                             t_rise_half = time1_rise if half_amp_level <= val1_rise else time2_rise
                        else:
                            t_rise_half = time1_rise + (time2_rise - time1_rise) * (half_amp_level - val1_rise) / (val2_rise - val1_rise)

    # Find decaying phase 50% crossing
    # Search from peak_index to end
    decaying_phase_data = event_data[peak_index:]
    # Points at or above half_amp_level in the context of decaying_phase_data indices
    points_at_or_above_half_amp_decay = np.where(decaying_phase_data >= half_amp_level)[0]
    # Points strictly below half_amp_level in the context of decaying_phase_data indices
    points_below_half_amp_decay = np.where(decaying_phase_data < half_amp_level)[0]

    if len(points_at_or_above_half_amp_decay) == 0 or len(points_below_half_amp_decay) == 0:
        # Data ends at or above half-amp or never crosses it on the decaying phase
        pass # t_decay_half remains np.nan
    else:
        # Last point at or above half_amp_level (relative to peak_index)
        idx1_decay_rel = points_at_or_above_half_amp_decay[-1]
         # First point strictly below half_amp_level (relative to peak_index, must be after idx1_decay_rel)
        valid_crossings_decay = points_below_half_amp_decay[points_below_half_amp_decay > idx1_decay_rel]

        if len(valid_crossings_decay) == 0:
            pass
        else:
            idx2_decay_rel = valid_crossings_decay[0]
            
            # Convert to absolute indices in event_data
            idx1_decay = peak_index + idx1_decay_rel
            idx2_decay = peak_index + idx2_decay_rel

            if idx2_decay == idx1_decay + 1: # Ensure points are adjacent
                val1_decay = event_data[idx1_decay]
                val2_decay = event_data[idx2_decay]
                time1_decay = idx1_decay * sampling_interval
                time2_decay = idx2_decay * sampling_interval

                if val1_decay == val2_decay: # Avoid division by zero
                    t_decay_half = time1_decay if half_amp_level >= val1_decay else time2_decay
                else:
                    # Interpolate: t = t1 + (t2-t1)*(level-y1)/(y2-y1)
                    # Here, level is half_amp_level, y1 is val1_decay, y2 is val2_decay
                    t_decay_half = time1_decay + (time2_decay - time1_decay) * (half_amp_level - val1_decay) / (val2_decay - val1_decay)
            else: # No adjacent points found for interpolation
                 # This case implies the data drops below half_amp_level not adjacently after being above it
                 pass

    if np.isnan(t_rise_half) or np.isnan(t_decay_half):
        return np.nan, np.nan, np.nan

    half_width = t_decay_half - t_rise_half
    
    # Ensure half_width is not negative due to edge cases or flat peaks
    if half_width < 0:
        return np.nan, t_rise_half, t_decay_half

    return half_width, t_rise_half, t_decay_half



def get_segment_stats(breakpoints: list, data: np.ndarray):
    '''
    Calculate median, variance, and slope for each segment in the provided data.
    '''
    values, slopes, variances = [], [], []
    for i, p2 in enumerate(breakpoints):
        p1 = breakpoints[i - 1] if i else 0
        p1 += 1
        p2 -= 1
        values.append(np.median(data[p1:p2]))
        variances.append(np.std(data[p1:p2]))
        if p2 - p1 > 1:
            coef = np.polynomial.polynomial.Polynomial.fit(np.arange(p1,p2), data[p1:p2], 1).convert().coef
            if len(coef) > 1:
                slopes.append(coef[1])
            else:
                slopes.append(0.0)
        else:
            slopes.append(0.0)

    return np.array(values), np.array(variances), np.array(slopes)



def get_steepest_rise_position(data: np.ndarray, filter_win: int=20):
    '''
    Calculate the position of the steepest rise in the given data.
    '''
    win = sc.signal.windows.hann(filter_win)
    filtered_data = sc.signal.convolve(data, win, mode='same') / sum(win)

    return np.argmax(np.gradient(filtered_data))



def baseline_score(positions: np.ndarray, median_values: np.ndarray, slope_values: np.ndarray, 
                   variance_values: np.ndarray, steepest_rise: int, weights: list=[0.5, 0.35, 0.1, 0.05], verbose: int=0) -> float:
    '''
    Calculate a weighted baseline score for the given data.
    Four parameters are used:
    1. Position relative to the steepest rise (penalizes positions after the steepest rise).
    2. Median value of the segment (lower median values are preferred).
    3. Slope of the segment (lower slopes are preferred).
    4. Variance of the segment (lower variance is preferred).
    The weights for each parameter can be adjusted using the 'weights' argument.

    '''
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
    weights = np.asarray(weights, dtype=float)

    return weights @ arr



def get_event_baseline(data: np.ndarray, bsl_duration: int, event_num: int, add_points,
                       peak_position: int, positions: np.ndarray, debug: bool=False):
    """
    Calculate the baseline and baseline variance for an event in the given data.  
    
    Uses change point detection to find the baseline segment.

    Parameters
    ----------
    - data (np.ndarray): The input data (i.e. the event snippet).
    - bsl_duration (int): The duration (in points) to consider for baseline calculation.
    - event_num (int): The index of the event.
    - add_points (int): The number of additional points to consider (typically 200 samples).
    - peak_position (int): The position of the peak relative to start of the event snippet.
    - positions (np.ndarray): The absolute positions of the events in the main trace.
    - debug (bool): If True, enables debug mode with additional plots.

    Returns
    ----------

    BaselineResult: A named tuple containing the following fields:
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
        coef = fit.convert().coef
        if debug:
            print(coef)
        if len(coef) > 1 and coef[1] < -0.12 and bsl_duration > 20:
            bsl_duration = bsl_duration // 2
            bsl_snippet = data[bsl_end - bsl_duration: bsl_end]
            fit = np.polynomial.polynomial.Polynomial.fit(np.arange(bsl_snippet.shape[0]), bsl_snippet, 1)
            coef = fit.convert().coef
            if debug:
                print(coef)
            if len(coef) > 1 and coef[1] < -0.12 and bsl_duration > 20:
                bsl_duration = bsl_duration // 2
                bsl_snippet = data[bsl_end - bsl_duration: bsl_end]
                fit = np.polynomial.polynomial.Polynomial.fit(np.arange(bsl_snippet.shape[0]), bsl_snippet, 1)
                coef = fit.convert().coef
                if debug:
                    print(coef)
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

