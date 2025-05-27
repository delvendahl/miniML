from __future__ import annotations
import numpy as np

# - - - - - - - - - - - - - - - - - - - - - - -
# functions for evaluation of individual events
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


def get_event_baseline(data: np.ndarray, event_num: int, add_points, diffs: np.ndarray, peak_positions: np.ndarray, positions: np.ndarray) -> tuple[float, float]:
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
    if np.isnan(baseline):
        raise ValueError('Baseline could not be determined. Will lead to downstream issues.')
    return baseline, bsl_var, bsl_start, bsl_end, bsl_duration


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

def get_event_risetime(data: np.ndarray, sampling_rate:int, baseline:float, min_percentage: float=10, max_percentage: float=90, amplitude:float=None) -> tuple[float, int, int]:
    """
    Get the risetime of an event (default, 10-90%). Data will automatically be resampled to 100 kHz (by linear interpolation) sampling rate for better accuracy.

    Parameters:
    - data: A list or array-like object containing the rise data.
    - sampling_rate (int): Sampling rate in Hz
    - baseline (float): Baseline value.
    - min_percentage (float): A float representing the minimum percentage for the risetime range. Defaults to 10%.
    - max_percentage (float): A float representing the maximum percentage for the risetime range. Defaults to 90%.
    - amplitude (float): Amplitude of the event. If not given, it is set to difference between peak and baseline.

    Returns:
    - risetime: A float representing the risetime of the event.
    - min_position_rise: A float representing the time point of the minimum position in the risetime range.
    - max_value_rise: A float representing the value of the resampled data at min_position_rise
    - max_position_rise: An float representing the time point of the maximum position in the risetime range.
    - max_value_rise: A float representing the value of the resampled data at max_position_rise
    """

    if not (0 <= min_percentage < max_percentage) and (min_percentage < max_percentage <= 100):
        raise ValueError('Invalid risetime parameters.')

    amplitude = data[-1] - baseline if not amplitude else amplitude

    target_sampling_rate = 100_000 # 100 kHz
    target_sampling = 1/target_sampling_rate

    current_sampling_rate = sampling_rate
    current_sampling = 1/current_sampling_rate

    time_ax_original = np.arange(0, data.shape[0])*current_sampling
    resampled_time_ax = np.arange(0, time_ax_original[-1] + target_sampling, target_sampling)

    rise_data = np.interp(resampled_time_ax, time_ax_original, data, left=None, right=None, period=None)

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
    min_position_rise = (rise_data.shape[0]) - rise_min_level_crossing
    max_position_rise = (rise_data.shape[0]) - rise_max_level_crossing
    if max_position_rise <= min_position_rise or min_position_rise==0 or max_position_rise >= rise_data.shape[0] - 1:
        min_position_rise = 0
        max_position_rise = rise_data.shape[0] - 1
        risetime = (max_position_rise - min_position_rise) * 0.8
    else:
        risetime = max_position_rise - min_position_rise

    risetime = risetime * (1/target_sampling_rate)
    
    min_value_rise = rise_data[min_position_rise]
    min_position_rise = min_position_rise * (1/target_sampling_rate)
    
    max_value_rise = rise_data[max_position_rise]
    max_position_rise = max_position_rise * (1/target_sampling_rate)

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
        The halfdecay time in points.
    """
    level = baseline + (data[peak_position] - baseline) / 2
    halfdecay_level = np.argmax(data[peak_position:] < level)
    
    halfdecay_position = int(peak_position + halfdecay_level)
    halfdecay_time = halfdecay_level
    
    return halfdecay_position, halfdecay_time


def get_event_charge(data: np.ndarray, start_point: int, end_point: int, baseline: float, sampling: float) -> float:
    """
    Calculate charge in a give trace between start and endpoint
    
    Parameters
    ----------
    data: np.ndarray
        The data to calculate the charge from.
    start_point: int
        The start point of the trace.
    end_point: int
        The end point of the trace.
    baseline: float 
        The baseline of the data.
    sampling: float
        The sampling interval of the data.
    
    Returns 
    ----------
    charge: float
        The charge in the trace for the given start and end point, calculated vs. the provied baseline value.
    """
    integrate_array = (data[start_point:end_point]) - baseline
    charge = np.trapz(integrate_array, dx=sampling)

    return charge


def get_event_halfwidth(event_data: np.ndarray, peak_index: int, baseline: float, amplitude: float, sampling_rate: float) -> tuple[float, float, float]:
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
