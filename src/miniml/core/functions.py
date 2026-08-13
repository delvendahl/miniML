from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import ruptures as rpt
import scipy as sc


class BaselineResult(NamedTuple):
    """
    Baseline statistics extracted for an event window.

    Attributes
    ----------
    value : float
        Mean baseline value.
    var : float
        Baseline standard deviation.
    start : int
        Start index of the baseline window.
    end : int
        End index of the baseline window.
    duration : int
        Length of the baseline window in samples.
    """

    value: float
    var: float
    start: int
    end: int
    duration: int


class RisetimeResult(NamedTuple):
    """
    Risetime statistics extracted for an event window.

    Attributes
    ----------
    duration : float
        Duration of the risetime window.
    start_time : float
        Time of the start of the risetime window.
    start_value : float
        Value at the start of the risetime window.
    end_time : float
        Time of the end of the risetime window.
    end_value : float
        Value at the end of the risetime window.
    percentage : tuple
        Tuple containing the lower and upper percentage thresholds used for
        risetime calculation.
    """

    duration: float
    start_time: float
    start_value: float
    end_time: float
    end_value: float
    percentage: tuple


def get_event_peak(
    data: np.ndarray,
    event_num: int,
    add_points: int,
    window_size: int,
    diffs: np.ndarray,
) -> int:
    """
    Locate the peak position of an event window.

    Parameters
    ----------
    data : np.ndarray
        Event snippet containing the candidate peak.
    event_num : int
        Index of the event currently being processed.
    add_points : int
        Number of pre-event samples included in the snippet.
    window_size : int
        Analysis window size in samples.
    diffs : np.ndarray
        Distances between neighboring event locations.

    Returns
    -------
    int
        Peak index within ``data``.
    """

    if diffs[event_num] < window_size:
        right_window_limit = int(diffs[event_num] / 2)

    else:
        right_window_limit = int(data.shape[0] / 5)

    peak_position = int(
        np.argmax(data[add_points : add_points + right_window_limit]) + add_points
    )

    return peak_position


def legacy_get_event_baseline(
    data: np.ndarray,
    duration: int,
    event_num: int,
    add_points: int,
    diffs: np.ndarray | list[int] | list[float],
    peak_positions: np.ndarray | list[int],
    positions: np.ndarray | list[int],
) -> BaselineResult:
    """
    Estimate the baseline for an event window.

    Parameters
    ----------
    data : np.ndarray
        Event snippet containing baseline and peak regions.
    duration : int
        Preferred baseline duration in samples.
    event_num : int
        Index of the event currently being processed.
    add_points : int
        Number of pre-event samples included in the snippet.
    diffs : np.ndarray | list[int] | list[float]
        Distances between neighboring event locations.
    peak_positions : np.ndarray | list[int]
        Peak positions relative to the event snippets.
    positions : np.ndarray | list[int]
        Absolute event positions in the full trace.

    Returns
    -------
    BaselineResult
        Baseline value, variability, and window bounds.

    Raises
    ------
    ValueError
        If no valid baseline could be determined.
    """
    previous_peak_in_trace = int(
        peak_positions[event_num - 1] + positions[event_num - 1] - add_points
    )
    steepest_rise_in_trace = int(positions[event_num])

    if (steepest_rise_in_trace - previous_peak_in_trace) <= (
        add_points * 1.2
    ) and event_num != 0:
        dd = (
            diffs[event_num - 1] - (peak_positions[event_num - 1] - add_points)
        )  # get distance between previous peak and steepest rise (= search window for onset)
        if (add_points - dd) < 0:
            min_position = np.argmin(data[0:add_points])
        else:
            min_position = np.argmin(data[add_points - dd : add_points]) + (
                add_points - dd
            )

        bsl_duration = int(
            duration / 10
        )  # make baseline duration shorter if previous event is close
        bsl_start = max(0, min_position - bsl_duration // 2)
        bsl_end = min_position + bsl_duration // 2
    else:
        bsl_duration = duration
        bsl_end = add_points - (peak_positions[event_num] - add_points) * 3
        bsl_end = max(bsl_end, bsl_duration)
        bsl_start = max(0, bsl_end - bsl_duration)

    if np.mean(data[bsl_start:bsl_end]) >= data[peak_positions[event_num]]:
        min_position = np.argmin(data[int(add_points / 2) : add_points]) + int(
            add_points / 2
        )

        bsl_duration = int(duration / 4)
        bsl_start = max(0, min_position - bsl_duration // 2)
        bsl_end = min_position + bsl_duration // 2

    baseline = float(np.mean(data[bsl_start:bsl_end]))
    bsl_var = float(np.std(data[bsl_start:bsl_end]))
    if np.isnan(baseline):
        raise ValueError(
            "Baseline could not be determined. Will lead to downstream issues."
        )

    return BaselineResult(
        value=baseline, var=bsl_var, start=bsl_start, end=bsl_end, duration=bsl_duration
    )


def get_segment_stats(
    breakpoints: list, data: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate median, variance, and slope for each segment defined by breakpoints.

    Parameters
    ----------
    breakpoints : list
        List of breakpoint indices.
    data : np.ndarray
        Input data.

    Returns
    ----------
    np.ndarray
        Median values for each segment.
    np.ndarray
        Variances for each segment.
    np.ndarray
        Slopes for each segment.
    """
    values, slopes, variances = [], [], []
    for i, p2 in enumerate(breakpoints):
        p1 = breakpoints[i - 1] if i else 0
        p1 += 1
        p2 -= 1
        values.append(np.median(data[p1:p2]))
        variances.append(np.std(data[p1:p2]))
        if p2 - p1 > 1:
            coef = (
                np.polynomial.polynomial.Polynomial.fit(
                    np.arange(p1, p2), data[p1:p2], 1
                )
                .convert()
                .coef
            )
            if len(coef) > 1:
                slopes.append(coef[1])
            else:
                slopes.append(0.0)
        else:
            slopes.append(0.0)

    return np.array(values), np.array(variances), np.array(slopes)


def get_steepest_rise_position(data: np.ndarray, filter_win: int = 20):
    win = sc.signal.windows.hann(filter_win)
    filtered_data = sc.signal.convolve(data, win, mode="same") / sum(win)

    return np.argmax(np.gradient(filtered_data))


def baseline_score(
    positions: np.ndarray | list[int],
    median_values: np.ndarray,
    slope_values: np.ndarray,
    variance_values: np.ndarray,
    steepest_rise: int,
    weights: Sequence[float] = (0.5, 0.35, 0.1, 0.05),
    verbose: int = 0,
) -> float:
    """
    Calculate a composite score for baseline selection based on multiple criteria.

    Parameters
    ----------
    positions : np.ndarray | list[int]
        List of event positions.
    median_values : np.ndarray
        List of median values for each segment.
    slope_values : np.ndarray
        List of slopes for each segment.
    variance_values : np.ndarray
        List of variances for each segment.
    steepest_rise : int
        Position of the steepest rise.
    weights : Sequence[float], optional
        Weights for each criterion, by default (0.5, 0.35, 0.1, 0.05)
    verbose : int, optional
        Verbosity level, by default 0

    Returns
    ----------
    float
        Baseline score.
    """
    rank_median = np.array(median_values).argsort().argsort()
    rank_slope = np.abs(slope_values).argsort().argsort()
    rank_var = np.array(variance_values).argsort().argsort()

    relative_positions = (
        np.array(positions, dtype=float) - (steepest_rise + 3)
    )  # Add samples because steepest rise position is sometimes too far left due to filtering
    bkps_after_event = relative_positions > 0
    relative_positions[bkps_after_event] = np.nan
    rank_position = np.abs(relative_positions).argsort().argsort()
    rank_position[bkps_after_event] += 10  # penalize positions after steepest rise

    if verbose:
        print("median values", median_values, rank_median)
        print("slopes", slope_values, rank_slope)
        print("variances", variance_values, rank_var)
        print("position", positions, rank_position)

    arr = np.stack([rank_position, rank_median, rank_slope, rank_var])
    a_weights = np.asarray(list(weights), dtype=float)

    return a_weights @ arr


def get_event_baseline(
    data: np.ndarray,
    bsl_duration: int,
    event_num: int,
    relative_event_position: int,
    positions: np.ndarray | list[int],
) -> BaselineResult:
    """
    Calculate the baseline and baseline variance for an event in the given data.

    Parameters
    ----------
    data : np.ndarray
        The input data array.
    bsl_duration : int
        The duration of the baseline window.
    event_num : int
        The index of the event for which the baseline is being calculated.
    relative_event_position : int
        The position of the event relative to the start of the data.
    positions : np.ndarray | list[int]
        The positions of all events in the data.

    Returns
    ----------
    BaselineResult.value : float
        The calculated baseline.
    BaselineResult.var : float
        The calculated baseline variance.
    BaselineResult.start : int
        The starting index for baseline calculation.
    BaselineResult.end : int
        The ending index for baseline calculation.
    BaselineResult.duration : int
        The duration of the baseline window.
    """
    previous_peak_present = False
    previous_peak_position = int(positions[event_num]) - int(positions[event_num - 1])
    if previous_peak_position < relative_event_position and event_num != 0:
        previous_peak_present = True

    if previous_peak_present:
        min_size = np.max([bsl_duration // 5, 3])  # ensure min_size >=3
        penalty = 0.5
        search_start = previous_peak_position
    else:
        min_size = bsl_duration
        penalty = 2
        search_start = 0

    search_end = int(relative_event_position * 1.75)
    search_end = min(search_end, data.shape[0])

    model = rpt.KernelCPD(kernel="rbf", min_size=min_size).fit(
        data[search_start:search_end]
    )
    result = model.predict(pen=penalty)

    if previous_peak_present:
        result = [pos + previous_peak_position for pos in result]

    values, variances, slopes = get_segment_stats(result, data)

    score = baseline_score(
        positions=result,
        median_values=values,
        slope_values=slopes,
        variance_values=variances,
        steepest_rise=relative_event_position,
        weights=[0.55, 0.35, 0.05, 0.05],
        verbose=0,
    )
    bsl_ix = np.argmin(score)

    if previous_peak_present:
        peak1_position = (
            np.argmax(data[previous_peak_position:relative_event_position])
            + previous_peak_position
        )
        if (
            result[bsl_ix] < peak1_position
        ):  # check if baseline is before previous peak position
            bsl_ix = int(np.argwhere(np.argsort(score) == 1)[0][0])

    bsl_start = result[bsl_ix - 1] if bsl_ix else 0
    bsl_end = result[bsl_ix]

    spacer = (bsl_end - bsl_start) // 10 if (bsl_end - bsl_start) // 10 > 3 else 0
    bsl_start += spacer
    bsl_end -= spacer

    # if slope is more negative than cutoff, split the baseline window in half and use the latter half
    slope_cutoff = np.var(data[-int(data.shape[0] * 0.1) :]) * -0.1
    if slopes[bsl_ix] < slope_cutoff:
        half_window = (bsl_end - bsl_start) // 2
        bsl_start += half_window

    # check if baseline is above event position, if so use minimum value in trace for baseline calculation
    if np.median(data[bsl_start:bsl_end]) >= data[relative_event_position]:
        min_search_start = peak1_position if previous_peak_present else 0
        min_position = (
            np.argmin(data[min_search_start:relative_event_position]) + min_search_start
        )

        bsl_start = min_position - 3
        bsl_end = min_position + 3

    return BaselineResult(
        value=np.median(data[bsl_start:bsl_end]),
        var=np.std(data[bsl_start:bsl_end]),
        start=bsl_start,
        end=bsl_end,
        duration=bsl_end - bsl_start,
    )


def get_event_onset(
    data: np.ndarray, peak_position: int, baseline: float, baseline_var: float
) -> int:
    """
    Locate the event onset preceding a peak.

    Parameters
    ----------
    data : np.ndarray
        Event snippet.
    peak_position : int
        Peak index within ``data``.
    baseline : float
        Baseline value.
    baseline_var : float
        Baseline standard deviation.

    Returns
    -------
    int
        Onset index within ``data``.
    """

    var_factor: float = 0.25

    bsl_thresh = baseline + var_factor * baseline_var
    arr = data[0:peak_position]
    below_threshold = arr[::-1] < bsl_thresh
    try:
        level_crossing = np.argmax(below_threshold)
    except ValueError:
        level_crossing = int(peak_position / 2)

    onset_position = peak_position - level_crossing
    if onset_position >= peak_position:
        level_crossing = int(peak_position / 2)
        onset_position = peak_position - level_crossing

    return int(onset_position)


def get_event_risetime(
    data: np.ndarray,
    sampling_rate: float,
    baseline: float,
    min_percentage: float = 10,
    max_percentage: float = 90,
    amplitude: float | None = None,
) -> RisetimeResult:
    """
    Measure the event rise time over a configurable amplitude range.

    Parameters
    ----------
    data : np.ndarray
        Rising portion of the event trace.
    sampling_rate : float
        Sampling rate in hertz.
    baseline : float
        Baseline value.
    min_percentage : float, default=10
        Lower rise-time percentage threshold.
    max_percentage : float, default=90
        Upper rise-time percentage threshold.
    amplitude : float | None, optional
        Peak-to-baseline amplitude. If omitted, it is inferred from the final
        sample in ``data``.

    Returns
    -------
    RisetimeResult
        Rise time, lower threshold crossing time, lower threshold value, upper
        threshold crossing time, upper threshold value, and rise-time percentage.

    Raises
    ------
    ValueError
        If the rise-time percentage bounds are invalid.
    """

    if min_percentage > max_percentage or min_percentage <= 0 or max_percentage >= 100:
        raise ValueError("Invalid risetime parameters.")

    amplitude = data[-1] - baseline if not amplitude else amplitude

    target_sampling_rate = 200_000  # Hz
    target_sampling = 1 / target_sampling_rate
    current_sampling = 1 / sampling_rate

    time_ax_original = np.arange(0, data.shape[0]) * current_sampling
    resampled_time_ax = np.arange(
        0, time_ax_original[-1] + target_sampling, target_sampling
    )

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
    rise_min_level_crossing = int(np.argmax(rise_min_threshold))
    rise_max_level_crossing = int(np.argmax(rise_max_threshold))
    min_position_rise = rise_data.shape[0] - rise_min_level_crossing
    max_position_rise = rise_data.shape[0] - rise_max_level_crossing
    if (
        max_position_rise <= min_position_rise
        or min_position_rise == 0
        or max_position_rise >= rise_data.shape[0] - 1
    ):
        min_position_rise = 0
        max_position_rise = rise_data.shape[0] - 1
        risetime = (max_position_rise - min_position_rise) * 0.8
    else:
        risetime = max_position_rise - min_position_rise

    risetime *= 1 / target_sampling_rate

    min_value_rise = rise_data[min_position_rise]
    min_position_rise *= 1 / target_sampling_rate

    max_value_rise = rise_data[max_position_rise]
    max_position_rise *= 1 / target_sampling_rate

    return RisetimeResult(
        duration=risetime,
        start_time=min_position_rise,
        start_value=min_value_rise,
        end_time=max_position_rise,
        end_value=max_value_rise,
        percentage=(min_percentage, max_percentage),
    )


def get_event_halfdecay_time(
    data: np.ndarray, peak_position: int, baseline: float
) -> tuple[int, int]:
    """
    Measure the half-decay position and duration of an event.

    Parameters
    ----------
    data : np.ndarray
        Event snippet.
    peak_position : int
        Peak index within ``data``.
    baseline : float
        Baseline value.

    Returns
    -------
    tuple[int, int]
        Half-decay index within ``data`` and the sample distance from the peak.
    """

    level = baseline + (data[peak_position] - baseline) / 2
    halfdecay_time = int(np.argmax(data[peak_position:] < level))
    halfdecay_position = int(peak_position + halfdecay_time)

    return halfdecay_position, halfdecay_time


def get_event_charge(
    data: np.ndarray,
    start_point: int,
    end_point: int,
    baseline: float,
    sampling: float,
) -> float:
    """
    Integrate event charge between two indices.

    Parameters
    ----------
    data : np.ndarray
        Trace data.
    start_point : int
        Integration start index.
    end_point : int
        Integration end index.
    baseline : float
        Baseline value subtracted before integration.
    sampling : float
        Sampling interval in seconds.

    Returns
    -------
    float
        Integrated charge relative to the supplied baseline.
    """

    integrate_array = (data[start_point:end_point]) - baseline
    # numpy deprecated and later removed np.trapz with v2.0
    try:
        charge = np.trapezoid(integrate_array, dx=sampling)  # type: ignore
    except AttributeError:
        charge = np.trapz(integrate_array, dx=sampling)

    return charge


def get_event_halfwidth(
    event_data: np.ndarray,
    peak_index: int,
    baseline: float,
    amplitude: float,
    sampling_rate: float,
) -> tuple[float, float, float]:
    """
    Measure event half-width and half-amplitude crossing times.

    Parameters
    ----------
    event_data : np.ndarray
        Single-event waveform snippet.
    peak_index : int
        Peak index within ``event_data``.
    baseline : float
        Baseline value for the event.
    amplitude : float
        Peak-to-baseline amplitude.
    sampling_rate : float
        Sampling rate in hertz.

    Returns
    -------
    tuple[float, float, float]
        Half-width, rise-to-half-amplitude time, and decay-to-half-amplitude
        time in seconds. Returns ``(np.nan, np.nan, np.nan)`` when the
        calculation is not possible.
    """

    if (
        peak_index < 0
        or peak_index >= len(event_data)
        or amplitude <= 0
        or sampling_rate <= 0
    ):
        return np.nan, np.nan, np.nan

    half_amp_level = baseline + amplitude / 2.0
    sampling_interval = 1.0 / sampling_rate
    t_rise_half = np.nan
    t_decay_half = np.nan

    # Find rising phase 50% crossing
    # Search from start up to peak_index
    rising_phase_data = event_data[: peak_index + 1]
    # Points strictly below half_amp_level
    points_below_half_amp_rise = np.where(rising_phase_data < half_amp_level)[0]
    # Points at or above half_amp_level
    points_at_or_above_half_amp_rise = np.where(rising_phase_data >= half_amp_level)[0]

    if (
        len(points_below_half_amp_rise) == 0
        or len(points_at_or_above_half_amp_rise) == 0
    ):
        # Data starts at or above half-amp or never crosses it on the rising phase
        pass  # t_rise_half remains np.nan
    else:
        # Last point strictly below half_amp_level
        idx1_rise = points_below_half_amp_rise[-1]
        # First point at or above half_amp_level (must be after idx1_rise)
        valid_crossings_rise = points_at_or_above_half_amp_rise[
            points_at_or_above_half_amp_rise > idx1_rise
        ]
        if len(valid_crossings_rise) == 0:
            pass  # Should not happen if points_below and points_at_or_above are both non-empty and peak is above half-amp
        else:
            idx2_rise = valid_crossings_rise[0]

            if idx2_rise == idx1_rise + 1:  # Ensure points are adjacent
                val1_rise = event_data[idx1_rise]
                val2_rise = event_data[idx2_rise]
                time1_rise = idx1_rise * sampling_interval
                time2_rise = idx2_rise * sampling_interval

                if val2_rise == val1_rise:  # Avoid division by zero if data is flat
                    t_rise_half = (
                        time1_rise if half_amp_level <= val1_rise else time2_rise
                    )
                else:
                    t_rise_half = time1_rise + (time2_rise - time1_rise) * (
                        half_amp_level - val1_rise
                    ) / (val2_rise - val1_rise)
            else:  # No adjacent points found for interpolation (e.g. peak is first point above)
                if (
                    event_data[peak_index] >= half_amp_level
                    and len(points_below_half_amp_rise) > 0
                ):
                    # if peak itself is the first point at or above, and there are points below
                    idx1_rise = points_below_half_amp_rise[-1]
                    idx2_rise = peak_index
                    if (
                        idx2_rise == idx1_rise + 1
                    ):  # if peak is adjacent to the point below
                        val1_rise = event_data[idx1_rise]
                        val2_rise = event_data[idx2_rise]
                        time1_rise = idx1_rise * sampling_interval
                        time2_rise = idx2_rise * sampling_interval
                        if val2_rise == val1_rise:
                            t_rise_half = (
                                time1_rise
                                if half_amp_level <= val1_rise
                                else time2_rise
                            )
                        else:
                            t_rise_half = time1_rise + (time2_rise - time1_rise) * (
                                half_amp_level - val1_rise
                            ) / (val2_rise - val1_rise)

    # Find decaying phase 50% crossing
    # Search from peak_index to end
    decaying_phase_data = event_data[peak_index:]
    # Points at or above half_amp_level in the context of decaying_phase_data indices
    points_at_or_above_half_amp_decay = np.where(decaying_phase_data >= half_amp_level)[
        0
    ]
    # Points strictly below half_amp_level in the context of decaying_phase_data indices
    points_below_half_amp_decay = np.where(decaying_phase_data < half_amp_level)[0]

    if (
        len(points_at_or_above_half_amp_decay) == 0
        or len(points_below_half_amp_decay) == 0
    ):
        # Data ends at or above half-amp or never crosses it on the decaying phase
        pass  # t_decay_half remains np.nan
    else:
        # Last point at or above half_amp_level (relative to peak_index)
        idx1_decay_rel = points_at_or_above_half_amp_decay[-1]
        # First point strictly below half_amp_level (relative to peak_index, must be after idx1_decay_rel)
        valid_crossings_decay = points_below_half_amp_decay[
            points_below_half_amp_decay > idx1_decay_rel
        ]

        if len(valid_crossings_decay) == 0:
            pass
        else:
            idx2_decay_rel = valid_crossings_decay[0]

            # Convert to absolute indices in event_data
            idx1_decay = peak_index + idx1_decay_rel
            idx2_decay = peak_index + idx2_decay_rel

            if idx2_decay == idx1_decay + 1:  # Ensure points are adjacent
                val1_decay = event_data[idx1_decay]
                val2_decay = event_data[idx2_decay]
                time1_decay = idx1_decay * sampling_interval
                time2_decay = idx2_decay * sampling_interval

                if val1_decay == val2_decay:  # Avoid division by zero
                    t_decay_half = (
                        time1_decay if half_amp_level >= val1_decay else time2_decay
                    )
                else:
                    # Interpolate: t = t1 + (t2-t1)*(level-y1)/(y2-y1)
                    # Here, level is half_amp_level, y1 is val1_decay, y2 is val2_decay
                    t_decay_half = time1_decay + (time2_decay - time1_decay) * (
                        half_amp_level - val1_decay
                    ) / (val2_decay - val1_decay)
            else:  # No adjacent points found for interpolation
                # This case implies the data drops below half_amp_level not adjacently after being above it
                pass

    if np.isnan(t_rise_half) or np.isnan(t_decay_half):
        return np.nan, np.nan, np.nan

    half_width = t_decay_half - t_rise_half

    # Ensure half_width is not negative due to edge cases or flat peaks
    if half_width < 0:
        return np.nan, t_rise_half, t_decay_half

    return half_width, t_rise_half, t_decay_half
