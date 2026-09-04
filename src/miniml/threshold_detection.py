#
# threshold based mEPSC detection
# inspired by Kudoh & Taguchi (2002)
# https://doi.org/10.1016/S0956-5663(02)00053-2
# simplified numpy implementation
#
from typing import NamedTuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


class DetectionResult(NamedTuple):
    """Collection of results of threshold-based event detection."""

    indices: np.ndarray
    """the indices of detected events"""
    avg: np.ndarray
    """the baseline window"""
    dt: float
    """event detection window"""
    peak_win: float
    """the peak window"""
    threshold: float
    """the threshold used for detection"""
    detection_trace: np.ndarray
    """the detection trace"""

    def __repr__(self):
        return f"DetectionResult(indices={self.indices}, avg={self.avg}, dt={self.dt}, peak_win={self.peak_win}, threshold={self.threshold}, detection_trace={self.detection_trace})"


def threshold_detection(data, sampling, threshold, baseline, dt, peak_win):
    """
    Detect events based on the threshold-based method.
    """
    nyq = 0.5 * (1 / sampling)
    # high-pass filter data
    sos = butter(4, 1 / nyq, btype="high", output="sos")
    filtered = sosfiltfilt(sos, data)

    # low-pass filter data
    sos = butter(4, 2000 / nyq, btype="low", output="sos")
    filtered = sosfiltfilt(sos, filtered)

    bsl_win = int(baseline / sampling)
    dt_win = int(dt / sampling)
    pk_win = int(peak_win / sampling)
    indices = []

    baseline = np.convolve(filtered, np.ones((bsl_win,)) / bsl_win)[(bsl_win - 1) :]
    smoothed_data = np.convolve(filtered, np.ones((3,)) / 3)[(3 - 1) :]

    thresholded_data = smoothed_data - np.roll(baseline, (dt_win + bsl_win))

    pos = np.where(thresholded_data < threshold)[0]
    indices = pos[np.where(np.diff(pos, prepend=0) > pk_win)[0]]

    return DetectionResult(
        indices,
        baseline,
        dt,
        peak_win,
        threshold,
        thresholded_data,
    )
