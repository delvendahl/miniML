#
# threshold based mEPSC detection
# inspired by Kudoh & Taguchi (2002)
# https://doi.org/10.1016/S0956-5663(02)00053-2
# simplified numpy implementation
# 

import numpy as np
import h5py
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


class DetectionResult(object):
    ''' Collection of results of threshold-based event detection. '''
    def __init__(self, indices, avg, dt, peak_win, threshold, detection_trace):
        self.indices = indices
        '''the indices of detected events'''
        self.avg = avg
        '''the baseline window'''
        self.dt = dt
        '''event detection window'''
        self.peak_win = peak_win
        '''the peak window'''
        self.threshold = threshold
        '''the threshold used for detection'''
        self.detection_trace = detection_trace
        '''the detection trace'''


def threshold_detection(data, sampling, threshold, baseline, dt, peak_win):
    ''' 
    Detect events based on the threshold-based method.
    '''
    nyq = 0.5 * (1 / sampling)
    # high-pass filter data
    sos = butter(4, 1 / nyq, btype='high', output='sos')
    filtered = sosfiltfilt(sos, data)

    # low-pass filter data
    sos = butter(4, 2000 / nyq, btype='low', output='sos')
    filtered = sosfiltfilt(sos, filtered)

    bsl_win = int(baseline / sampling)
    dt_win = int(dt / sampling)
    pk_win = int(peak_win / sampling)
    indices = []

    baseline = np.convolve(filtered, np.ones((bsl_win,))/bsl_win)[(bsl_win-1):]
    smoothed_data = np.convolve(filtered, np.ones((3,))/3)[(3-1):]

    thresholded_data = smoothed_data - np.roll(baseline, (dt_win + bsl_win))

    pos = np.where(thresholded_data < threshold)[0]
    indices = pos[np.where(np.diff(pos, prepend=0) > pk_win)[0]]

    return DetectionResult(indices, baseline, dt, peak_win, threshold, thresholded_data)


if __name__ == '__main__':

    # load data
    filename = '../example_data/gc_mini_trace.h5'
    sampling = 2e-5

    with h5py.File(filename, 'r') as f:
        data = f['mini_data'][:]
    data *= 1e12
    time = np.arange(0, len(data)) * sampling

    matching = threshold_detection(data, sampling, threshold=-5, baseline=0.0008, dt=0.0008, peak_win=0.0015)

    print(len(matching.indices))
    
    plt.plot(time, data)
    plt.plot(time[matching.indices], data[matching.indices], 'o')
    plt.show()
