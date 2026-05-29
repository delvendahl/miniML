#
# deconvolution based mEPSC detection
# as described by Pernía-Andrade et al. (2012)
# https://doi.org/10.1016/j.bpj.2012.08.039
#

import h5py
import numpy as np
from scipy.signal import find_peaks, savgol_filter, butter, sosfiltfilt
from scipy.fft import fft, ifft
from scipy.stats import norm
import matplotlib.pyplot as plt


class DeconvolutionResult(object):
    ''' Collection of results of deconvolution-based event detection. '''
    def __init__(self, indices, threshold, kernel, detection_trace):
        self.indices = indices
        '''the indices of detected events, since zero padding is used for convolution, all indices need to get shifted by N/2 where N is the length of the kernel'''
        self.threshold = threshold
        '''the threshold used for detection'''
        self.kernel = kernel
        '''the kernel that was used for deconvolution'''
        self.detection_trace = detection_trace
        '''the detection trace'''


def make_template(t_rise=0.0005, t_decay=0.001, duration=0.005, sampling=2e-5) -> np.ndarray:
    '''
    Creates a normalized template for deconvolution.
    :param t_rise: rise time of the template in milliseconds
    :param t_decay: decay time of the template in milliseconds
    :param duration: duration of the template in milliseconds
    :param sampling: sampling interval of the template in seconds
    '''
    x = np.arange(0, duration, sampling)
    amp_prime = (t_decay/t_rise)**(t_rise/(t_rise-t_decay))
    y = 1/amp_prime * (-np.exp(-x/t_rise) + np.exp(-x/t_decay))

    return y / -np.max(y)


def deconvolution(data, kernel, threshold, sampling):
    '''
    Performs deconvolution of input data with a pre-defined, normalized template.
    Uses discrete Fourier transformation-based deconvolution. Threshold is defined
    as multiple of standard deviation of the deconvolved trace.
    :param data: input data vector
    :param kernel: normalized template
    :param threshold: threshold for detection
    '''
    nyq = 0.5 * 1/sampling
    sos = butter(4, 1.0 / nyq, btype='high', analog=False, output='sos')
    filtered_data = sosfiltfilt(sos, data[50:-50])

    data_fft = fft(filtered_data, workers=-1)
    tmpl_fft = fft(kernel, n=filtered_data.shape[0], workers=-1)

    # deconvolution
    deconv_dat = np.real(ifft(data_fft * tmpl_fft, workers=-1))

    # z-score result
    deconv_dat = (deconv_dat - np.mean(deconv_dat)) / np.std(deconv_dat)
    deconv_dat = savgol_filter(deconv_dat, 149, 2)

    mu, sigma = norm.fit(deconv_dat)
    detect_threshold = mu + sigma * threshold

    # peak detection in the same manner as template matching.
    pos = np.where(deconv_dat < detect_threshold)[0] if detect_threshold < 0 else np.where(deconv_dat > detect_threshold)[0]
    indices = pos[np.where(np.diff(pos, prepend=0) > 1)[0]] # - N//2
    indices = indices[np.where(indices > 0)[0]] # Handle negative indices

    from collections import namedtuple
    result = namedtuple('DeconvolutionResult', ['indices', 'threshold', 'kernel', 'detection_trace'])

    return result(indices=indices, threshold=detect_threshold, kernel=kernel, detection_trace=deconv_dat)


if __name__ == '__main__':

    # load data
    filename = '../example_data/gc_mini_trace.h5'
    sampling = 2e-5

    with h5py.File(filename, 'r') as f:
        data = f['mini_data'][:]
    data *= 1e12
    time = np.arange(0, len(data)) * sampling

    matching = deconvolution(data, make_template(sampling=sampling), threshold=5, sampling=sampling)

    print(len(matching.indices))

    plt.plot(time, data)
    plt.plot(time[matching.indices], data[matching.indices], 'o')
    plt.show()
