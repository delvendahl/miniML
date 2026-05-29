#
# template matching for mEPSC detection
# according to Clements & Bekkers (1997)
# https://doi.org/10.1016/S0006-3495(97)78062-7
# implementation by Martin Rückl, Friedrich Johenning, Stephen Lenzi
# https://github.com/samuroi/SamuROI
#

import h5py
import numpy as np
import matplotlib.pyplot as plt


class TemplateMatchResult(object):
    ''' Collection of results of least squares optimization for template matching. '''
    def __init__(self, indices, detection_trace, s, c, threshold, kernel):
        self.indices = indices
        '''the indices of detected events, since zero padding is used for convolution, all indices need to get shifted by N/2 where N is the length of the kernel'''
        self.detection_trace = detection_trace
        '''the criterion vector used for comparison with the threshold'''
        self.s = s
        '''the vector of optimal scaling parameters'''
        self.c = c
        '''the vector of optimal offset parameters'''
        self.threshold = threshold
        '''the threshold used for detection'''
        self.kernel = kernel
        '''the kernel that was used for matching'''


def make_template(t_rise=0.0005, t_decay=0.001, baseline=0.001, duration=0.004, sampling=2e-5) -> np.ndarray:
    '''
    Creates a normalized template for template matching.
    :param t_rise: rise time of the template in milliseconds
    :param t_decay: decay time of the template in milliseconds
    :param baseline: baseline duration of the template in milliseconds
    :param duration: duration of the template in milliseconds
    :param sampling: sampling interval of the template in seconds
    '''
    x = np.arange(0, duration + baseline, sampling)
    a = 1.0
    y = a * (1 - np.exp(-(x - baseline) / t_rise)) * np.exp(-(x - baseline) / t_decay)
    y[x < baseline] = 0

    return y / np.max(y)


def template_matching(data, kernel, threshold):
    """
    .. note::
            Threshold values usually should be in the range 1 to 5 for reasonable results.
    Input :math:`\mathbf{y}` and :math:`\mathbf{e}` are two vectors, the normalized
    template that should be used for matching and the data vector. Some intermediate values are:
    :math:`\overline{e} = \frac{1}{K}\sum_k e_k`
    :math:`\overline{y_n} = \frac{1}{K}\sum_k y_{n+k}`
    The goal is to minimize the least squares distance:
    :math:`\chi_n^2(S,C)=\sum_K\left[y_{n+k} - (S e_k +C)\right]^2`
    W.r.t. the variables :math:`S` and :math:`C`. According to (ClementsBekkers, Appendix  I)[1] the result is:
    :math:`S_n = \frac{\sum_k e_k y_{n+k}-1/K \sum_k e_k \sum_k y_{n+k}}{\sum e_k^2-1/K \sum_k e_k \sum_k e_k} = \frac{\sum_k e_k y_{n+k}-K\overline{e}\ \overline{y_n}}{\sum e_k^2-K\overline{e}^2}`
    and
    :math:`C_n = \overline{y_n} -S_n \overline{e}`
    :param data: 1D numpy array with the timeseries to analyze, above denoted as :math:`\mathbf{y}`
    :param kernel: 1D numpy array with the template to use, above denoted as :math:`\mathbf{e}`
    :param threshold: scalar value usually between 4 to 5.
    :return: A result object :py:class:`template_matching.ClementsBekkersResult`
    [1] http://dx.doi.org/10.1016%2FS0006-3495(97)78062-7
    """
    if len(data) <= len(kernel):
        raise Exception('Data length needs to exceed kernel length.')

    e = kernel[::-1]    # reverse kernel, since we use convolve
    N = len(e)  # the size of the template

    # the sum over e (scalar)
    sum_e = np.sum(e)

    # the sum over e^2 (scalar)
    sum_ee = np.sum(e ** 2)

    # convolution mode
    # mode = 'full' # yields output of length N+M-1
    mode = 'same'  # yields output of length max(M,N)

    # the sum over blocks of y (vector of size N)
    sum_y = np.convolve(data, np.ones_like(e), mode=mode)

    # the sum over blocks of y*y (vector of size N)
    sum_yy = np.convolve(data ** 2, np.ones_like(e), mode=mode)
    # the sum_k  e_k y_{n+k}
    sum_ey = np.convolve(data, e, mode=mode)

    # the optimal scaling factor
    s_n = (sum_ey - sum_e * sum_y / N) / (sum_ee - sum_e * sum_e / N)

    # the optimal offset
    c_n = (sum_y - s_n * sum_e) / N

    # the sum of squared errors when using optimal scaling and offset values
    sse_n = sum_yy + sum_ee * s_n ** 2 + N * c_n ** 2 - 2 * (s_n * sum_ey + c_n * sum_y - s_n * c_n * sum_e)

    # the detection criterion
    crit = s_n / (sse_n / (N - 1)) ** 0.5

    # threshold crossings
    pos = np.where(crit < threshold)[0] if threshold < 0 else np.where(crit > threshold)[0]
    indices = pos[np.where(np.diff(pos, prepend=0) > 1)[0]] - N//2
    indices = indices[np.where(indices > 0)[0]] # Handle negative indices

    from collections import namedtuple
    result = namedtuple('TemplateMatchResult', ['indices', 'detection_trace', 's', 'c', 'threshold', 'kernel'])

    return result(indices=indices, detection_trace=crit, s=s_n, c=c_n, threshold=threshold, kernel=kernel)


if __name__ == '__main__':

    # load data
    filename = '../example_data/gc_mini_trace.h5'
    sampling = 2e-5

    with h5py.File(filename, 'r') as f:
        data = f['mini_data'][:]
    data *= 1e12
    time = np.arange(0, len(data)) * sampling

    matching = template_matching(data, make_template(sampling=sampling), threshold=-4)

    print(len(matching.indices))

    plt.plot(time, data)
    plt.plot(time[matching.indices], data[matching.indices], 'o')
    plt.show()

    plt.plot(time, matching.detection_trace)
    plt.show()
