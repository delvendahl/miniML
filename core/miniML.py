from __future__ import annotations
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter, butter, sosfiltfilt, find_peaks
from scipy.ndimage import maximum_filter1d
from scipy.signal import resample
import pickle as pkl
from scipy import signal
from miniML_functions import (get_event_peak, get_event_baseline, get_event_onset, get_event_risetime, 
                              get_event_halfdecay_time, get_event_charge)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#  --------------------------------------------------  #
#  general functions                                    #
def exp_fit(x: np.ndarray, amp: float, tau: float, offset: float) -> np.ndarray:
    """
    Fits an exponential curve to the given data.

    Parameters:
        x (np.ndarray): The input data.
        amp (float): The amplitude of the exponential curve.
        tau (float): The time constant of the exponential curve.
        offset (float): The offset of the exponential curve.

    Returns:
        np.ndarray: The fitted exponential curve.
    """

    return amp * np.exp(-(x - x[0]) / tau) + offset


def mEPSC_template(x: np.ndarray, a: float, t_rise: float, t_decay: float, x0: float) -> np.ndarray:
    """
    Generates a template miniature excitatory postsynaptic current (mEPSC) based on the given parameters.

    Parameters:
        x (np.ndarray): An array of x values.
        a (float): The amplitude of the mEPSCs.
        t_rise (float): The rise time constant of the mEPSCs.
        t_decay (float): The decay time constant of the mEPSCs.
        x0 (float): The onset time point for the mEPSCs.

    Returns:
        np.ndarray: An array of y values representing an mEPSC template.

    Note:
        - The formula used to calculate the template is y = a * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay).
        - Any values of x that are less than x0 will be set to 0 in the resulting array.
    """
    y = a * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay)
    y[x < x0] = 0

    return y


def lowpass_filter(data: np.ndarray, sampling_rate: float, lowpass: float=500, order: int=4) -> np.ndarray:
    """
    Apply a lowpass filter to the input data.

    Parameters:
        data (np.ndarray): The input data to be filtered.
        sampling_rate (float): The sampling rate of the input data.
        lowpass (float, optional): The cutoff frequency for the lowpass filter. Defaults to 500.
        order (int, optional): The order of the filter. Defaults to 4.

    Returns:
        np.ndarray: The filtered data.

    """
    nyq = sampling_rate * 0.5
    sos = butter(order, lowpass / nyq, btype='low', output='sos')

    return sosfiltfilt(sos, data)


@tf.function
def minmax_scaling(x: tf.Tensor) -> tf.Tensor:
    """
    Applies min-max scaling to the input tensor.

    Args:
        x (tf.Tensor): The input tensor to be scaled.

    Returns:
        tf.Tensor: The scaled tensor.
    """
    x_min = tf.expand_dims(tf.math.reduce_min(x), axis=-1)
    x_max = tf.expand_dims(tf.math.reduce_max(x), axis=-1)

    return tf.math.divide(tf.math.subtract(x, x_min), tf.math.subtract(x_max, x_min))


#  --------------------------------------------------  #
#  miniML classes                                      #
class MiniTrace():
    '''miniML class for a time series data trace containing synaptic events. Data are stored as float64 numpy ndarray.

    Parameters
    ----------
    data: np.ndarray | list, default=[]
        The data to be analysed.
    sampling_interval: float, default=1
        The sampling interval of the data in seconds.
    y_unit: str, default=''
        The physical unit of the y-axis.
    filename: str, default=''
        The filename of the trace.

    Attributes
    ----------
    events: np.ndarray
        Detected events as 2d array.
    '''
    def __init__(self, data: np.ndarray | list=None, sampling_interval: float=1, y_unit: str='', filename: str='') -> None:
        self.data = data
        self.sampling = sampling_interval
        self.events = []
        self.y_unit = y_unit
        self.filename = filename

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data) -> None:
        # ensure data is float64 to avoid issues with minmax_scale
        self._data = data.astype(np.float64)

    @property
    def sampling(self) -> float:
        return self._sampling

    @sampling.setter
    def sampling(self, value) -> None:
        if value < 0:
            raise ValueError('Sampling interval must be positive')
        self._sampling = value

    @property
    def sampling_rate(self) -> float:
        return np.round(1/self.sampling)

    @property
    def time_axis(self) -> np.ndarray:
        ''' Returns time axis as numpy array '''
        return np.arange(0, len(self.data)) * self.sampling

    @property
    def total_time(self) -> float:
        ''' Returns the total duration of the recording '''
        return len(self.data) * self.sampling

    @classmethod
    def from_h5_file(cls, filename: str, tracename: str='mini_data', scaling: float=1e12, 
                     sampling: float=2e-5, unit: str='pA') -> MiniTrace:
        ''' Loads data from an hdf5 file. Name of the dataset needs to be specified.

        Parameters
        ----------
        filename: str
            Path of the .h5 file to load.
        tracename: str, default='mini_data'
            Name of the dataset in the file to be loaded.
        scaling: float, default=1e12
            Scaling factor applied to the data. Defaults to 1e12 (i.e. pA)
        sampling: float, default=2e-5
            The sampling interval of the data in seconds. Defaults to 20 microseconds (i.e. 50kHz sampling rate).
        unit: string, default='pA'
            Data unit string after scaling. Used for display purposes.

        Returns
        -------
        MiniTrace
            An initialized MiniTrace object.

        Raises  
        ------
        FileNotFoundError
            When the specified file does not exist.
        '''
        with h5py.File(filename, 'r') as f:
            path = f.visit(lambda key : key if isinstance(f[key], h5py.Dataset) and key.split('/')[-1] == tracename else None)
            if path is None:
                raise FileNotFoundError('Trace not found in file')
            data = f[path][:] * scaling

        print(f'Data loaded from {filename} with shape {data.shape}')

        return cls(data=data, sampling_interval=sampling, y_unit=unit, filename=os.path.split(filename)[-1])

    @classmethod
    def from_heka_file(cls, filename: str, rectype: str, group: int=1, exclude_series:list=[], exclude_sweeps:dict={},
                        scaling: float=1e12, unit: str=None, resample: bool=True) -> MiniTrace:
        ''' Loads data from a HEKA .dat file. Name of the PGF sequence needs to be specified.

        Parameters
        ----------
        filename: string
            Path of a .dat file.
        rectype: string
            Name of the PGF sequence in the file to be loaded.
        group: int, default=1
            HEKA group to load data from. HEKA groups are numbered starting from 1. Defaults to 1. 
        exclude_series: list, default=[].
            List of HEKA series to exclude.
        exclude_sweeps: dict, default={}.
            Dictionary with sweeps to exclude from analysis. E.g. {2 : [4, 5]} excludes sweeps 4 & 5 from series 2.
        scaling: float, default=1e12
            Scaling factor applied to the data. Defaults to 1e12 (i.e. pA)
        unit: str, default=''
            Data unit, to be set when using scaling factor.
        resample: boolean, defaulT=rue
            Resample data in case of sampling rate mismatch.

        Returns
        -------
        MiniTrace
            An initialized MiniTrace object.
        Raises
        ------
        Exception or ValueError
            If the file is not a valid .dat file.
        ValueError
            When the sampling rates of different series mismatch and resampling is set to False.
        '''
        if not os.path.splitext(filename)[-1].lower() == '.dat':
            raise Exception('Incompatible file type. Method only loads .dat files.')

        import FileImport.HekaReader as heka
        bundle = heka.Bundle(filename)

        group = group - 1
        if group < 0 or group > len(bundle.pul.children) - 1:
            raise IndexError('Group index out of range')

        bundle_series = dict()
        for i, SeriesRecord in enumerate(bundle.pul[group].children):
            bundle_series.update({i: SeriesRecord.Label})

        series = [series_number for series_number, record_type in bundle_series.items() \
                  if record_type == rectype and series_number not in exclude_series]
        
        series_data = []
        series_resistances = []
        for i in series:
            sweep_data = []
            for j in range(bundle.pul[group][i].NumberSweeps):
                if i not in exclude_sweeps:
                    try:
                        sweep_data.append(bundle.data[group, i, j, 0])   
                    except IndexError as e:
                        pass
                else:
                    if j not in exclude_sweeps[int(i)]:
                        try:
                            sweep_data.append(bundle.data[group, i, j, 0])   
                        except IndexError as e:
                            pass
            series_data.append((np.array(sweep_data).flatten(), bundle.pgf[i].SampleInterval))
            series_resistances.append((1/bundle.pul[group][i][0][0].GSeries)*1e-6)

        max_sampling_interval = max([el[1] for el in series_data])
        data = np.array([], dtype=np.float64)
        for i, dat in enumerate(series_data):
            if dat[1] < max_sampling_interval:
                if not resample:
                    raise ValueError(f'Sampling interval of series {i} is smaller than maximum sampling interval of all series')
                step = int(max_sampling_interval / dat[1])
                data = np.append(data, dat[0][::step])
            else:
                data = np.append(data, dat[0])
        
        data_unit = unit if unit is not None else bundle.pul[group][series[0]][0][0].YUnit

        MiniTrace.excluded_sweeps = exclude_sweeps
        MiniTrace.exlucded_series = exclude_series
        MiniTrace.Rseries = series_resistances

        return cls(data=data * scaling, sampling_interval=max_sampling_interval, 
                   y_unit=data_unit, filename=os.path.split(filename)[-1])


    @classmethod
    def from_axon_file(cls, filepath: str, channel: int=0, scaling: float=1.0, unit: str=None) -> MiniTrace:
        ''' Loads data from an AXON .abf file.

        Parameters
        ----------
        filepath: string
            Path of a .abf file.
        channel: int, default: 0
            The recording channel to load
        scaling: float, default=1.0
            Scaling factor applied to the data.
        unit: str, default=''
            Data unit, to be set when using scaling factor.
        Returns
        -------
        MiniTrace
            An initialized MiniTrace object.
        Raises
        ------
        Exception
            If the file is not a valid .abf file.
        IndexError
            When the selected channel does not exist in the file.
        '''
        if not os.path.splitext(filepath)[-1].lower() == '.abf':
            raise Exception('Incompatible file type. Method only loads .abf files.')

        import pyabf
        abf_file = pyabf.ABF(filepath)
        if channel not in abf_file.channelList:
            raise IndexError('Selected channel does not exist.')

        data_unit = unit if unit is not None else abf_file.adcUnits[channel]

        return cls(data=abf_file.data[channel] * scaling, sampling_interval=1/abf_file.sampleRate, 
                    y_unit=data_unit, filename=os.path.split(filepath)[-1])


    def plot_trace(self) -> None:
        ''' Plots the trace '''
        plt.plot(self.time_axis, self.data)
        plt.xlabel('Time [s]')
        plt.ylabel(f'[{self.y_unit}]')
        plt.show()


    def detrend(self, detrend_type: str='linear', num_segments: int=0) -> MiniTrace:
        ''' Detrend the data. '''
        from scipy.signal import detrend
        num_data = self.trace.data.shape[0]
        breaks = np.arange(num_data/num_segments, num_data, num_data/num_segments, dtype=np.int64) if num_segments > 1 else 0
        detrended = detrend(self.trace.data, bp=breaks, type=detrend_type)

        return MiniTrace(detrended, self.sampling, y_unit=self.y_unit, filename=self.filename)


    def filter(self, notch: float=None, highpass: float=None, lowpass: float=None, order: int=4,
               savgol: float=None) -> MiniTrace:
        ''' Filters trace with a combination of notch, high- and lowpass filters.
        If both lowpass and savgol arguments are passed, only the lowpass filter is applied. 
        notch: float, default=None
            Notch filter frequency (Hz)
        highpass: float, default=None
            Highpass cutoff frequency (Hz).
        lowpass: float, default=None
            Lowpass cutoff frequency (Hz). Set to None to turn filtering off.
        order: int, default=4
            Order of the filter.
        savgol: float, default=None
            The time window for Savitzky-Golay smoothing (ms).
            
        returns: MiniTrace
            A filtered MiniTrace object.
        '''
        from scipy.signal import iirnotch, cheby2, filtfilt, sosfilt
        filtered_data = self.data.copy()
        nyq = 0.5 * self.sampling_rate

        if notch:
            b_notch, a_notch = iirnotch(notch, 2.0, self.sampling_rate)
            filtered_data = filtfilt(b_notch, a_notch, filtered_data)
        if highpass:
            sos = butter(order, highpass / nyq, btype='high', output='sos')
            filtered_data = sosfilt(sos, filtered_data)
        if lowpass:
            if savgol:
                print('Warning: Two lowpass filteres selected, Savgol filter is ignored.')
            sos = cheby2(order, 60, lowpass / nyq, btype='low', analog=False, output='sos', fs=None)
            filtered_data = sosfiltfilt(sos, filtered_data)
        elif savgol:
            filtered_data = savgol_filter(filtered_data, int(savgol/1000/self.sampling), polyorder=order)

        return MiniTrace(filtered_data, sampling_interval=self.sampling, y_unit=self.y_unit, filename=self.filename)


    def resample(self, sampling_frequency: float=None) -> MiniTrace:
        ''' Resamples the data trace to the given frequency 
        
        sampling_frequency: float
            Sampling frequency in Hz of the output data
            
        returns: MiniTrace
            A resampled MiniTrace object
        '''
        if sampling_frequency is None:
            return self

        resampling_factor = np.round(self.sampling_rate / sampling_frequency, 2)
        resampled_data = resample(self.data, int(self.data.shape[0]/resampling_factor))
        new_sampling_interval = self.sampling * resampling_factor

        return MiniTrace(resampled_data, sampling_interval=new_sampling_interval, y_unit=self.y_unit, filename=self.filename)


    def _extract_event_data(self, positions: np.ndarray, before: int, after: int) -> np.ndarray:
        '''
        Extracts events from trace

        Parameters
        ------
        positions: np.ndarray
            The event positions.
        before: int
            Number of samples before event position for event extraction. Positions-before must be positive.
        after: int
            Number of samples after event positions for event extraction. Positions+after must smaller 
            than total number of samples in self.data.
        returns: np.ndarray
            2d array with events of shape (len(positions), before+after).
        
        Raises
        ------
        ValueError
            When the indices are too close to self.data boundaries
        '''
        if np.any(positions - before < 0) or np.any(positions + after >= self.data.shape[0]):
            raise ValueError('Cannot extract time windows exceeding input data size.')

        indices = positions + np.arange(-before, after)[:, None, None]

        return np.squeeze(self.data[indices].T, axis=1)


class EventStats():
    '''miniML class for event statistics.
    Parameters
    ----------
    amplitudes: np.ndarray
        Amplitudes of individual events.
    scores: np.ndarray
        Prediction scores of individual events.
    charges: np.ndarray
        Charge transfer of individual events.
    risetimes: np.ndarray
        10-90 percent rise times of events.
    halfdecays: np.ndarray
        Half decay times of events.
    avg_tau_decay: float
        Average decay time constant (seconds).
    rec_time: float
        Total recording duration (seconds).
    y_unit: str
        Data unit.
    Attributes
    ----------
    event_count: number of events
    '''
    def __init__(self, amplitudes, scores, charges, risetimes, decaytimes, tau, time, unit: str) -> None:
        self.amplitudes = amplitudes
        self.event_scores = scores
        self.charges = charges
        self.risetimes = risetimes
        self.halfdecays = decaytimes
        self.avg_tau_decay = tau
        self.rec_time = time
        self.y_unit = unit
        self.event_count = len(self.amplitudes)

    def mean(self, values: np.ndarray) -> float:
        ''' Returns mean of event parameter '''
        if ~np.all(np.isnan(values)) and self.event_count:
            return np.nanmean(values)
        else:
            return np.nan

    def std(self, values: np.ndarray) -> float:
        ''' Returns standard deviation of event parameter '''
        return np.nanstd(values, ddof=1) if values.shape[0] > 1 else np.nan

    def median(self, values: np.ndarray) -> float:
        ''' Returns median of event parameter '''
        return np.median(values)

    def cv(self, values: np.ndarray) -> float:
        ''' Returns coefficient of variation of event parameter '''
        return abs(self.std(values) / self.mean(values))

    def frequency(self) -> float:
        ''' Returns frequency of events '''
        return len(self.amplitudes) / self.rec_time


    def print(self) -> None:
        ''' Prints event statistics to stdout '''
        print('\nEvent statistics:\n-------------------------')
        print(f'    Number of events: {self.event_count}')
        print(f'    Average score: {self.mean(self.event_scores):.3f}')
        print(f'    Event frequency: {self.frequency():.4f} Hz')
        print(f'    Mean amplitude: {self.mean(self.amplitudes):.4f} {self.y_unit}')
        print(f'    Median amplitude: {self.median(self.amplitudes):.4f} {self.y_unit}')
        print(f'    Std amplitude: {self.std(self.amplitudes):.4f} {self.y_unit}')
        print(f'    CV amplitude: {self.cv(self.amplitudes):.3f}')
        print(f'    Mean charge: {self.mean(self.charges):.5f} pC')
        print(f'    CV charge: {self.cv(self.charges):.3f}')
        print(f'    Mean 10-90 risetime: {self.mean(self.risetimes)*1000:.3f} ms')
        print(f'    Mean half decay time: {self.mean(self.halfdecays)*1000:.3f} ms')
        print(f'    Tau decay: {self.avg_tau_decay*1000:.3f} ms')
        print('-------------------------')



class EventDetection():
    '''miniML main class with methods for event detection and -analysis.
    Parameters
    ----------
    data: miniML MiniTrace object
        The data trace to be analysed.
    window_size: int, default=600
        The window size for the event detection (samples per event window).
    event_direction: str, default='negative'
        Event direction in data. Should be 'negative' or any other string for positive events.
    training_direction: str, default='negative'
        Event direction during training. Should be 'negative' or 'positive'. All provided GitHub
        models were trained with negative events (improved TL performance). If a model is trained
        with positive events, this needs to be specified to run inference.
    batch_size: int, default=128
        The batch size for the event detection (used in model.predict).
    model_path: str, default=''
        The path of the model file (.h5) to be used for event detection.
    model_threshold: float, default=None
        The threshold for the model; range=(0,1).
    compile_model: bool, default=False
        Whether to compile the model.
    callbacks: list, default=[]
        List of callback function(s) to be used during event detection.
    Attributes
    ----------
    event_locations: np.ndarray
        The individual event locations
    event_scores: np.ndarray
        The individual prediction scores of events
    event_peak_locations: np.ndarray
        The individual event peak locations in samples
    event_peak_times: np.ndarray
        The individual event peak times
    events: np.ndarray
        The events as 2d array
    event_stats: EventStats object
        Contains event statistics
    '''
    def __init__(self, data: MiniTrace, window_size: int=600, event_direction: str='negative', training_direction: str='negative',
                 batch_size: int=128, model_path: str='', model_threshold: float=None, compile_model=True, callbacks: list=[]) -> None:
        self.trace = data
        self.prediction = None
        self.window_size = window_size
        self.event_direction = event_direction
        self.training_direction = training_direction
        self.event_locations = np.array([])
        self.event_scores = np.array([])
        self.events = np.array([])
        self.batch_size = batch_size
        self.model_path = model_path
        self.model = None
        self.model_threshold = None
        if model_path and model_threshold:
            self.load_model(model_path, model_threshold, compile=compile_model)
            self.callbacks = callbacks

    @property
    def event_direction(self):
        return self._event_direction

    @event_direction.setter
    def event_direction(self, event_direction_str: str):
        self._event_direction = -1 if event_direction_str.lower() == 'negative' else 1

    @property
    def training_direction(self):
        return self._training_direction

    @training_direction.setter
    def training_direction(self, training_direction_str: str):
        self._training_direction = -1 if training_direction_str.lower() == 'negative' else 1


    def _init_arrays(self, attr_names, shape, dtype):
        ''' initialize multiple 1d ndarrays with given shape containing NaNs '''
        for label in attr_names:
            value = -1 if 'int' in str(dtype) else np.NaN
            setattr(self, str(label), np.full(int(shape), value, dtype=dtype))


    def events_present(self) -> bool:
        ''' Checks if events are present '''
        num_events = self.events.shape[0]
        
        return num_events != 0


    def load_model(self, filepath: str, threshold: float=0.5, compile=True) -> None:
        ''' Loads trained miniML model from hdf5 file '''
        self.model = tf.keras.models.load_model(filepath, compile=compile)
        self.model_threshold = threshold
        print(f'Model loaded from {filepath}')


    def __predict(self, stride: int) -> None:
        '''
        Performs prediction on a data trace using a sliding window of size `window_size` with a stride size given by `stride`.
        The prediction is performed on the data using the miniML model.
        Speed of prediction depends on batch size of model.predict(), but too high batch sizes will give low precision results.
        stride: int
            Stride length in samples for prediction. Must be a positive integer smaller than the window size.
        Raises  
            ValueError when stride is below 1 or above window length
        '''

        # resample values for prediction:
        trace = resample(self.trace.data, int(len(self.trace.data)*self.resampling_factor))

        # invert the trace if event_direction and training_direction are different.
        if self.event_direction != self.training_direction:
            trace *= -1

        win_size = self.window_size*self.resampling_factor
        stride = self.stride_length*self.resampling_factor

        if stride <= 0 or stride > self.window_size:
            raise ValueError('Invalid stride')
        
        ds = tf.keras.utils.timeseries_dataset_from_array(data=np.expand_dims(trace, axis=1).astype(np.float32), 
                                                          targets=None, 
                                                          sequence_length=win_size, 
                                                          sequence_stride=stride,
                                                          batch_size=None,
                                                          shuffle=False)
        
        ds = ds.map(minmax_scaling, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        self.prediction = tf.squeeze(self.model.predict(ds, verbose=1, callbacks=self.callbacks))


    def _find_event_locations(self, limit: int, rel_prom_cutoff: float=0.25, peak_w:int=10):
        '''
        Find approximate event positions based on negative threshold crossings in prediction trace. Extract
        segment of peak windows in prediction trace and search for peaks in first derivative. If no peak is found,
        the maximum first derivate is used as peak localization.
        Returns trace indices and scores of events            
        limit: int
            Right trace limit to make sure events at the very border are not picked up.
            Prevents problems with downstream analysis.
        rel_prom_cutoff: float
            Relative prominence cutoff. Determines the minimum relative prominence for detection of overlapping events
        peak_w: int
            Minimum peak width for detection peaks to be accepted
        '''

        win = signal.windows.hann(self.convolve_win)
        
        # set all values for resampling traces
        data_trace = resample(self.trace.data, int(len(self.trace.data)*self.resampling_factor))
        data_trace *= self.event_direction # (-1 = 'negative', 1 else)
        
        win_size = int(self.window_size*self.resampling_factor)
        stride = int(self.stride_length*self.resampling_factor)
        sampling = self.trace.sampling/self.resampling_factor
        add_points = int(win_size/3)
        limit=win_size + add_points

        filtered_prediction = maximum_filter1d(self.prediction, size=5, origin=-2)
        _, peak_properties = find_peaks(x=filtered_prediction, height=self.model_threshold, 
                                        prominence=self.model_threshold, width=peak_w)

        start_pnts = np.array(peak_properties['left_ips'] * stride + win_size/4, dtype=np.int64)
        end_pnts =  np.array(peak_properties['right_ips'] * stride + win_size/2, dtype=np.int64)


        # filter raw data trace, calculate gradient and filter first derivative trace        
        trace_convolved = signal.convolve(data_trace-np.mean(data_trace), win, mode='same') / sum(win)
        gradient = np.gradient(trace_convolved, sampling)
        smth_gradient = signal.convolve(gradient-np.mean(gradient), win, mode='same') / sum(win)

        # get threshold based on standard deviation of the derivative of event-free data sections
        split_data = np.split(smth_gradient, np.vstack((start_pnts, end_pnts)).ravel('F'))
        event_free_data = np.concatenate(split_data[::2]).ravel()
        threshold = int(4 * np.std(event_free_data))

        event_locations, event_scores = [], []

        for i, position in enumerate(peak_properties['right_ips'] * stride): 
            if position < win_size:
                continue
            peaks, peak_params = find_peaks(x=smth_gradient[start_pnts[i]:end_pnts[i]], height=threshold, 
                                prominence=threshold)
            
            if peaks.shape[0] > 1: # If > 1 peak found; apply relative prominence cutoff of .25
                rel_prom = peak_params['prominences']/np.max(peak_params['prominences'])
                inds = np.where(rel_prom >= rel_prom_cutoff)[0]            
                peaks = peaks[inds]
                for my_param in peak_params:
                    peak_params[my_param] = peak_params[my_param][inds]
            
            if not len(peaks): # If no peak found: default argmax finding
                peaks = np.array([np.argmax(smth_gradient[start_pnts[i]:end_pnts[i]])])

            for peak in peaks:
                if (start_pnts[i] + peak) >= (data_trace.shape[0] - limit):
                    continue
                if start_pnts[i] + peak not in event_locations:
                    event_locations.append(start_pnts[i] + peak)
                    event_scores.append(peak_properties['peak_heights'][i])

        ### Check for duplicates:
        if np.array(event_locations).shape[0] != np.unique(np.array(event_locations)).shape[0]:
            print('removed duplicates')

        event_locations, event_scores = np.array(event_locations), np.array(event_scores)        
        unique_indices = np.unique(event_locations, return_index=True)[1]

        event_locations = event_locations[unique_indices]
        event_scores, event_scores[unique_indices]

        num_locations = event_locations.shape[0]
        
        remove = []
        for ind, i in enumerate(event_locations):
            close = event_locations[np.isclose(i, event_locations, atol=win_size/100)]
            if close.shape[0] > 1:
                for ind, removal in enumerate(close):
                    if ind > 0:
                        remove.append(removal)
        remove = np.unique(np.array(remove))
        for i in remove:
            remaining_indices = np.where(event_locations != i)[0]
            event_locations = event_locations[remaining_indices]
            event_scores = event_scores[remaining_indices]
        
        if event_locations.shape[0] != num_locations:
            print('removed event locations via atol criterium')
        
        event_locations = (event_locations / self.resampling_factor).astype(int)

        return np.asarray(event_locations, dtype=np.int64), event_scores


    def _get_event_properties(self, filter: bool=True) -> dict:
        '''
        Find more detailed event location properties required for analysis. Namely, baseline, event onset,
        peak half-decay and 10 & 90% rise positions. Also extracts the actual event properties, such as
        amplitude or half-decay time.
        filter: bool
            If true, properties are extracted from the filtered data.
        '''
        ### Prepare data
        diffs = np.diff(self.event_locations, append=self.trace.data.shape[0]) # Difference in points between the event locations
        add_points = int(self.window_size/3)
        after=self.window_size + add_points
        positions = self.event_locations
        
        ### Set parameters for charge calculation
        factor_charge = 4
        num_combined_charge_events = 1
        calculate_charge = False # will be set to True in the loop if double event criteria are fulfilled; not a flag for charge 

        if np.any(positions - add_points < 0) or np.any(positions + after >= self.trace.data.shape[0]):
            raise ValueError('Cannot extract time windows exceeding input data size.')

        if filter:
            win = signal.windows.hann(int(self.convolve_win/self.resampling_factor))
            mini_trace = signal.convolve(self.trace.data, win, mode='same') / sum(win)
        else:
            mini_trace = self.trace.data

        mini_trace *= self.event_direction   

        self._init_arrays(['event_peak_locations', 'event_start', 'min_positions_rise', 'max_positions_rise'], positions.shape[0], dtype='np.int64')
        self._init_arrays(['event_peak_values', 'event_bsls', 'decaytimes', 'charges', 'risetimes', 'half_decay'], positions.shape[0], dtype='np.float64')               

        for ix, position in enumerate(positions):
            indices = position + np.arange(-add_points, after)
            data = mini_trace[indices]
            if filter:
                data_unfiltered = self.trace.data[indices]*self.event_direction
            else:
                data_unfiltered = data
            
            event_peak = get_event_peak(data=data,event_num=ix,add_points=add_points,window_size=self.window_size,diffs=diffs)
            self.event_peak_locations[ix] = int(event_peak)
            self.event_peak_values[ix] = data[event_peak]
            peak_spacer = int(self.window_size/100)
            self.event_peak_values[ix] = np.mean(data_unfiltered[int(event_peak-peak_spacer):int(event_peak+peak_spacer)])
            
            baseline, baseline_var = get_event_baseline(data=data,event_num=ix,diffs=diffs,add_points=add_points,peak_positions=self.event_peak_locations,positions=positions)
            self.event_bsls[ix] = baseline
            
            onset_position = get_event_onset(data=data,peak_position=event_peak,baseline=baseline,baseline_var=baseline_var)
            self.event_start[ix] = onset_position
            
            risetime, min_position_rise, max_position_rise = get_event_risetime(data=data,peak_position=event_peak,onset_position=onset_position)
            self.risetimes[ix] = risetime
            self.min_positions_rise[ix] = min_position_rise
            self.max_positions_rise[ix] = max_position_rise

            level = baseline + (data[event_peak] - baseline) / 2
            if diffs[ix] < add_points: # next event close; check if we can get halfdecay
                right_lim = diffs[ix]+add_points # Right limit is the onset of the next event
                test_arr =  data[event_peak:right_lim]
                if test_arr[test_arr<level].shape[0]: # means that event goes below 50% ampliude before max rise of the next event; 1/2 decay can be calculated
                    halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data[0:right_lim],peak_position=event_peak,baseline=baseline)
                else:
                    halfdecay_position, halfdecay_time = np.nan, np.nan
            else:  
                halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data, peak_position=event_peak, baseline=baseline)

            self.half_decay[ix] = halfdecay_position
            self.decaytimes[ix] = halfdecay_time
            
            # calculate charges
            ### For charge; multiple event check done outside function.
            if ix < positions.shape[0]-1:
                if num_combined_charge_events == 1: # define onset position for charge calculation
                    onset_in_trace = positions[ix] - (add_points-self.event_start[ix])
                    baseline_for_charge = self.event_bsls[ix]

                if np.isnan(self.half_decay[ix]):
                    num_combined_charge_events += 1
                
                else:
                    ### Get distance from peak to next event location.
                    peak_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points)
                    next_event_location = positions[ix+1]
                    delta_peak_location = next_event_location - peak_in_trace
                    
                    # determine end of area calculation based on event decay
                    endpoint = int(self.event_peak_locations[ix] + factor_charge*(int(self.half_decay[ix]) - self.event_peak_locations[ix]))
                    delta_peak_endpoint = endpoint-self.event_peak_locations[ix]

                    if delta_peak_location > delta_peak_endpoint: # Next event_location further away than the charge window; calculate charge
                        calculate_charge = True
                    else:
                        num_combined_charge_events += 1

                if calculate_charge:
                    endpoint_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points) + delta_peak_endpoint
                    charge = get_event_charge(trace_data=mini_trace, start_point=onset_in_trace, end_point=endpoint_in_trace, baseline=baseline_for_charge, sampling=self.trace.sampling)
                    
            else: # Handle the last event
                if num_combined_charge_events == 1: # define onset position for charge calculation
                    onset_in_trace = positions[ix] - (add_points-self.event_start[ix])
                    baseline_for_charge = self.event_bsls[ix]
                
                peak_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points)
                endpoint = int(self.event_peak_locations[ix] + factor_charge*(int(self.half_decay[ix]) - self.event_peak_locations[ix]))
                delta_peak_endpoint = endpoint-self.event_peak_locations[ix]
                endpoint_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points) + delta_peak_endpoint
                
                if endpoint_in_trace > mini_trace.shape[0]:
                    endpoint_in_trace = mini_trace.shape[0]

                charge = get_event_charge(trace_data=mini_trace, start_point=onset_in_trace, end_point=endpoint_in_trace, baseline=baseline_for_charge, sampling=self.trace.sampling)
                calculate_charge = True
            if calculate_charge: # Charge was caclulated; check how many potentially overlapping events contributed.
                charge = [charge/num_combined_charge_events]*num_combined_charge_events
                for ix_adjuster in range(len(charge)):
                    self.charges[ix-ix_adjuster] = charge[ix_adjuster]
                            
                # Reset values after calculation
                calculate_charge = False
                num_combined_charge_events = 1

        ## Convert units
        self.event_peak_values *= self.event_direction
        self.event_bsls *= self.event_direction
        self.risetimes *= self.trace.sampling
        self.decaytimes *= self.trace.sampling
        self.charges *= self.event_direction

        ## map indices back to original trace
        for ix, position in enumerate(positions):
            self.event_peak_locations[ix] = int(self.event_peak_locations[ix] + self.event_locations[ix] - self.add_points)
            self.event_start[ix] = int(self.event_start[ix] + self.event_locations[ix] - self.add_points)
            self.min_positions_rise[ix] = int(min_position_rise + self.event_locations[ix] - self.add_points)
            self.max_positions_rise[ix] = int(max_position_rise + self.event_locations[ix] - self.add_points)            
            
            if not np.isnan(self.half_decay[ix]):
                self.half_decay[ix] = int(self.half_decay[ix] + self.event_locations[ix] - self.add_points)


    def _get_average_event_properties(self) -> dict:
        '''extracts event properties for the event average the same way the individual events are analysed'''
        ### Prepare data
        diffs = [int(self.window_size / 3) * 10] * 3 # Difference in points between the event locations
        add_points = int(self.window_size / 3)

        ### Set parameters for charge calculation
        factor_charge = 4
        data = np.mean(self.events, axis=0) * self.event_direction

        event_peak = get_event_peak(data=data,event_num=0,add_points=add_points,window_size=self.window_size,diffs=diffs)
        event_peak_value = data[event_peak]

        baseline, baseline_var = get_event_baseline(data=data,event_num=0,diffs=diffs,add_points=add_points,peak_positions=[event_peak],positions=[add_points])
        onset_position = get_event_onset(data=data,peak_position=event_peak,baseline=baseline,baseline_var=baseline_var)
        risetime, min_position_rise, max_position_rise = get_event_risetime(data=data,peak_position=event_peak,onset_position=onset_position)

        halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data,peak_position=event_peak, baseline=baseline)        
        endpoint = int(event_peak + factor_charge*halfdecay_position)
        charge = get_event_charge(trace_data=data, start_point=onset_position, end_point=endpoint, baseline=baseline, sampling=self.trace.sampling)

        ## Convert units back.
        event_peak_value = event_peak_value - baseline
        event_peak_value *= self.event_direction
        baseline *= self.event_direction
        risetime *= self.trace.sampling
        halfdecay_time *= self.trace.sampling
        charge *= self.event_direction
        
        results = {
            'amplitude':event_peak_value,
            'baseline':baseline,
            'risetime':risetime,
            'halfdecay_time':halfdecay_time,
            'charge':charge,
            'event_peak':event_peak,
            'onset_position':onset_position,
            'min_position_rise':min_position_rise,
            'max_position_rise':max_position_rise,
            'halfdecay_position':halfdecay_position,
            'endpoint_charge':endpoint
            }
        return results


    def detect_events(self, stride: int=None, eval: bool=False, verbose: bool=True, peak_w:int=5,
                      rel_prom_cutoff: float=0.25, convolve_win: int=20, resample_to_600: bool=True) -> None:
        '''
        Wrapper function to perform event detection, extraction and analysis
        
        Parameters
        ----------
        stride: int, default = None
            The stride used during prediction. If not specified, it will be set to 1/30 of the window size
        eval: bool, default = False
            Whether to evaluate detected events.
        verbose: bool, default = True
            Whether to print the output. 
        peak_w: int, default = 5
            The minimum prediction peak width.
        rel_prom_cutoff: int, float = 0.25
            The relative prominence cutoff. Overlapping events are separated based on a peak-finding in the first derivative. To be considered
            an event, any detected peak must have at least 25% prominence of the largest detected prominence.
        convolve_win: int, default = 20
            Window size for the hanning window used to filter the data and derivative for event analysis.
        resample_to_600: bool, default = True
            Whether to resample the the data to match a 600 point window. Should always be true, unless a model was trained with a different window size.
        '''
        if resample_to_600:
            self.resampling_factor = 600/self.window_size
        else:
            self.resampling_factor = 1
        self.peak_w = peak_w
        self.rel_prom_cutoff = rel_prom_cutoff
        self.convolve_win = convolve_win
        self.add_points = int(self.window_size/3)
        if stride is None:
            self.stride_length = int(self.window_size/30)
        else:
            self.stride_length = stride

        self.__predict(stride)
        self.event_locations, self.event_scores = self._find_event_locations(limit=self.window_size + self.add_points, rel_prom_cutoff=rel_prom_cutoff, peak_w=peak_w)

        if self.event_locations.shape[0] > 0:
            self.events = self.trace._extract_event_data(positions=self.event_locations, 
                                                         before=self.add_points, after=self.window_size + self.add_points)

            self._get_event_properties()
            self.events = self.events - self.event_bsls[:, None]
            
            self.average_event_properties = self._get_average_event_properties()
            
            # Fit the average event; take a subset of the window.
            fit_start = int(self.window_size/6) # 1/2 of add points, i.e. half the stretch added to the events.
            fit_end = int(self.window_size/2)

            self.fitted_avg_event = self._fit_event(
                data=np.mean(self.events, axis=0)[fit_start:fit_end],
                amplitude=self.average_event_properties['amplitude'],
                t_rise=self.average_event_properties['risetime'],
                t_decay=self.average_event_properties['halfdecay_time'],
                x_offset=(self.average_event_properties['onset_position'] - fit_start)*self.trace.sampling)            

            if eval:
                self._eval_events(verbose=verbose)


    def _get_average_event_decay(self) -> float:
        ''' Returns the decay time constant of the averaged events '''
        event_x = np.arange(0, self.events.shape[1]) * self.trace.sampling
        event_avg = np.average(self.events, axis=0) * self.event_direction
        fit_start = np.argmax(event_avg) + int(0.01 * self.window_size)
        fit, _ = curve_fit(exp_fit, event_x[fit_start:], event_avg[fit_start:],
                           p0=[np.amax(event_avg), self.events.shape[1] / 50 * self.trace.sampling, 0])

        return fit[1]


    def _fit_event(self, data, amplitude, t_rise, t_decay, x_offset) -> np.ndarray:
        '''
        Performs a rudimentary fit to input event.
        time constants and offsets are in time domain.
        '''

        W_Coeff = [amplitude, t_rise, t_decay, x_offset]
        x = np.arange(0, data.shape[0]) * self.trace.sampling
        try:
            popt, _ = curve_fit(mEPSC_template, x, data, p0=W_Coeff)
        except RuntimeError:
            popt = np.array([np.nan]*4)
        
        results = {
            'amplitude':popt[0],
            'risetime':popt[1],
            't_decay':popt[2],
            'x_offset':popt[3]}

        return results


    def _eval_events(self, verbose: bool=True) -> None:
        ''' Evaluates events. Calculates mean, std and median of amplitudes & charge, as well as decay tau and
        frequency of events. Results are stored as EventStats object in self.event_stats.
        In addition, times of event peaks, onset and half decay are calculated. 
        '''
        if not self.events_present():
            return

        self.event_stats = EventStats(amplitudes=self.event_peak_values - self.event_bsls,
                                      scores=self.event_scores,
                                      tau=self._get_average_event_decay(),
                                      charges=self.charges,
                                      risetimes=self.risetimes,
                                      decaytimes=self.decaytimes,
                                      time=self.trace.total_time,
                                      unit=self.trace.y_unit)
        
        self.event_peak_times = self.event_peak_locations * self.trace.sampling
        self.event_start_times = self.event_start * self.trace.sampling
        self.half_decay_times = self.half_decay * self.trace.sampling

        self.interevent_intervals = np.diff(self.event_peak_times)

        if verbose:
            self.event_stats.print()


    def plot_single_event(self, event_num: int=0) -> None:
        ''' Plot a single events '''
        if event_num > self.events.shape[0]:
            print('Plot error: Event does not exist')
            return
        fig = plt.figure('Event')
        plt.plot(self.events[event_num])
        plt.show()


    def plot_events(self, save_fig: str='') -> None:
        ''' Plot all events (overlaid) '''
        if not self.events_present():
            return
        fig = plt.figure('Events')
        plt.plot(np.arange(0, self.events.shape[1]) * self.trace.sampling, self.events.T)
        plt.ylabel(f'{self.trace.y_unit}')
        plt.xlabel('time (s)')

        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
            return

        plt.show()


    def plot_event_average(self) -> None:
        ''' plot the average event waveform '''
        if not self.events_present():
            return
        fig = plt.figure('Event average')
        ev_average = np.mean(self.events, axis=0)
        plt.plot(np.arange(0, self.events.shape[1]) * self.trace.sampling, ev_average)
        plt.ylabel(f'{self.trace.y_unit}')
        plt.xlabel('time (s)')
        plt.show()


    def plot_event_overlay(self) -> None:
        '''
        plot the average event waveform overlayed on top of the individual events
        plus the fitted event.
        '''
        if not self.events_present():
            return
        fig = plt.figure('Event average and fit')
        plt.plot(np.arange(0, self.events.shape[1]) * self.trace.sampling, self.events.T, c='#014182', alpha=0.3)
        
        # average
        ev_average = np.mean(self.events, axis=0)
        plt.plot(np.arange(0, self.events.shape[1]) * self.trace.sampling, ev_average, c='#a90308',linewidth='3', label='average event')
        
        # fit
        fitted_ev = mEPSC_template(
                np.arange(0, self.events.shape[1]-int(self.window_size/6)) * self.trace.sampling,
                self.fitted_avg_event['amplitude'],
                self.fitted_avg_event['risetime'],
                self.fitted_avg_event['t_decay'],
                self.fitted_avg_event['x_offset'])

        plt.plot(np.arange(int(self.window_size/6), self.events.shape[1]) * self.trace.sampling,
                 fitted_ev, c='#f0833a', ls='--', label='fit')

        plt.ylabel(f'{self.trace.y_unit}')
        plt.xlabel('time (s)')
        plt.legend()
        plt.show()


    def plot_event_histogram(self, plot: str='amplitude', cumulative: bool=False) -> None:
        ''' Plot event amplitude or frequency histogram '''
        if not self.events_present():
            return
        if plot == 'frequency':
            data = np.diff(self.event_locations * self.trace.sampling, prepend=0)
            xlab_str = 'inter-event interval (s)'
        elif plot == 'amplitude':
            data = self.event_stats.amplitudes
            xlab_str = f'amplitude ({self.trace.y_unit})'
        else:
            return
        histtype = 'step' if cumulative else 'bar'
        ylab_str = 'cumulative frequency' if cumulative else 'count'
        fig = plt.figure(f'{plot}_histogram')
        plt.hist(data, bins='auto', cumulative=cumulative, density=cumulative, histtype=histtype)
        plt.ylabel(ylab_str)
        plt.xlabel(xlab_str)
        plt.show()


    def plot_prediction(self, include_data: bool=False, plot_event_params: bool=False, plot_filtered_prediction: bool=False, plot_filtered_trace: bool=False, save_fig: str='') -> None:
        ''' 
        Plot prediction trace, optionally together with data and detection result.
        
        include_data: bool
            Boolean whether to include data and detected event peaks in the plot.
        plot_event_params: bool
            Boolean whether to plot event onset and half decay points.
        plot_filtered_prediction: bool
            Boolean whether to plot filtered prediction trace (maximum filter).
        plot_filtered_trace: bool
            Boolean whether to plot filtered prediction trace (hann window). If
            True, the first and last 100 points remain unchanged, to mask edge artifacts.
        save_fig: str
            Filename to save the figure to (in SVG format). If provided, plot will not be shown.
        '''
        trace_cols = '#014182'
        thresh_cols = '#f0833a'
                
        fig = plt.figure('prediction')
        if include_data:
            ax1 = plt.subplot(211)
        prediction_x = np.arange(0, len(self.prediction)) * self.trace.sampling * self.stride_length
        if plot_filtered_prediction:
            plt.plot(prediction_x, maximum_filter1d(self.prediction, size=5, origin=-2), c=trace_cols)
        else:
            plt.plot(prediction_x, self.prediction, c=trace_cols)
        plt.axhline(self.model_threshold, ls='--', c=thresh_cols)
        plt.ylabel('probability')

        if include_data:
            plt.tick_params('x', labelbottom=False)
            _ = plt.subplot(212, sharex=ax1)
            if plot_filtered_trace:
                win = signal.windows.hann(int(self.convolve_win/self.resampling_factor))
                main_trace = signal.convolve(self.trace.data, win, mode='same') / sum(win)
                main_trace[0:100] = self.trace.data[0:100]
                main_trace[main_trace.shape[0]-100:main_trace.shape[0]] = self.trace.data[main_trace.shape[0]-100:main_trace.shape[0]]
                plt.plot(self.trace.time_axis, self.trace.data, c='k', alpha=0.4)
                plt.plot(self.trace.time_axis, main_trace, c=trace_cols)
            
            else:
                main_trace = self.trace.data
                plt.plot(self.trace.time_axis, main_trace, c=trace_cols)
            try:
                plt.scatter(self.event_peak_times, main_trace[self.event_peak_locations], c=thresh_cols, s=20, zorder=2, label='peak positions')
                if plot_event_params:
                    plt.scatter(self.event_start_times, main_trace[self.event_start], c='#a90308', s=20, zorder=2, label='event onset')
                    
                    ### remove np.nans from halfdecay
                    half_decay_for_plot = self.half_decay[np.where(~np.isnan(self.half_decay))[0]].astype(np.int64)
                    half_decay_times_for_plot = self.half_decay_times[np.where(~np.isnan(self.half_decay_times))[0]]
                    plt.scatter(half_decay_times_for_plot, main_trace[half_decay_for_plot], c='#287c37', s=20, zorder=2, label='half decay')

                data_range = np.abs(np.max(main_trace) - np.min(main_trace))
                dat_min = np.min(main_trace)
                plt.eventplot(self.event_peak_times, lineoffsets=dat_min - data_range/15, 
                            linelengths=data_range/20, color='k', lw=1.5)
            except:
                pass
            plt.tick_params('x')
            plt.ylabel(f'{self.trace.y_unit}')
            
        plt.xlabel('time (s)')
        plt.legend(loc='upper right')
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
            return
        plt.show()
    

    def plot_event_locations(self, plot_filtered: bool=False, save_fig: str='') -> None:
        ''' 
        Plot prediction trace, together with data and detected event positions (before any actual analysis is done).
        
        plot_filtered: bool
            Boolean whether to plot filtered prediction trace (maximum filter).
        save_fig: str
            Filename to save the figure to (in SVG format). If provided, plot will not be shown.
        '''
        fig = plt.figure('event locations')
        ax1 = plt.subplot(211)
        prediction_x = np.arange(0, len(self.prediction)) * self.stride_length
        if plot_filtered:
            plt.plot(prediction_x, maximum_filter1d(self.prediction, size=5, origin=-2))
        else:
            plt.plot(prediction_x, self.prediction)
        plt.axhline(self.model_threshold, color='orange', ls='--')
        plt.ylabel('probability')

        plt.tick_params('x', labelbottom=False)
        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.trace.data)
        try:
            plt.scatter(self.event_locations, self.trace.data[self.event_locations], c='orange', s=20, zorder=2)

            data_range = np.abs(np.max(self.trace.data) - np.min(self.trace.data))
            dat_min = np.min(self.trace.data)
            plt.eventplot(self.event_locations, lineoffsets=dat_min - data_range/15, 
                        linelengths=data_range/20, color='k', lw=1.5)
        except IndexError as e:
            pass
        plt.tick_params('x')
        plt.ylabel(f'{self.trace.y_unit}')
            
        plt.xlabel('time in points')
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
        else:
            plt.show()


    def plot_detection(self, save_fig: str='') -> None:
        ''' 
        Plot detection results together with data.
        
        save_fig: str
            Filename to save the figure to (in SVG format).
        '''
        fig = plt.figure('detection')
        plt.plot(self.trace.time_axis, self.trace.data, zorder=1)
        if hasattr(self, 'event_stats'):
            plt.scatter(self.event_peak_times, self.trace.data[self.event_peak_locations], c='orange', s=20, zorder=2)
            dat_range = np.abs(np.max(self.trace.data) - np.min(self.trace.data))
            dat_min = np.min(self.trace.data)
            plt.eventplot(self.event_peak_times, lineoffsets=dat_min - dat_range/15, linelengths=dat_range/20, color='k', lw=1.5)

        plt.xlabel('s')
        plt.ylabel(f'{self.trace.y_unit}')
        if save_fig:
            if not save_fig.endswith('.svg'):
                save_fig = save_fig + '.svg'
            plt.savefig(save_fig, format='svg')
            plt.close()
            return
        plt.show()


    def save_to_h5(self, filename: str, include_prediction: bool=False) -> None:
        ''' 
        Save detection results to an hdf5 (.h5) file.
        
        filename: str
            Filename to save results to. Needs to be an .h5 file.
        include_prediction: bool
            Boolean wether to include the prediction trace in the output file.
        '''
        if not hasattr(self, 'event_stats'):
            self._eval_events()
            if not hasattr(self, 'event_stats'):
                print('Save error: No events found')
                return
                
        if not filename.endswith('h5'):
            filename = ''.join(filename, '.h5')

        with h5py.File(filename, 'w', track_order=True) as f:
            f.create_dataset('events', data=np.array(self.events))
            f.create_dataset('event_params/event_locations', data=np.array(self.event_locations))
            f.create_dataset('event_params/event_scores', data=np.array(self.event_scores))
            f.create_dataset('event_params/event_amplitudes', data=self.event_stats.amplitudes)
            f.create_dataset('event_params/event_bsls', data=np.array(self.event_bsls))
            f.create_dataset('event_statistics/amplitude_average', data=self.event_stats.mean(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/amplitude_stdev', data=self.event_stats.std(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/amplitude_median', data=self.event_stats.median(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/charge_mean', data=self.event_stats.mean(self.event_stats.charges))
            f.create_dataset('event_statistics/charge_median', data=self.event_stats.median(self.event_stats.charges))
            f.create_dataset('event_statistics/frequency', data=self.event_stats.frequency())

            f.attrs['amplitude_unit'] = self.trace.y_unit
            f.attrs['recording_time'] = self.trace.data.shape[0] * self.trace.sampling
            f.attrs['source_filename'] = self.trace.filename
            f.attrs['miniml_model'] = self.model_path
            f.attrs['miniml_model_threshold'] = self.model_threshold
            f.attrs['stride'] = self.stride_length
            f.attrs['window'] = self.window_size
            f.attrs['event_direction'] = self.event_direction

            if include_prediction:
                f.create_dataset('prediction', data=self.prediction)
        print(f'Events saved to {filename}')


    def save_to_csv(self, path: str='', overwrite: bool=False) -> None:
        ''' 
        Save detection results to a .csv file. 
        Generates two files, one with averages and one with the values for the individual events.
        Filename is automatically generated.
        
        path: str
            path or directory where the file is saved
        '''
        path += '/'
        filename = self.trace.filename
        filename = filename.rsplit('.', maxsplit=1)[0]

        if not hasattr(self, 'event_stats'):
            self._eval_events()
            if not hasattr(self, 'event_stats'):
                print('Save error: No events found')
                return
        
        individual = np.stack((
            np.array(self.event_locations),
            np.array(self.event_scores),
            self.event_stats.amplitudes,
            self.event_stats.charges,
            self.event_stats.risetimes,
            self.event_stats.halfdecays))
        
        avgs = np.array((
            self.event_stats.mean(self.event_stats.amplitudes),
            self.event_stats.std(self.event_stats.amplitudes),
            self.event_stats.median(self.event_stats.amplitudes),
            self.event_stats.mean(self.event_stats.charges),
            self.event_stats.mean(self.event_stats.risetimes),
            self.event_stats.mean(self.event_stats.halfdecays),
            self.event_stats.frequency()))
        
        colnames = [f'event_{i}' for i in range(len(self.event_locations))]

        individual = pd.DataFrame(individual, index=['location', 'score', 'amplitude', 'charge', 'risetime', 'decaytime'], columns=colnames)
        avgs = pd.DataFrame(avgs, index=['amplitude mean', 'amplitude std', 'amplitude median', 'charge mean', 'risetime mean', 'decaytime mean', 'frequency'])
        
        if overwrite:
            individual.to_csv(f'{path}{filename}_individual.csv')
            avgs.to_csv(f'{path}{filename}_avgs.csv', header=False)
            print(f'events saved to {path}{filename}')
        else:
            if filename+'_individual.csv' in os.listdir(path) or filename+'_avgs.csv' in os.listdir(path):
                print(f'WARNING: file {filename} already exists. For overwriting existing files, set "overwrite = True"')
            else:
                individual.to_csv(f'{path}{filename}_individual.csv')
                avgs.to_csv(f'{path}{filename}_avgs.csv', header=False)
                print(f'events saved to {path}{filename}')


    def save_to_pickle(self, filename: str='', include_prediction:bool=True, include_data:bool=True) -> None:
        ''' 
        Save detection results to a .pickle file.         
        filename: str
            Name and if desired directory in which to save the file
        include_prediction: bool
            Include the prediction trace.
        include_data: bool
            Save the mini trace together with the analysis results
        '''
        if not hasattr(self, 'event_stats'):
            self._eval_events()
            if not hasattr(self, 'event_stats'):
                print('Save error: No events found')
                return

        if not filename.endswith('pickle'):
            filename += '.pickle'

        results = {
            'event_location_parameters':{
                'event_locations':np.array(self.event_locations),
                'event_scores':np.array(self.event_scores),
                'event_peak_locations':self.event_peak_locations,
                'event_baselines':self.event_bsls,
                'event_onset_locations':self.event_start,
                'min_positions_rise':self.min_positions_rise,
                'max_positions_rise':self.max_positions_rise,
                'half_decay_positions':self.half_decay,},
             
            'individual_values':{
                'amplitudes':self.event_stats.amplitudes,
                'charges':self.event_stats.charges,
                'risetimes':self.event_stats.risetimes,
                'half_decaytimes':self.event_stats.halfdecays},            
            
            'average_values':{
                'amplitude mean':self.event_stats.mean(self.event_stats.amplitudes),
                'amplitude std':self.event_stats.std(self.event_stats.amplitudes),
                'amplitude median':self.event_stats.median(self.event_stats.amplitudes),
                'charge mean':self.event_stats.mean(self.event_stats.charges),
                'risetime mean':self.event_stats.mean(self.event_stats.risetimes),
                'half_decaytime mean':self.event_stats.mean(self.event_stats.halfdecays),
                'decay_tau':self.event_stats.avg_tau_decay*1000,
                'frequency':self.event_stats.frequency()},
            
            'average_event_properties':self.average_event_properties,
            'average_event_fit':self.fitted_avg_event,
            'metadata':{
                ### trace information:
                'source_filename':self.trace.filename,
                'y_unit':self.trace.y_unit,
                'recording_time':self.trace.data.shape[0] * self.trace.sampling,
                'sampling':self.trace.sampling,
                'sampling_rate':self.trace.sampling_rate,

                ### miniML information
                'miniml_model':self.model_path,
                'miniml_model_threshold':self.model_threshold,
                
                ### event detection params:
                'window_size':self.window_size,
                'stride':self.stride_length,
                'add_points':self.add_points,
                'resampling_factor':self.resampling_factor,

                ### event analysis params:
                'convolve_win':self.convolve_win,
                'min_peak_w':self.peak_w,
                'rel_prom_cutoff':self.rel_prom_cutoff,
                'event_direction':self.event_direction},
            'events':self.events}


        if include_prediction:
            results['prediction']=self.prediction.numpy() # Save prediction as numpy array

        if include_data:
            results['mini_trace']=self.trace.data

        with open(filename, 'wb') as handle:
            pkl.dump(results, handle)
        print(f'events saved to {filename}')



class EventAnalysis(EventDetection):
    '''miniML class for analysis of events detected by an alternative method. Convenient for method comparison.
    Parameters
    ----------
    trace: miniML trace object
        The raw data as miniML trace object.
    window_size: int
        Number of samples to extract for each individual event.
    event_direction: str
        The direction of the events.
    event_positions: np.ndarray or list
        The position(s) of detected events.
    Methods
    ----------
    eval_events: 
        Perform event analysis.
    '''
    def __init__(self, trace, window_size, event_direction, event_positions, convolve_win, resampling_factor):
        super().__init__(data=trace, window_size=window_size, event_direction=event_direction, convolve_win=convolve_win)
        self.add_points = int(self.window_size/3)
        self.resampling_factor = resampling_factor
        self.event_locations = event_positions[np.logical_and(
                                                self.add_points < event_positions, 
                                                event_positions < len(self.trace.data) - (self.window_size + self.add_points))]
        self.event_locations = self.event_locations.astype(np.int64)
        self.events = self.trace._extract_event_data(self.event_locations, before=self.add_points, 
                                                     after=self.window_size + self.add_points)

    def eval_events(self, filter: bool=True, verbose: bool=True) -> None:
        if self.event_locations.shape[0] > 0:
            super()._get_event_properties(filter=filter)
            self.events = self.events - self.event_bsls[:, None]
            super()._eval_events(verbose=verbose)
