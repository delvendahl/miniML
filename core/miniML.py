from __future__ import annotations
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pandas as pd
import pickle as pkl
from pathlib import Path
from scipy import signal
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d
from miniML_functions import (get_event_peak, get_event_baseline, get_event_onset, get_event_risetime, 
                              get_event_halfdecay_time, get_event_charge, get_event_halfwidth)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#  --------------------------------------------------  #
#  general functions                                   #
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


def mEPSC_template(x: np.ndarray, amplitude: float, t_rise: float, t_decay: float, x0: float) -> np.ndarray:
    """
    Generates a template miniature excitatory postsynaptic current (mEPSC) based on the given parameters.

    Parameters:
        x (np.ndarray): An array of x values.
        amplitude (float): The amplitude of the mEPSCs.
        t_rise (float): The rise time constant of the mEPSCs.
        t_decay (float): The decay time constant of the mEPSCs.
        x0 (float): The onset time point for the mEPSCs.

    Returns:
        np.ndarray: An array of y values representing an mEPSC template.

    Note:
        - The formula used to calculate the template is y = amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay).
        - Any values of x that are less than x0 will be set to 0 in the resulting array.
    """
    y = amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay)
    y[x < x0] = 0

    return y


def is_keras_model(filepath: str) -> bool:
    """
    Checks if a given HDF5 contains a keras model.
    """
    with h5py.File(filepath, 'r') as f:
        return 'keras_version' in f.attrs
    

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

        return cls(data=data, sampling_interval=sampling, y_unit=unit, filename=Path(filename).name)


    @classmethod
    def from_heka_file(cls, filename: str, rectype: str, group: int=0, load_series: list=[], exclude_series: list=[], 
                       exclude_sweeps: dict={}, scaling: float=1, unit: str=None, resample: bool=True) -> MiniTrace:
        ''' Loads data from a HEKA .dat file. Name of the PGF sequence needs to be specified.

        Parameters
        ----------
        filename: string
            Path of a .dat file.
        rectype: string
            Name of the PGF sequence in the file to be loaded.
        group: int, default=1
            HEKA group to load data from. Note that HEKA groups are numbered starting from 1, but Python idexes from zero. 
            Hence, group 1 in HEKA is group 0 in Python. 
        load_series: list, default=[]
            List of HEKA series to load. Uses zero-indexing, i.e. HEKA series 1 is 0 in the list.
        exclude_series: list, default=[].
            List of HEKA series to exclude.
        exclude_sweeps: dict, default={}.
            Dictionary with sweeps to exclude from analysis. E.g. {2 : [4, 5]} excludes sweeps 4 & 5 from series 2.
        scaling: float, default=1e12
            Scaling factor applied to the data. Defaults to 1e12 (i.e. pA)
        unit: str, default=''
            Data unit, to be set when using scaling factor.
        resample: boolean, default=rue
            Resample data in case of sampling rate mismatch.

        Returns
        -------
        MiniTrace
            An initialized MiniTrace object.

        Raises
        ------
        Exception or ValueError
            If the file is not a valid .dat file.
        IndexError
            When the group index is out of range.
        ValueError
            When the sampling rates of different series mismatch and resampling is set to False.
        '''
        if not Path(filename).suffix.lower() == '.dat':
            raise Exception('Incompatible file type. Method only loads .dat files.')

        import FileImport.HekaReader as heka
        bundle = heka.Bundle(filename)

        if group < 0 or group > len(bundle.pul.children) - 1:
            raise IndexError('Group index out of range')

        bundle_series = dict()
        for i, SeriesRecord in enumerate(bundle.pul[group].children):
            bundle_series.update({i: SeriesRecord.Label})

        if load_series == []:
            series_list = [series_number for series_number, record_type in bundle_series.items() \
                    if record_type == rectype and series_number not in exclude_series]
        else:
            load_series = [x for x in load_series if x not in exclude_series]
            series_list = [series_number for series_number, record_type in bundle_series.items() \
                    if record_type == rectype and series_number in load_series]

        series_data = []
        series_resistances = []
        for series in series_list:
            sweep_data = []
            for sweep in range(bundle.pul[group][series].NumberSweeps):
                if series not in exclude_sweeps:
                    try:
                        sweep_data.append(bundle.data[group, series, sweep, 0])   
                    except IndexError as e:
                        pass
                else:
                    if sweep not in exclude_sweeps[int(series)]:
                        try:
                            sweep_data.append(bundle.data[group, series, sweep, 0])   
                        except IndexError as e:
                            pass
            pgf_series_index = sum(len(bundle.pul[i].children) for i in range(group)) + series
            series_data.append((np.array(sweep_data).flatten(), bundle.pgf[pgf_series_index].SampleInterval))
            series_resistances.append((1/bundle.pul[group][series][0][0].GSeries) * 1e-6)

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
        
        data_unit = unit if unit is not None else bundle.pul[group][series_list[0]][0][0].YUnit

        MiniTrace.excluded_sweeps = exclude_sweeps
        MiniTrace.exlucded_series = exclude_series
        MiniTrace.Rseries = series_resistances

        bundle.close()
        
        return cls(data=data * scaling, sampling_interval=max_sampling_interval, 
                   y_unit=data_unit, filename=Path(filename).name)


    @classmethod
    def from_axon_file(cls, filename: str, channel: int=0, scaling: float=1.0, unit: str=None) -> MiniTrace:
        ''' Loads data from an AXON .abf file.

        Parameters
        ----------
        filename: string
            Path of a .abf file.
        channel: int, default=0
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
        if not Path(filename).suffix.lower() == '.abf':
            raise Exception('Incompatible file type. Method only loads .abf files.')

        import pyabf
        abf_file = pyabf.ABF(filename)
        if channel not in abf_file.channelList:
            raise IndexError('Selected channel does not exist.')

        data_unit = unit if unit is not None else abf_file.adcUnits[channel]

        return cls(data=abf_file.data[channel] * scaling, sampling_interval=1/abf_file.sampleRate, 
                    y_unit=data_unit, filename=Path(filename).name)


    def plot_trace(self) -> None:
        ''' Plots the trace '''
        plt.plot(self.time_axis, self.data)
        plt.xlabel('Time [s]')
        plt.ylabel(f'[{self.y_unit}]')
        plt.show()


    def detrend(self, detrend_type: str='linear', num_segments: int=0) -> MiniTrace:
        ''' Detrend the data. 

        Parameters
        ----------
        detrend_type: str, default='linear'
            Type of detrending. Options: 'linear', 'constant'
        num_segments: int, default=0
            Number of segments for detrending. Increase in case of non-linear trends in the data.

        Returns
        -------
        MiniTrace
            The detrended MiniTrace object.
        '''
        num_data = self.data.shape[0]
        breaks = np.arange(num_data/num_segments, num_data, num_data/num_segments, dtype=np.int64) if num_segments > 1 else 0
        detrended = signal.detrend(self.data, bp=breaks, type=detrend_type)

        return MiniTrace(detrended, self.sampling, y_unit=self.y_unit, filename=self.filename)


    def filter(self, line_freq: float=None, width: float=None, highpass: float=None, lowpass: float=None, order: int=4,
               savgol: float=None, hann: int=None) -> MiniTrace:
        ''' Filters trace with a combination of line frequency, high- and lowpass filters.
        If both lowpass and savgol arguments are passed, only the lowpass filter is applied. 

        Parameters
        ----------
        line_freq: float, default=None
            Line noise filter frequency (Hz). Line noise is removed by spectrum interpolation.
        width: float, default=None
            Width of the line noise filter (Hz).
        highpass: float, default=None
            Highpass cutoff frequency (Hz).
        lowpass: float, default=None
            Lowpass cutoff frequency (Hz). Set to None to turn filtering off.
        order: int, default=4
            Order of the filter.
        savgol: float, default=None
            The time window for Savitzky-Golay smoothing (ms).
        hann: int, default=None
            The length of the Hann window (samples).
            
        Returns
        -------
        MiniTrace
            A filtered MiniTrace object.
        '''
        filtered_data = self.data.copy()
        nyq = 0.5 * self.sampling_rate

        if line_freq:
            from scipy.fftpack import rfft, irfft, rfftfreq
            fft = rfft(filtered_data)
            xf = rfftfreq(filtered_data.shape[0], 1 / self.sampling_rate)
            multiples = 6
            for freq in np.arange(line_freq, (multiples * line_freq), line_freq):
                fft[np.where(xf > freq - width/2)[0][0]:np.where(xf > freq + width/2)[0][0]] = 0

            filtered_data = irfft(fft)
        if highpass:
            sos = signal.butter(order, highpass / nyq, btype='high', output='sos')
            filtered_data = signal.sosfilt(sos, filtered_data)
        if lowpass:
            if savgol:
                print('Warning: Two lowpass filters selected, Savgol filter is ignored.')
            sos = signal.butter(order, lowpass / nyq, btype='low', analog=False, output='sos', fs=None)
            filtered_data = signal.sosfiltfilt(sos, filtered_data)
        elif savgol:
            filtered_data = signal.savgol_filter(filtered_data, int(savgol / 1000 / self.sampling), polyorder=order)
        elif hann:
            win = signal.windows.hann(hann)
            filtered_data = signal.convolve(filtered_data, win, mode='same') / sum(win)
            # Hann window generates edge artifacts due to zero-padding. Retain unfiltered data at edges.
            filtered_data[:hann] = self.data[:hann]
            filtered_data[filtered_data.shape[0] - hann : filtered_data.shape[0]] = self.data[filtered_data.shape[0] - hann : filtered_data.shape[0]]

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
        resampled_data = signal.resample(self.data, int(self.data.shape[0]/resampling_factor))
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
            Number of samples after event positions for event extraction. Positions+after must be smaller 
            than the total number of samples in self.data.

        Returns
        ------
        np.ndarray
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
        10-90 percent rise times of individual events.
    slopes: np.ndarray
        Rise slopes of individual events.
    halfdecays: np.ndarray
        Half decay times of individual events.
    halfwidths: np.ndarray
        Half-width of individual events (seconds).
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
    def __init__(self, amplitudes, scores, charges, risetimes, slopes, decaytimes, halfwidths, tau, time, unit: str) -> None:
        self.amplitudes = amplitudes
        self.event_scores = scores
        self.charges = charges
        self.risetimes = risetimes
        self.slopes = slopes
        self.halfdecays = decaytimes
        self.halfwidths = halfwidths
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
        if ~np.all(np.isnan(values)) and self.event_count:
            return np.nanmedian(values)
        else:
            return np.nan

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
        print(f'    Mean half-width: {self.mean(self.halfwidths)*1000:.3f} ms')
        print(f'    Tau decay: {self.avg_tau_decay*1000:.3f} ms')
        print('-------------------------')



class EventDetection():
    '''miniML main class with methods for event detection and analysis.

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
    verbose: int, default=1
        set verbose level (0 = no output, 1 = info, 2 = full)
    batch_size: int, default=128
        The batch size for the event detection (used in model.predict).
    model_path: str, default=''
        The path of the model file (.h5) to be used for event detection.
    model: tf.keras.Model, default=None
        A model instance to be used for event detection. Overrides loading from model_path method if specified.
    model_threshold: float, default=0.5
        The minimum peak heigth of the model prediction to be considered as an event; range=(0,1).
    compile_model: bool, default=True
        Whether to compile the model.
    callbacks: list, default=[]
        List of callback functions to be used during event detection.

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
    def __init__(self, data: MiniTrace, window_size: int=600, event_direction: str='negative', training_direction: str='negative', verbose=1,
                 batch_size: int=128, model: tf.keras.Model=None, model_path: str='', model_threshold: float=0.5, compile_model=True, callbacks: list=[]) -> None:
        self.trace = data
        self.prediction = None
        self.window_size = window_size
        self.event_direction = event_direction
        self.training_direction = training_direction
        self.verbose = verbose
        self.event_locations = np.array([])
        self.event_scores = np.array([])
        self.events = np.array([])
        self.batch_size = batch_size
        self.model_path = model_path
        self.model = None
        self.model_threshold = None
        if model:
            self.model = model
            self.model_threshold = model_threshold
            self.callbacks = callbacks
        elif model_path:
            self.load_model(filepath=model_path, threshold=model_threshold, compile=compile_model)
            self.callbacks = callbacks
        self.deleted_events = 0
        
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


    def _init_arrays(self, attr_names: list, shape: int, dtype: type) -> None:
        ''' initialize multiple 1d ndarrays with given shape containing NaNs '''
        for label in attr_names:
            value = -1 if 'int' in str(dtype) else np.nan
            setattr(self, str(label), np.full(int(shape), value, dtype=dtype))


    def events_present(self) -> bool:
        ''' Checks if events are present '''
        num_events = self.events.shape[0]
        
        return num_events != 0


    def load_model(self, filepath: str, threshold: float=0.5, compile: bool=True) -> None:
        ''' Loads a trained miniML model from hdf5 file '''
        if not is_keras_model(filepath):
            raise ValueError('Model file is not a valid Keras model')
        self.model = tf.keras.models.load_model(filepath, compile=compile)
        self.model_threshold = threshold
        if self.verbose:
            print(f'Model loaded from {filepath}')


    def lowpass_filter(self, data: np.ndarray, cutoff: float, order: int=4) -> np.ndarray:
        '''
        Butterworth lowpass filter. 
        '''
        nyq = 0.5 * self.trace.sampling_rate

        sos = signal.butter(order, cutoff / nyq, btype='lowpass', analog=False, output='sos', fs=None)

        return signal.sosfiltfilt(sos, data)
    

    def hann_filter(self, data: np.ndarray, filter_size: int) -> np.ndarray:
        '''
        Hann window filter. Start and end of the data are not filtered to avoid artifacts
        resulting from zero padding.
        '''
        if filter_size == 0:
            return data
        win = signal.windows.hann(filter_size)    
        filtered_data = signal.convolve(data, win, mode='same') / sum(win)
        filtered_data[:filter_size] = data[:filter_size]
        filtered_data[-filter_size:] = data[-filter_size:]

        return filtered_data


    def _linear_interpolation(self, data: np.ndarray, interpol_to_len: int) -> tuple[np.ndarray, float]:
        '''
        linear interpolation of a data stretch to match the indicated number of points.

        Returns
        -------
        data_interpolated:
            the interpolated data
        interpol_factor:
            the factor by which the data was up- or downsampled
        '''        
        x = np.arange(0, data.shape[0])
        x_interpol = np.linspace(0, data.shape[0], interpol_to_len)
        
        interpol_factor = len(x_interpol) / len(x)
        data_interpolated = np.interp(x_interpol, x, data, left=None, right=None, period=None)

        return data_interpolated, interpol_factor


    def __predict(self) -> None:
        '''
        Performs prediction on a data trace using a sliding window of size `window_size` with a stride size given by `stride`.
        The prediction is performed on the data using the miniML model.
        Speed of prediction depends on batch size of model.predict(), but too high batch sizes will give low precision results.

        Raises  
        ------
        ValueError
            When stride is below 1 or above window length
        '''
        # resample values for prediction:
        data = signal.resample(self.trace.data, round(len(self.trace.data)*self.resampling_factor))
        
        # invert the trace if event_direction and training_direction are different.
        if self.event_direction != self.training_direction:
            data *= -1

        win_size = round(self.window_size * self.resampling_factor)
        stride = round(self.stride_length * self.resampling_factor)

        if stride <= 0 or stride > win_size:
            raise ValueError('Invalid stride')
        
        ds = tf.keras.utils.timeseries_dataset_from_array(data=np.expand_dims(data, axis=1).astype(np.float32), 
                                                          targets=None, 
                                                          sequence_length=win_size, 
                                                          sequence_stride=stride,
                                                          batch_size=None,
                                                          shuffle=False)
        
        ds = ds.map(minmax_scaling, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.prediction = tf.squeeze(self.model.predict(ds, verbose=self.verbose, callbacks=self.callbacks))
        

    def _interpolate_prediction_trace(self) -> tuple[np.ndarray, float]:
        '''
        Interpolate the prediction trace such that it corresponds 1:1 to the raw data before resampling.
        Last few points of the data will not have prediction values because the data is shorter than the
        required window size.
        '''
        stride = round(self.stride_length*self.resampling_factor)
        pn = len(self.prediction) - 1 
        pn_mapped = pn * stride
        pn_in_raw_data = round(pn_mapped/self.resampling_factor)
        resampled_prediction, interpol_factor = self._linear_interpolation(data=self.prediction, interpol_to_len=pn_in_raw_data)

        return resampled_prediction, interpol_factor


    def _get_prediction_peaks(self, peak_w: int=10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        '''
        Find peaks in prediction trace and extracted start- and endpoints of event areas based on left 
        and right ips respectively.
        '''
        filtered_prediction = maximum_filter1d(self.prediction, size=int(5*self.interpol_factor), origin=-2)
        
        _, peak_properties = signal.find_peaks(x=filtered_prediction, height=self.model_threshold,
                                               prominence=self.model_threshold, width=peak_w*self.interpol_factor)

        start_pnts = np.array(peak_properties['left_ips'] + self.window_size/4, dtype=np.int64)
        # check if start_pnts are larger than right_ips and limit to this value minus buffer of a quarter of peak width
        boolean_indices = start_pnts > peak_properties['right_ips']
        start_pnts[boolean_indices] = peak_properties['right_ips'][boolean_indices] - peak_properties['widths'][boolean_indices] / 4
        end_pnts =  np.array(peak_properties['right_ips'] + self.window_size/2, dtype=np.int64)
        scores = peak_properties['peak_heights']

        return start_pnts, end_pnts, scores


    def _make_smth_gradient(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Generate a smoothed gradient trace of the data.
        '''
        # filter raw data trace, calculate gradient and filter first derivative trace
        if self.convolve_win > 0:
            trace_convolved = self.hann_filter(data=self.trace.data, filter_size=self.convolve_win)
        else:
            trace_convolved = self.lowpass_filter(data=self.trace.data, cutoff=self.trace.sampling_rate / self.filter_factor, order=4)
        trace_convolved *= self.event_direction # (-1 = 'negative', 1 else)
        
        gradient = np.gradient(trace_convolved, self.trace.sampling)
        # gradient[:int(self.convolve_win * 1.5)] = 0
        # gradient[-int(self.convolve_win * 1.5):] = 0

        smth_gradient = self.hann_filter(data=gradient, filter_size=self.gradient_convolve_win)
        smth_gradient[:self.gradient_convolve_win] = 0
        smth_gradient[-self.gradient_convolve_win:] = 0

        return gradient, smth_gradient


    def _get_grad_threshold(self, grad: np.ndarray, start_pnts: np.ndarray, end_pnts: np.ndarray) -> int:
        '''
        Get threshold based on standard deviation of the derivative of event-free data sections.
        '''
        split_data = np.split(grad, np.vstack((start_pnts, end_pnts)).ravel('F'))
        event_free_data = np.concatenate(split_data[::2]).ravel()
        grad_threshold = int(4 * np.std(event_free_data))

        return grad_threshold


    def _find_event_locations(self, limit: int, scores: np.ndarray, rel_prom_cutoff: float=0.25) -> tuple[np.ndarray, np.ndarray]:
        '''
        Find approximate event positions based on negative threshold crossings in prediction trace. Extract
        segment of peak windows in prediction trace and search for peaks in first derivative. If no peak is found,
        the maximum first derivate is used as peak localization.

        Parameters
        ------
        limit: int
            Right trace limit to make sure events at the very border are not picked up.
        scores: numpy array
            Prediction value for the events
        rel_prom_cutoff: float
            Relative prominence cutoff. Determines the minimum relative prominence for detection of overlapping events

        Returns
        ------
        event_locations: numpy array
            Location of steepest rise of the events
        event_scores: numpy array
            Prediction value for the events
        '''
        # Remove indices at left and right borders to prevent boundary issues.
        mask = (self.start_pnts > self.window_size) & (self.end_pnts < self.prediction.shape[0])
        
        self.end_pnts = self.end_pnts[mask]
        self.start_pnts = self.start_pnts[mask]
        scores = scores[mask]

        event_locations, event_scores = [], []
        for i, _ in enumerate(self.start_pnts): 
            peaks, peak_params = signal.find_peaks(x=self.smth_gradient[self.start_pnts[i]:self.end_pnts[i]], 
                                                   height=self.grad_threshold, prominence=self.grad_threshold)
            
            if peaks.shape[0] > 1: # If > 1 peak found; apply relative prominence cutoff
                rel_prom = peak_params['prominences'] / np.max(peak_params['prominences'])
                inds = np.argwhere(rel_prom >= rel_prom_cutoff).flatten()
                peaks = peaks[inds]
                for my_param in peak_params:
                    peak_params[my_param] = peak_params[my_param][inds]
            
            if not len(peaks): # If no peak found: default argmax finding
                peaks = np.array([np.argmax(self.smth_gradient[self.start_pnts[i]:self.end_pnts[i]])])

            for peak in peaks:
                if (self.start_pnts[i] + peak) >= (self.trace.data.shape[0] - limit):
                    continue
                if self.start_pnts[i] + peak not in event_locations:
                    event_locations.append(self.start_pnts[i] + peak)
                    event_scores.append(scores[i])

        return np.array(event_locations), np.array(event_scores)       


    def _remove_duplicate_locations(self) -> None:
        '''
        Remove event locations and associated scores that have potentially been picked up by
        overlapping start-/ end-points of different detection peaks.
        '''
        unique_indices = np.unique(self.event_locations, return_index=True)[1]
        self.event_locations = self.event_locations[unique_indices]
        self.event_scores = self.event_scores[unique_indices]
        
        duplicate_indices = np.argwhere(np.diff(self.event_locations)<self.window_size/100).flatten() + 1
        self.event_locations = np.delete(self.event_locations, duplicate_indices)
        self.event_scores = np.delete(self.event_scores, duplicate_indices)
        self.event_locations = np.asarray(self.event_locations, dtype=np.int64)


    def _get_event_properties(self, filter: bool=True) -> dict:
        '''
        Find more detailed event location properties required for analysis. Namely, baseline, event onset,
        peak half-decay and 10 & 90% rise positions. Also extracts the actual event properties, such as
        amplitude or half-decay time.
        
        Parameters
        ------
        filter: bool
            If true, properties are extracted from the filtered data.
        '''
        ### Prepare data
        diffs = np.diff(self.event_locations, append=self.trace.data.shape[0]) # Difference in points between the event locations
        add_points = int(self.window_size/3)
        after = self.window_size + add_points
        positions = self.event_locations

        ### Set parameters for charge calculation
        charge_factor = 4
        combined_charge_events = 1
        calculate_charge = False # will be set to True in the loop if double event criteria are fulfilled; not a flag for charge 

        if np.any(positions - add_points < 0) or np.any(positions + after >= self.trace.data.shape[0]):
            raise ValueError('Cannot extract time windows exceeding input data size.')
        # Filter data if required
        if filter:
            if self.convolve_win > 0:
                mini_trace = self.hann_filter(data=self.trace.data, filter_size=self.convolve_win)
            else:
                mini_trace = self.lowpass_filter(data=self.trace.data, cutoff=self.trace.sampling_rate / self.filter_factor, order=4)
        else:
            mini_trace = self.trace.data.copy()
        mini_trace *= self.event_direction

        self._init_arrays(['event_peak_locations', 'bsl_starts', 'bsl_ends', 'event_start'], positions.shape[0], dtype=np.int64)
        self._init_arrays(['event_peak_values', 'event_bsls', 'event_bsl_durations', 'decaytimes', 'charges', 'risetimes', 'half_decay', 'halfwidths', 'rise_half_amp_times', 'decay_half_amp_times'], positions.shape[0], dtype=np.float64)               
        self._init_arrays(['min_positions_rise', 'max_positions_rise', 'min_values_rise', 'max_values_rise'], positions.shape[0], dtype=np.float64)               

        for ix, position in enumerate(positions):
            indices = position + np.arange(-add_points, after)
            data = mini_trace[indices]
            
            event_peak_pos = get_event_peak(data=data, event_num=ix, add_points=add_points, window_size=self.window_size, diffs=diffs)

            self.event_peak_locations[ix] = int(event_peak_pos)
            self.event_peak_values[ix] = np.mean(data[event_peak_pos - self.peak_spacer:event_peak_pos + self.peak_spacer])
            
            baseline, baseline_var, bsl_start, bsl_end, bsl_duration = get_event_baseline(data=data, event_num=ix, diffs=diffs, add_points=add_points,
                                                                                          peak_positions=self.event_peak_locations, positions=positions)
            self.bsl_starts[ix] = bsl_start
            self.bsl_ends[ix] = bsl_end

            self.event_bsls[ix] = baseline
            self.event_bsl_durations[ix] = bsl_duration
            
            onset_position = get_event_onset(data=data, peak_position=event_peak_pos, baseline=baseline, baseline_var=baseline_var)
            self.event_start[ix] = onset_position

            risetime, min_position_rise, min_value_rise, max_position_rise, max_value_rise = get_event_risetime(data=data[bsl_start:int(event_peak_pos)], 
                                                                                                                sampling_rate=self.trace.sampling_rate, baseline=baseline, 
                                                                                                                amplitude=self.event_peak_values[ix] - baseline)
            self.risetimes[ix] = risetime
            self.min_positions_rise[ix] = min_position_rise
            self.min_values_rise[ix] = min_value_rise

            self.max_positions_rise[ix] = max_position_rise
            self.max_values_rise[ix] = max_value_rise

            half_amplitude_level = baseline + (data[event_peak_pos] - baseline) / 2
            if diffs[ix] < add_points: # next event close; check if we can get halfdecay
                right_lim = diffs[ix] + add_points # Right limit is the onset of the next event
                test_arr =  data[event_peak_pos:right_lim]
                if test_arr[test_arr < half_amplitude_level].shape[0]: # means that event goes below 50% ampliude before max rise of the next event; 1/2 decay can be calculated
                    halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data[0:right_lim], peak_position=event_peak_pos, baseline=baseline)
                else:
                    halfdecay_position, halfdecay_time = np.nan, np.nan
            else:  
                halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data, peak_position=event_peak_pos, baseline=baseline)

            self.half_decay[ix] = halfdecay_position
            self.decaytimes[ix] = halfdecay_time

            # Calculate halfwidth and related times
            current_amplitude = abs(self.event_peak_values[ix] - self.event_bsls[ix])
            if self.event_direction == -1 : # event_peak_values and bsls are already inverted for negative events later, so use original direction for amplitude calc with data
                 current_amplitude = abs(data[event_peak_pos] - baseline)

            halfwidth, t_rise_half, t_decay_half = get_event_halfwidth(
                event_data=data,
                peak_index=event_peak_pos,
                baseline=baseline,
                amplitude=current_amplitude,
                sampling_rate=self.trace.sampling_rate
            )
            self.halfwidths[ix] = halfwidth
            self.rise_half_amp_times[ix] = t_rise_half
            self.decay_half_amp_times[ix] = t_decay_half
            
            # calculate charges
            ### For charge; multiple event check done outside function.
            if ix < positions.shape[0] - 1:
                if combined_charge_events == 1: # define onset position for charge calculation
                    onset_in_trace = positions[ix] - (add_points - self.event_start[ix])
                    baseline_for_charge = self.event_bsls[ix]

                if np.isnan(self.half_decay[ix]):
                    combined_charge_events += 1
                
                else:
                    ### Get distance from peak to next event location.
                    peak_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points)
                    next_event_location = positions[ix + 1]
                    delta_peak_location = next_event_location - peak_in_trace
                    
                    # determine end of area calculation based on event decay
                    endpoint = int(self.event_peak_locations[ix] + charge_factor * (int(self.half_decay[ix]) - self.event_peak_locations[ix]))
                    delta_peak_endpoint = endpoint - self.event_peak_locations[ix]

                    if delta_peak_location > delta_peak_endpoint: # Next event_location further away than the charge window; calculate charge
                        calculate_charge = True
                    else:
                        combined_charge_events += 1

                if calculate_charge:
                    endpoint_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points) + delta_peak_endpoint
                    charge = get_event_charge(data=mini_trace, start_point=onset_in_trace, end_point=endpoint_in_trace, baseline=baseline_for_charge, sampling=self.trace.sampling)
                    
            else: # Handle the last event
                if combined_charge_events == 1: # define onset position for charge calculation
                    onset_in_trace = positions[ix] - (add_points - self.event_start[ix])
                    baseline_for_charge = self.event_bsls[ix]
                
                peak_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points)
                endpoint = int(self.event_peak_locations[ix] + charge_factor * (int(self.half_decay[ix]) - self.event_peak_locations[ix]))
                delta_peak_endpoint = endpoint - self.event_peak_locations[ix]
                endpoint_in_trace = positions[ix] + (self.event_peak_locations[ix] - add_points) + delta_peak_endpoint
                
                if endpoint_in_trace > mini_trace.shape[0]:
                    endpoint_in_trace = mini_trace.shape[0]

                charge = get_event_charge(data=mini_trace, start_point=onset_in_trace, end_point=endpoint_in_trace, baseline=baseline_for_charge, sampling=self.trace.sampling)
                calculate_charge = True
            if calculate_charge: # Charge was calculated; check how many potentially overlapping events contributed.
                charge = [charge / combined_charge_events] * combined_charge_events
                for ix_adjuster in range(len(charge)):
                    self.charges[ix - ix_adjuster] = charge[ix_adjuster]
                            
                # Reset values after calculation
                calculate_charge = False
                combined_charge_events = 1

        ## Convert units
        self.event_peak_values *= self.event_direction
        self.event_bsls *= self.event_direction
        self.max_values_rise *= self.event_direction
        self.min_values_rise *= self.event_direction

        self.decaytimes *= self.trace.sampling
        self.charges *= self.event_direction

        ## map indices back to original trace
        for ix, position in enumerate(positions):
            self.event_peak_locations[ix] = int(self.event_peak_locations[ix] + self.event_locations[ix] - self.add_points)
            self.bsl_starts[ix] = int(self.bsl_starts[ix] + self.event_locations[ix] - self.add_points)
            self.bsl_ends[ix] = int(self.bsl_ends[ix] + self.event_locations[ix] - self.add_points)
            
            self.event_start[ix] = int(self.event_start[ix] + self.event_locations[ix] - self.add_points)
            self.rise_half_amp_times[ix] += (self.event_locations[ix] - self.add_points) * self.trace.sampling
            self.decay_half_amp_times[ix] += (self.event_locations[ix] - self.add_points) * self.trace.sampling
            self.min_positions_rise[ix] += (self.bsl_starts[ix] * self.trace.sampling)
            self.max_positions_rise[ix] += (self.bsl_starts[ix] * self.trace.sampling)
            
            if not np.isnan(self.half_decay[ix]):
                self.half_decay[ix] = int(self.half_decay[ix] + self.event_locations[ix] - self.add_points)
        
        
    def _get_singular_event_indices(self):
        '''
        Extract indices of events that have no overlap with any other events.
        '''
        no_events_in_decay = np.where(np.diff(self.event_locations) > self.window_size * 1.5)[0]
        no_events_in_rise = (np.where(np.diff(self.event_locations) > self.window_size * 0.5)[0]) + 1
        self.singular_event_indices = np.intersect1d(no_events_in_rise, no_events_in_decay, assume_unique=False, return_indices=False)
        
        # First and last event will be lost due to intersecting. Add manually if they qualify.
        if 0 in no_events_in_decay:
            self.singular_event_indices = np.insert(self.singular_event_indices, 0, 0)
        if len(self.event_locations) - 1 in no_events_in_rise:
            self.singular_event_indices = np.append(self.singular_event_indices, [len(self.event_locations) - 1])

        # if all events are overlapping, use all of them.
        if not len(self.singular_event_indices):
            self.singular_event_indices = np.array(list(range(self.event_locations.shape[0])))

            
    def _get_average_event_properties(self) -> dict:
        '''
        Extract properties of the event average the same way the individual events are analysed.
        '''
        add_points = int(self.window_size / 3)
        diffs = [add_points * 10] # Set right limit larger than window size
        charge_factor = 4 # Charge window is 4 * decay time
        data = np.mean(self.events[self.singular_event_indices], axis=0) * self.event_direction

        event_peak = get_event_peak(data=data, event_num=0, add_points=add_points, window_size=self.window_size, diffs=diffs)
        event_peak_value = data[event_peak]

        baseline, baseline_var, bsl_start, _, _ = get_event_baseline(data=data, event_num=0, diffs=diffs, 
                                                                     add_points=add_points, peak_positions=[event_peak], positions=[add_points])
        onset_position = get_event_onset(data=data, peak_position=event_peak, baseline=baseline, baseline_var=baseline_var)
                
        risetime, min_position_rise, min_value_rise, max_position_rise, max_value_rise = get_event_risetime(
                data=data[bsl_start:int(event_peak)], sampling_rate=self.trace.sampling_rate, baseline=baseline, amplitude=event_peak_value - baseline)

        halfdecay_position, halfdecay_time = get_event_halfdecay_time(data=data, peak_position=event_peak, baseline=baseline)        
        endpoint = int(event_peak + charge_factor * halfdecay_position)
        charge = get_event_charge(data=data, start_point=onset_position, end_point=endpoint, baseline=baseline, sampling=self.trace.sampling)

        results = {'amplitude': event_peak_value - baseline,
                   'baseline': baseline * self.event_direction,
                   'risetime': risetime * self.trace.sampling,
                   'halfdecay_time': halfdecay_time * self.trace.sampling,
                   'charge': charge * self.event_direction,
                   'event_peak': event_peak,
                   'onset_position': onset_position,
                   'min_position_rise': min_position_rise,
                   'min_value_rise': min_value_rise * self.event_direction,
                   'max_position_rise': max_position_rise,
                   'max_value_rise': max_value_rise * self.event_direction,
                   'halfdecay_position': halfdecay_position,
                   'endpoint_charge': endpoint
                   }
        
        return results


    def detect_events(self, stride: int=None, eval: bool=False, resample_to_600: bool=True, peak_w: int=5, rel_prom_cutoff: float=0.25, 
                      filter_factor: float=20.0, convolve_win: int=0, gradient_convolve_win: int=0) -> None:
        '''
        Wrapper function to perform event detection, extraction and analysis
        
        Parameters
        ------
        stride: int, default = None
            The stride used during prediction. If not specified, it will be set to 1/30 of the window size
        eval: bool, default = False
            Whether to evaluate detected events.
        resample_to_600: bool, default = True
            Whether to resample the the data to match a 600 point window. Should always be true, unless a model was trained with a different window size.
        peak_w: int, default = 5
            The minimum prediction peak width.
        rel_prom_cutoff: int, float = 0.25
            The relative prominence cutoff. Overlapping events are separated based on a peak-finding in the first derivative. To be considered
            an event, any detected peak must have at least 25% prominence of the largest detected prominence.
        filter_factor: float, default = 20
            Filter factor for the lowpass filter used to filter the data for event analysis. Fraction of sampling rate (20 = 1/20 of sampling rate). 
        convolve_win: int, default = 0
            Window size for the hanning window used to filter the data for event analysis. If 0, no filtering is applied. Used for legacy compatibility.
        gradient_convolve_win: int, default = 0
            Window size for the hanning window used to filter the derivative for event analysis
        '''   
        self.peak_w = peak_w
        self.rel_prom_cutoff = rel_prom_cutoff
        self.filter_factor = filter_factor
        self.convolve_win = convolve_win
        self.add_points = int(self.window_size / 3)
        
        self.stride_length = stride if stride else round(self.window_size / 30)
        self.gradient_convolve_win = gradient_convolve_win
        self.resampling_factor = 600 / self.window_size if resample_to_600 else 1
        
        # Define peak spacer, i.e. number of points left / right of detected event peaks to use for amplitude calculation.
        if int(self.window_size / 300) < 1:
            self.peak_spacer = 1
        else:
            self.peak_spacer = int(self.window_size / 300)

        self.__predict()
        
        # Linear interpolation of prediction trace to match the original data.
        self.prediction, self.interpol_factor = self._interpolate_prediction_trace()
        self.start_pnts, self.end_pnts, scores = self._get_prediction_peaks(peak_w=peak_w)
        self.gradient, self.smth_gradient = self._make_smth_gradient()
        self.grad_threshold = self._get_grad_threshold(grad=self.smth_gradient, start_pnts=self.start_pnts, end_pnts=self.end_pnts)
        self.event_locations, self.event_scores = self._find_event_locations(limit=self.window_size + self.add_points, scores=scores,
                                                                             rel_prom_cutoff=rel_prom_cutoff)
        self._remove_duplicate_locations()

        self.slopes = self.smth_gradient[self.event_locations]

        if self.event_locations.shape[0] > 0:
            self._get_singular_event_indices()
            self.events = self.trace._extract_event_data(positions=self.event_locations, 
                                                         before=self.add_points, after=self.window_size + self.add_points)

            self._get_event_properties()
            self.events = self.events - self.event_bsls[:, None]
            self.average_event_properties = self._get_average_event_properties()
            
            if eval:
                self._eval_events()


    def _get_average_event_decay(self) -> float:
        '''
        Returns the decay time constant of the averaged events.
        '''
        events_for_avg = self.events[self.singular_event_indices]

        event_x = np.arange(0, events_for_avg.shape[1]) * self.trace.sampling
        event_avg = np.average(events_for_avg, axis=0) * self.event_direction
        if events_for_avg.shape[0] < 4:
            fit_start = np.argmax(np.convolve(event_avg, np.ones(5) / 5, mode='same')) + int(0.01 * self.window_size)
        else:
            fit_start = np.argmax(event_avg) + int(0.01 * self.window_size)
        if fit_start > events_for_avg.shape[1] - int(0.2 * self.window_size): # not a valid starting point
            return np.full(3, np.nan)
        try:
            self.avg_decay_fit_start = fit_start
            fit, _ = curve_fit(exp_fit, event_x[fit_start:], event_avg[fit_start:],
                               p0=[np.amax(event_avg) + 1, events_for_avg.shape[1] / 50 * self.trace.sampling, 0],
                               bounds=([0, 0, -np.inf], [np.inf, 1e3, np.inf]))
            return fit
        except RuntimeError:
            return np.full(3, np.nan)


    def _eval_events(self) -> None:
        ''' Evaluates events. Calculates mean, std and median of amplitudes & charge, as well as decay tau and
        frequency of events. Results are stored as EventStats object in self.event_stats.
        In addition, times of event peaks, onset and half decay are calculated. 
        '''
        if not self.events_present():
            return
        if not len(self.singular_event_indices):
            self.singular_event_indices = np.array(list(range(self.event_locations.shape[0])))

        self.avg_decay_fit = self._get_average_event_decay()
        self.event_stats = EventStats(amplitudes=self.event_peak_values - self.event_bsls,
                                      scores=self.event_scores,
                                      tau=self.avg_decay_fit[1],
                                      charges=self.charges,
                                      risetimes=self.risetimes,
                                      slopes=self.slopes,
                                      decaytimes=self.decaytimes,
                                      halfwidths=self.halfwidths,
                                      time=self.trace.total_time,
                                      unit=self.trace.y_unit)
        
        self.event_peak_times = self.event_peak_locations * self.trace.sampling
        self.half_decay_times = self.half_decay * self.trace.sampling
        self.event_start_times = self.event_start * self.trace.sampling
        self.interevent_intervals = np.diff(self.event_peak_times, prepend=np.nan)

        if self.verbose:
            self.event_stats.print()

    
    def delete_events(self, event_indices: list=[], eval: bool=True) -> None:
        '''
        Deletes events from the event list. The indices of the events to be deleted are passed as an array.

        Parameters
        ----------
        event_indices: list
            Indices of the events to be deleted.
        eval: bool
            Whether to re-evaluate the events after deletion. If False, the event statistics will not be updated.

        Raises
        ------
        ValueError
            When the event index does not exist.
        '''
        if not self.events_present():
            return
        
        if not hasattr(self, 'event_stats'):
            self._eval_events()

        for event in event_indices:
            if event < 0 or event >= self.event_locations.shape[0]:
                raise ValueError(f'Event {event} does not exist.')

        self.event_locations = np.delete(self.event_locations, event_indices, axis=0)
        self.event_peak_locations = np.delete(self.event_peak_locations, event_indices, axis=0)
        self.event_peak_times = np.delete(self.event_peak_times, event_indices, axis=0)
        self.event_peak_values = np.delete(self.event_peak_values, event_indices, axis=0)
        self.event_start = np.delete(self.event_start, event_indices, axis=0)
        self.decaytimes = np.delete(self.decaytimes, event_indices, axis=0)
        self.risetimes = np.delete(self.risetimes, event_indices, axis=0)
        self.charges = np.delete(self.charges, event_indices, axis=0)
        self.event_bsls = np.delete(self.event_bsls, event_indices, axis=0)
        self.bsl_starts = np.delete(self.bsl_starts, event_indices, axis=0)
        self.bsl_ends = np.delete(self.bsl_ends, event_indices, axis=0)
        self.min_positions_rise = np.delete(self.min_positions_rise, event_indices, axis=0)
        self.max_positions_rise = np.delete(self.max_positions_rise, event_indices, axis=0)
        self.half_decay = np.delete(self.half_decay, event_indices, axis=0)
        self.events = np.delete(self.events, event_indices, axis=0)
        self.event_scores = np.delete(self.event_scores, event_indices, axis=0)
        
        self.deleted_events += len(event_indices)

        if eval:
            self.detection._get_singular_event_indices()
            self.detection._eval_events()


    def save_to_h5(self, filename: str, include_prediction: bool=False) -> None:
        ''' Save detection results to an hdf5 (.h5) file.
        
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
            f.create_dataset('event_params/event_charges', data=self.event_stats.charges)
            f.create_dataset('event_params/event_risetimes', data=self.event_stats.risetimes)
            f.create_dataset('event_params/event_halfdecays', data=self.event_stats.halfdecays)
            f.create_dataset('event_params/event_halfwidths', data=self.event_stats.halfwidths)
            f.create_dataset('event_params/event_bsls', data=np.array(self.event_bsls))
            f.create_dataset('event_params/event_intervals', data=np.array(self.interevent_intervals))
            f.create_dataset('event_statistics/amplitude_average', data=self.event_stats.mean(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/amplitude_stdev', data=self.event_stats.std(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/amplitude_median', data=self.event_stats.median(self.event_stats.amplitudes))
            f.create_dataset('event_statistics/charge_mean', data=self.event_stats.mean(self.event_stats.charges))
            f.create_dataset('event_statistics/charge_median', data=self.event_stats.median(self.event_stats.charges))
            f.create_dataset('event_statistics/risetime_mean', data=self.event_stats.mean(self.event_stats.risetimes))
            f.create_dataset('event_statistics/risetime_median', data=self.event_stats.median(self.event_stats.risetimes))
            f.create_dataset('event_statistics/decaytime_mean', data=self.event_stats.mean(self.event_stats.halfdecays))
            f.create_dataset('event_statistics/decaytime_median', data=self.event_stats.median(self.event_stats.halfdecays))
            f.create_dataset('event_statistics/halfwidth_mean', data=self.event_stats.mean(self.event_stats.halfwidths))
            f.create_dataset('event_statistics/halfwidth_median', data=self.event_stats.median(self.event_stats.halfwidths))
            f.create_dataset('event_statistics/decay_from_fit', data=self.event_stats.avg_tau_decay)
            f.create_dataset('event_statistics/frequency', data=self.event_stats.frequency())
            f.create_dataset('event_statistics/iei_mean', data=self.event_stats.mean(self.interevent_intervals))
            f.create_dataset('event_statistics/iei_median', data=self.event_stats.median(self.interevent_intervals))

            f.attrs['amplitude_unit'] = self.trace.y_unit
            f.attrs['recording_time'] = self.trace.data.shape[0] * self.trace.sampling
            f.attrs['source_filename'] = self.trace.filename
            f.attrs['miniml_model'] = self.model_path
            f.attrs['miniml_model_threshold'] = self.model_threshold
            f.attrs['minimum_peak'] = self.peak_w
            f.attrs['stride'] = self.stride_length
            f.attrs['window'] = self.window_size
            f.attrs['event_direction'] = self.event_direction
            if self.convolve_win > 0:
                f.attrs['convolve_win'] = self.convolve_win
            else:
                f.attrs['filter_factor'] = self.filter_factor
            f.attrs['gradient_convolve_win'] = self.gradient_convolve_win
            f.attrs['relative_prominence'] = self.rel_prom_cutoff
            f.attrs['deleted_events'] = self.deleted_events

            if include_prediction:
                f.create_dataset('prediction', data=self.prediction)
        print(f'Events saved to {filename}')


    def save_to_csv(self, filename: str='') -> None:
        ''' Save detection results to a .csv file. Generates two files, one with averages and one with the values for the individual events.
        Filenames are automatically generated.
        
        filename: str
            filename, including path. Results will be split into "filename + _avgs.csv" and "filename + _individual.csv"
        '''
        if filename.endswith('.csv'):
            filename = filename.removesuffix('.csv')
        
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
            self.event_stats.halfdecays,
            self.event_stats.halfwidths,
            self.interevent_intervals))
        
        avgs = np.array((
            self.event_stats.mean(self.event_stats.amplitudes),
            self.event_stats.std(self.event_stats.amplitudes),
            self.event_stats.median(self.event_stats.amplitudes),
            self.event_stats.mean(self.event_stats.charges),
            self.event_stats.mean(self.event_stats.risetimes),
            self.event_stats.mean(self.event_stats.halfdecays),
            self.event_stats.mean(self.event_stats.halfwidths),
            self.event_stats.avg_tau_decay,
            self.event_stats.frequency(),
            self.event_stats.mean(self.interevent_intervals)))
        
        column_names = [f'event_{i}' for i in range(len(self.event_locations))]

        individual = pd.DataFrame(individual, index=['location', 'score', 'amplitude', 'charge', 'risetime', 'decaytime', 'halfwidth', 'interval'], columns=column_names)
        avgs = pd.DataFrame(avgs, index=['amplitude mean', 'amplitude std', 'amplitude median', 'charge mean', 'risetime mean', 'decaytime mean', 'halfwidth mean', 'tau_avg', 'frequency', 'iei mean'], columns=['value'])
        
        individual.to_csv(f'{filename}_individual.csv')
        avgs.to_csv(f'{filename}_avgs.csv', header=False)
        print(f'events saved to {filename}_avgs.csv and {filename}_individual.csv')


    def save_to_pickle(self, filename: str='', include_prediction: bool=True, include_data: bool=True) -> None:
        ''' Save detection results to a .pickle file.         

        Parameters
        ------
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
                'min_values_rise':self.min_values_rise,
                'max_values_rise':self.max_values_rise,
                'half_decay_positions':self.half_decay,},
             
            'individual_values':{
                'amplitudes':self.event_stats.amplitudes,
                'charges':self.event_stats.charges,
                'risetimes':self.event_stats.risetimes,
                'half_decaytimes':self.event_stats.halfdecays,
                'event_intervals':self.interevent_intervals,            
                'halfwidths':self.event_stats.halfwidths},
            
            'average_values':{
                'amplitude mean':self.event_stats.mean(self.event_stats.amplitudes),
                'amplitude std':self.event_stats.std(self.event_stats.amplitudes),
                'amplitude median':self.event_stats.median(self.event_stats.amplitudes),
                'charge mean':self.event_stats.mean(self.event_stats.charges),
                'risetime mean':self.event_stats.mean(self.event_stats.risetimes),
                'half_decaytime mean':self.event_stats.mean(self.event_stats.halfdecays),
                'halfwidth mean':self.event_stats.mean(self.event_stats.halfwidths),
                'decay_tau':self.event_stats.avg_tau_decay*1000,
                'frequency':self.event_stats.frequency(),
                'iei_mean':self.event_stats.mean(self.interevent_intervals),
                'iei_median':self.event_stats.median(self.interevent_intervals)},
            
            'average_event_properties':self.average_event_properties,
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
                'filter_factor':self.filter_factor,
                'gradient_convolve_win':self.gradient_convolve_win,
                'min_peak_w':self.peak_w,
                'rel_prom_cutoff':self.rel_prom_cutoff,
                'event_direction':self.event_direction,
                'deleted_events':self.deleted_events},
            'events':self.events}

        if include_prediction:
            results['prediction']=self.prediction # Save prediction as numpy array

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
    verbose: int
        Verbosity level
    event_positions: np.ndarray or list
        The position(s) of detected events.
    filter_factor: int
        Fraction of the sampling rate used to lowpass filter the data for analysis.
    convolve_win: int
        Window size for the hanning window used to filter the data for event analysis.
    resampling_factor: float
        The factor by which to resample the data.

    Methods
    ----------
    eval_events(): 
        Perform event analysis.
    '''
    def __init__(self, trace, window_size, event_direction, verbose, event_positions, filter_factor, convolve_win, gradient_convolve_win, resampling_factor):
        super().__init__(data=trace, window_size=window_size, event_direction=event_direction, verbose=verbose)
        self.add_points = int(self.window_size/3)
        self.event_direction = event_direction
        self.filter_factor = filter_factor
        self.convolve_win = convolve_win
        self.resampling_factor = resampling_factor

        self.event_locations = event_positions[np.logical_and(
                                                self.add_points < event_positions, 
                                                event_positions < len(self.trace.data) - (self.window_size + self.add_points))]
        self.event_locations = self.event_locations.astype(np.int64)
        self.events = self.trace._extract_event_data(self.event_locations, before=self.add_points, 
                                                     after=self.window_size + self.add_points)
        self.gradeint_convolve_win = gradient_convolve_win
        
    def eval_events(self, filter: bool=True) -> None:
        if self.event_locations.shape[0] > 0:
            super()._get_event_properties(filter=filter)
            self.events = self.events - self.event_bsls[:, None]
            super()._eval_events()

