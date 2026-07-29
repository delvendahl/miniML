from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pyabf
from scipy import signal


class MiniTrace:
    """miniML class for a time series data trace containing synaptic events. Data are stored as float64 numpy ndarray.

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
    """

    def __init__(
        self,
        data: np.ndarray | list | None = None,
        sampling_interval: float = 1,
        y_unit: str = "",
        filename: str = "",
    ) -> None:
        if data is None:
            data = []
        self.data = data
        self.sampling = sampling_interval
        self.events = []
        self.y_unit = y_unit
        self.filename = filename

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, data: np.ndarray | list[float] | list[int]) -> None:
        # ensure data is float64 to avoid issues with minmax_scale
        self._data = np.array(data).astype(np.float64)

    @property
    def sampling(self) -> float:
        return self._sampling

    @sampling.setter
    def sampling(self, value: float) -> None:
        if value < 0:
            raise ValueError("Sampling interval must be positive")
        self._sampling = value

    @property
    def sampling_rate(self) -> float:
        return np.round(1 / self.sampling)

    @property
    def time_axis(self) -> np.ndarray:
        """Returns time axis as numpy array"""
        return np.arange(len(self.data)) * self.sampling

    @property
    def total_time(self) -> float:
        """Returns the total duration of the recording"""
        return len(self.data) * self.sampling

    @classmethod
    def from_h5_file(
        cls,
        filename: str,
        tracename: str = "mini_data",
        scaling: float = 1e12,
        sampling: float = 2e-5,
        unit: str = "pA",
    ) -> MiniTrace:
        """Loads data from an hdf5 file. Name of the dataset needs to be specified.

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
        """
        with h5py.File(filename, "r") as f:
            path = f.visit(
                lambda key: (
                    key
                    if isinstance(f[key], h5py.Dataset)
                    and key.split("/")[-1] == tracename
                    else None
                )
            )
            if path is None:
                raise FileNotFoundError("Trace not found in file")
            data = f[path][:] * scaling

        return cls(
            data=data,
            sampling_interval=sampling,
            y_unit=unit,
            filename=Path(filename).name,
        )

    @classmethod
    def from_heka_file(
        cls,
        filename: str,
        rectype: str,
        group: int = 0,
        load_series: list = [],
        exclude_series: list = [],
        exclude_sweeps: dict = {},
        scaling: float = 1,
        unit: str = "",
        resample: bool = True,
    ) -> MiniTrace:
        """Loads data from a HEKA .dat file. Name of the PGF sequence needs to be specified.

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
        ValueError
            If the file is not a valid .dat file.
        IndexError
            When the group index is out of range.
        ValueError
            When the sampling rates of different series mismatch and resampling is set to False.
        """
        if not Path(filename).suffix.lower() == ".dat":
            raise ValueError("Incompatible file type. Method only loads .dat files.")

        from miniml.fileio import heka_reader as heka

        bundle = heka.Bundle(filename)

        if group < 0 or group > len(bundle.pul.children) - 1:
            raise IndexError("Group index out of range")

        bundle_series = {}
        for i, SeriesRecord in enumerate(bundle.pul[group].children):
            bundle_series.update({i: SeriesRecord.Label})

        if load_series == []:
            series_list = [
                series_number
                for series_number, record_type in bundle_series.items()
                if record_type == rectype and series_number not in exclude_series
            ]
        else:
            load_series = [x for x in load_series if x not in exclude_series]
            series_list = [
                series_number
                for series_number, record_type in bundle_series.items()
                if record_type == rectype and series_number in load_series
            ]

        series_data = []
        series_resistances = []
        for series in series_list:
            sweep_data = []
            for sweep in range(len(bundle.pul[group][series])):
                if series not in exclude_sweeps:
                    sweep_data.append(bundle.data[group, series, sweep, 0])
                else:
                    if sweep not in exclude_sweeps[int(series)]:
                        sweep_data.append(bundle.data[group, series, sweep, 0])
            pgf_series_index = (
                sum(len(bundle.pul[i].children) for i in range(group)) + series
            )
            series_data.append(
                (
                    np.array(sweep_data).flatten(),
                    bundle.pgf[pgf_series_index].SampleInterval,
                )
            )
            series_resistances.append(
                (1 / bundle.pul[group][series][0][0].GSeries) * 1e-6
            )

        max_sampling_interval = max([el[1] for el in series_data])
        data = np.array([], dtype=np.float64)
        for i, dat in enumerate(series_data):
            if dat[1] < max_sampling_interval:
                if not resample:
                    raise ValueError(
                        f"Sampling interval of series {i} is smaller than maximum sampling interval of all series"
                    )
                step = int(max_sampling_interval / dat[1])
                data = np.append(data, dat[0][::step])
            else:
                data = np.append(data, dat[0])

        data_unit = unit if unit else bundle.pul[group][series_list[0]][0][0].YUnit

        # MiniTrace.excluded_sweeps = exclude_sweeps
        # MiniTrace.exlucded_series = exclude_series
        # MiniTrace.Rseries = series_resistances

        bundle.close()

        return cls(
            data=data * scaling,
            sampling_interval=max_sampling_interval,
            y_unit=data_unit,
            filename=Path(filename).name,
        )

    @classmethod
    def from_axon_file(
        cls, filename: str, channel: int = 0, scaling: float = 1.0, unit: str = ""
    ) -> MiniTrace:
        """Loads data from an AXON .abf file.

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
        """
        if not Path(filename).suffix.lower() == ".abf":
            raise Exception("Incompatible file type. Method only loads .abf files.")

        abf_file = pyabf.ABF(filename)
        if channel not in abf_file.channelList:
            raise IndexError("Selected channel does not exist.")

        data_unit = unit if unit else abf_file.adcUnits[channel]

        return cls(
            data=abf_file.data[channel] * scaling,
            sampling_interval=1 / abf_file.sampleRate,
            y_unit=data_unit,
            filename=Path(filename).name,
        )

    def plot_trace(self) -> None:
        """Plots the trace"""
        plt.plot(self.time_axis, self.data)
        plt.xlabel("Time [s]")
        plt.ylabel(f"[{self.y_unit}]")
        plt.show()

    def detrend(self, detrend_type: str = "linear", num_segments: int = 0) -> MiniTrace:
        """Detrend the data.

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
        """
        num_data = self.data.shape[0]
        breaks = (
            np.arange(
                num_data / num_segments,
                num_data,
                num_data / num_segments,
                dtype=np.int64,
            )
            if num_segments > 1
            else 0
        )
        detrended = signal.detrend(self.data, bp=breaks, type=detrend_type)

        return MiniTrace(
            detrended, self.sampling, y_unit=self.y_unit, filename=self.filename
        )

    def filter(
        self,
        line_freq: float | None = None,
        width: float | None = None,
        highpass: float | None = None,
        lowpass: float | None = None,
        order: int = 4,
        savgol: float | None = None,
        hann: int | None = None,
    ) -> MiniTrace:
        """Filters trace with a combination of line frequency, high- and lowpass filters.
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
        """
        filtered_data = self.data.copy()
        nyq = 0.5 * self.sampling_rate

        if line_freq is not None:
            if width is None:
                raise ValueError("Width must be specified for line noise filtering.")

            from scipy.fftpack import irfft, rfft, rfftfreq

            fft = rfft(filtered_data)
            xf = rfftfreq(filtered_data.shape[0], 1 / self.sampling_rate)
            multiples = 6
            for freq in np.arange(line_freq, (multiples * line_freq), line_freq):
                fft[
                    np.where(xf > freq - width / 2)[0][0] : np.where(
                        xf > freq + width / 2
                    )[0][0]
                ] = 0

            filtered_data = irfft(fft)
        if highpass is not None:
            sos = signal.butter(order, highpass / nyq, btype="high", output="sos")
            filtered_data = signal.sosfilt(sos, filtered_data)
        if lowpass is not None:
            if savgol is not None:
                print(
                    "Warning: Two lowpass filters selected, Savgol filter is ignored."
                )
            sos = signal.butter(
                order, lowpass / nyq, btype="low", analog=False, output="sos", fs=None
            )
            filtered_data = signal.sosfiltfilt(sos, filtered_data)
        elif savgol is not None:
            filtered_data = signal.savgol_filter(
                filtered_data, int(savgol / 1000 / self.sampling), polyorder=order
            )
        elif hann is not None:
            win = signal.windows.hann(hann)
            filtered_data = signal.convolve(filtered_data, win, mode="same") / sum(win)
            # Hann window generates edge artifacts due to zero-padding. Retain unfiltered data at edges.
            filtered_data[:hann] = self.data[:hann]
            filtered_data[filtered_data.shape[0] - hann : filtered_data.shape[0]] = (
                self.data[filtered_data.shape[0] - hann : filtered_data.shape[0]]
            )

        return MiniTrace(
            filtered_data,
            sampling_interval=self.sampling,
            y_unit=self.y_unit,
            filename=self.filename,
        )

    def resample(self, sampling_frequency: float | None = None) -> MiniTrace:
        """Resamples the data trace to the given frequency

        sampling_frequency: float
            Sampling frequency in Hz of the output data

        returns: MiniTrace
            A resampled MiniTrace object
        """
        if sampling_frequency is None:
            return self

        resampling_factor = np.round(self.sampling_rate / sampling_frequency, 2)
        resampled_data = signal.resample(
            self.data, int(self.data.shape[0] / resampling_factor)
        )
        new_sampling_interval = self.sampling * resampling_factor

        return MiniTrace(
            resampled_data,
            sampling_interval=new_sampling_interval,
            y_unit=self.y_unit,
            filename=self.filename,
        )

    def _extract_event_data(
        self, positions: np.ndarray, before: int, after: int
    ) -> np.ndarray:
        """
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
        """
        if np.any(positions - before < 0) or np.any(
            positions + after >= self.data.shape[0]
        ):
            raise ValueError("Cannot extract time windows exceeding input data size.")

        indices = positions + np.arange(-before, after)[:, None, None]

        return np.squeeze(self.data[indices].T, axis=1)
