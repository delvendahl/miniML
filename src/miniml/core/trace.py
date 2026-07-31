from __future__ import annotations

from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class MiniTrace:
    """
    Represent a synaptic time-series trace.

    Parameters
    ----------
    data : np.ndarray | list | None, optional
        Trace samples. Values are stored internally as ``float64``.
    sampling_interval : float, default=1
        Sampling interval of the trace in seconds.
    y_unit : str, default=""
        Physical unit of the signal amplitude.
    filename : str, default=""
        Source filename associated with the trace.

    Attributes
    ----------
    data : np.ndarray
        Trace samples stored as a one-dimensional ``float64`` array.
    sampling : float
        Sampling interval in seconds.
    events : list
        Event snippets associated with the trace.
    y_unit : str
        Physical unit of the signal amplitude.
    filename : str
        Source filename associated with the trace.
    """

    excluded_sweeps: ClassVar[dict[int, list[int]]] = {}
    excluded_series: ClassVar[list[int]] = []
    Rseries: ClassVar[list[float]] = []

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
        """
        Return the trace time axis.

        Returns
        -------
        np.ndarray
            Time values in seconds for each sample in the trace.
        """
        return np.arange(len(self.data)) * self.sampling

    @property
    def total_time(self) -> float:
        """
        Return the total recording duration.

        Returns
        -------
        float
            Recording duration in seconds.
        """
        return len(self.data) * self.sampling

    def plot_trace(self) -> None:
        """
        Plot the trace using Matplotlib.
        """
        plt.plot(self.time_axis, self.data)
        plt.xlabel("Time [s]")
        plt.ylabel(f"[{self.y_unit}]")
        plt.show()

    def detrend(self, detrend_type: str = "linear", num_segments: int = 0) -> MiniTrace:
        """
        Remove linear or constant trends from the trace.

        Parameters
        ----------
        detrend_type : str, default="linear"
            Detrending mode passed to ``scipy.signal.detrend``.
        num_segments : int, default=0
            Number of breakpoint segments used during detrending.

        Returns
        -------
        MiniTrace
            Detrended trace.
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
        """
        Filter the trace with one or more smoothing operations.

        If both ``lowpass`` and ``savgol`` are provided, only the low-pass
        filter is applied.

        Parameters
        ----------
        line_freq : float | None, optional
            Line-noise frequency in hertz.
        width : float | None, optional
            Width of the line noise filter (Hz).
        highpass : float | None, optional
            Highpass cutoff frequency (Hz).
        lowpass : float | None, optional
            Low-pass cutoff frequency in hertz.
        order : int, default=4
            Filter order.
        savgol : float | None, optional
            Time window for Savitzky-Golay smoothing in milliseconds.
        hann : int | None, optional
            Hann window length in samples.

        Returns
        -------
        MiniTrace
            Filtered trace.

        Raises
        ------
        ValueError
            If ``line_freq`` is provided without ``width``.
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
        """
        Resample the trace to a target sampling frequency.

        Parameters
        ----------
        sampling_frequency : float | None, optional
            Target sampling frequency in hertz.

        Returns
        -------
        MiniTrace
            Resampled trace. If ``sampling_frequency`` is None, the current
            instance is returned unchanged.
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
        Extract event windows from the trace.

        Parameters
        ----------
        positions : np.ndarray
            Event positions in samples.
        before : int
            Number of samples to include before each event position.
        after : int
            Number of samples to include after each event position.

        Returns
        -------
        np.ndarray
            Array of extracted event windows with shape
            ``(len(positions), before + after)``.

        Raises
        ------
        ValueError
            If any requested extraction window exceeds the trace bounds.
        """
        if np.any(positions - before < 0) or np.any(
            positions + after >= self.data.shape[0]
        ):
            raise ValueError("Cannot extract time windows exceeding input data size.")

        indices = positions + np.arange(-before, after)[:, None, None]

        return np.squeeze(self.data[indices].T, axis=1)
