import pickle as pkl

import h5py
import keras
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.optimize import curve_fit

from miniml.core.functions import (
    get_event_baseline,
    legacy_get_event_baseline,
    get_event_charge,
    get_event_halfdecay_time,
    get_event_halfwidth,
    get_event_onset,
    get_event_peak,
    get_event_risetime,
)
from miniml.core.trace import MiniTrace
from miniml.core.util import exp_fit, minmax_scaling, robust_noise_mad
from miniml.fileio.util import is_keras_model


class EventStats:
    """
    Store summary statistics for detected events.

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitudes of individual events.
    scores : np.ndarray
        Prediction scores of individual events.
    charges : np.ndarray
        Charge transfer of individual events.
    risetimes : np.ndarray
        10-90 percent rise times of individual events.
    slopes : np.ndarray
        Rise slopes of individual events.
    decaytimes : np.ndarray
        Half decay times of individual events.
    halfwidths : np.ndarray
        Half-width of individual events (seconds).
    tau : float
        Average decay time constant (seconds).
    time : float
        Total recording duration (seconds).
    unit : str
        Data unit.

    Attributes
    ----------
    amplitudes : np.ndarray
        Amplitudes of individual events.
    event_scores : np.ndarray
        Prediction scores of individual events.
    charges : np.ndarray
        Charge transfer of individual events.
    risetimes : np.ndarray
        10-90 percent rise times of individual events.
    slopes : np.ndarray
        Rise slopes of individual events.
    halfdecays : np.ndarray
        Half decay times of individual events.
    halfwidths : np.ndarray
        Half-widths of individual events.
    avg_tau_decay : float
        Average decay time constant.
    rec_time : float
        Total recording duration.
    y_unit : str
        Signal unit.
    event_count : int
        Number of detected events.
    """

    def __init__(
        self,
        amplitudes: np.ndarray,
        scores: np.ndarray,
        charges: np.ndarray,
        risetimes: np.ndarray,
        slopes: np.ndarray,
        decaytimes: np.ndarray,
        halfwidths: np.ndarray,
        tau,
        time,
        unit: str,
    ) -> None:
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
        """
        Return the mean of an event metric, ignoring NaN values.

        Parameters
        ----------
        values : np.ndarray
            Event metric values.

        Returns
        -------
        float
            Mean value, or ``np.nan`` if the input is empty or all-NaN.
        """
        if ~np.all(np.isnan(values)) and self.event_count:
            return np.nanmean(values).item()
        else:
            return float("nan")

    def std(self, values: np.ndarray) -> float:
        """
        Return the sample standard deviation of an event metric.

        Parameters
        ----------
        values : np.ndarray
            Event metric values.

        Returns
        -------
        float
            Sample standard deviation, or ``np.nan`` when fewer than two values
            are available.
        """
        if values.shape[0] > 1:
            return np.nanstd(values, ddof=1).item()

        return float("nan")

    def median(self, values: np.ndarray) -> float:
        """
        Return the median of an event metric, ignoring NaN values.

        Parameters
        ----------
        values : np.ndarray
            Event metric values.

        Returns
        -------
        float
            Median value, or ``np.nan`` if the input is empty or all-NaN.
        """
        if ~np.all(np.isnan(values)) and self.event_count:
            return np.nanmedian(values).item()
        else:
            return float("nan")

    def cv(self, values: np.ndarray) -> float:
        """
        Return the coefficient of variation of an event metric.

        Parameters
        ----------
        values : np.ndarray
            Event metric values.

        Returns
        -------
        float
            Absolute ratio of standard deviation to mean.
        """
        return float(abs(self.std(values) / self.mean(values)))

    def frequency(self) -> float:
        """
        Return the detected event frequency.

        Returns
        -------
        float
            Event frequency in hertz.
        """
        return float(len(self.amplitudes) / self.rec_time)

    def print(self) -> None:
        """
        Print event summary statistics to standard output.
        """
        print("\nEvent statistics:\n-------------------------")
        print(f"    Number of events: {self.event_count}")
        print(f"    Average score: {self.mean(self.event_scores):.3f}")
        print(f"    Event frequency: {self.frequency():.4f} Hz")
        print(f"    Mean amplitude: {self.mean(self.amplitudes):.4f} {self.y_unit}")
        print(f"    Median amplitude: {self.median(self.amplitudes):.4f} {self.y_unit}")
        print(f"    Std amplitude: {self.std(self.amplitudes):.4f} {self.y_unit}")
        print(f"    CV amplitude: {self.cv(self.amplitudes):.3f}")
        print(f"    Mean charge: {self.mean(self.charges):.5f} pC")
        print(f"    CV charge: {self.cv(self.charges):.3f}")
        print(f"    Mean 10-90 risetime: {self.mean(self.risetimes) * 1000:.3f} ms")
        print(f"    Mean half decay time: {self.mean(self.halfdecays) * 1000:.3f} ms")
        print(f"    Mean half-width: {self.mean(self.halfwidths) * 1000:.3f} ms")
        print(f"    Tau decay: {self.avg_tau_decay * 1000:.3f} ms")
        print("-------------------------")


class EventDetection:
    """
    Detect and analyze synaptic events in a trace.

    Parameters
    ----------
    data : MiniTrace
        Trace to analyze.
    window_size : int, default=600
        Detection window size in samples.
    event_direction : str, default="negative"
        Event polarity in the trace.
    training_direction : str, default="negative"
        Event polarity used when the model was trained.
    verbose : int, default=1
        Verbosity level for prediction and reporting.
    batch_size : int, default=128
        Batch size used by ``model.predict``.
    model : keras.Model | None, optional
        Model instance to use for event detection.
    model_path : str, default=""
        Path to a saved Keras model.
    model_threshold : float, default=0.5
        Minimum model prediction peak height required for event detection.
    compile_model : bool, default=True
        Whether to compile the model.
    callbacks : list | None, optional
        Callbacks passed to ``model.predict``.

    Attributes
    ----------
    trace : MiniTrace
        Source trace being analyzed.
    prediction : np.ndarray
        Model prediction trace.
    event_locations : np.ndarray
        Detected event onset locations in samples.
    event_scores : np.ndarray
        Prediction scores associated with detected events.
    event_peak_locations : np.ndarray
        Event peak locations in samples.
    event_peak_times : np.ndarray
        Event peak times in seconds.
    events : np.ndarray
        Extracted event windows.
    event_stats : EventStats
        Summary statistics for the detected events.
    """

    def __init__(
        self,
        data: MiniTrace,
        window_size: int = 600,
        event_direction: str = "negative",
        training_direction: str = "negative",
        verbose: int = 1,
        batch_size: int = 128,
        model: keras.Model | None = None,
        model_path: str = "",
        model_threshold: float = 0.5,
        compile_model: bool = True,
        callbacks: list | None = None,
    ) -> None:
        self.trace = data
        self.prediction: np.ndarray = np.array([])
        self.window_size = window_size
        self.event_direction = event_direction
        self.training_direction = training_direction
        self.verbose = verbose
        self.event_locations = np.array([])
        self.event_scores = np.array([])
        self.events = np.array([])
        self.batch_size = batch_size
        self.model_path = model_path
        self.model: keras.Model
        self.model_threshold = None
        if model is not None:
            self.model = model
            self.model_threshold = model_threshold
            self.callbacks = callbacks or []
        elif model_path:
            self.load_model(
                filepath=model_path,
                threshold=model_threshold,
                compile=compile_model,
            )
            self.callbacks = callbacks or []
        self.deleted_events = 0

    @property
    def event_direction(self):
        return self._event_direction

    @event_direction.setter
    def event_direction(self, event_direction_str: str):
        self._event_direction = -1 if event_direction_str.lower() == "negative" else 1

    @property
    def training_direction(self):
        return self._training_direction

    @training_direction.setter
    def training_direction(self, training_direction_str: str):
        self._training_direction = (
            -1 if training_direction_str.lower() == "negative" else 1
        )

    def events_present(self) -> bool:
        """
        Check whether any event windows are currently stored.

        Returns
        -------
        bool
            True if at least one event is present.
        """
        num_events = self.events.shape[0]

        return num_events != 0

    def load_model(
        self, filepath: str, threshold: float = 0.5, compile: bool = True
    ) -> None:
        """
        Load a trained miniML model from an HDF5 file.

        Parameters
        ----------
        filepath : str
            Path to the saved Keras model.
        threshold : float, default=0.5
            Prediction threshold used during peak detection.
        compile : bool, default=True
            Whether to compile the loaded model.

        Raises
        ------
        ValueError
            If the file does not contain a valid Keras model.
        """
        if not is_keras_model(filepath):
            raise ValueError("Model file is not a valid Keras model")
        self.model: keras.Model = keras.models.load_model(filepath, compile=compile)
        self.model_threshold = threshold
        if self.verbose:
            print(f"Model loaded from {filepath}")

    def lowpass_filter(
        self, data: np.ndarray, cutoff: float, order: int = 4
    ) -> np.ndarray:
        """
        Apply a Butterworth low-pass filter.

        Parameters
        ----------
        data : np.ndarray
            Input trace to filter.
        cutoff : float
            Cutoff frequency in hertz.
        order : int, default=4
            Filter order.

        Returns
        -------
        np.ndarray
            Filtered trace.
        """
        nyq = 0.5 * self.trace.sampling_rate

        sos = signal.butter(
            order, cutoff / nyq, btype="lowpass", analog=False, output="sos", fs=None
        )

        return signal.sosfiltfilt(sos, data)

    def hann_filter(self, data: np.ndarray, filter_size: int) -> np.ndarray:
        """
        Apply a Hann-window smoothing filter.

        Parameters
        ----------
        data : np.ndarray
            Input trace to filter.
        filter_size : int
            Hann window size.

        Returns
        -------
        np.ndarray
            Smoothed trace with unfiltered edges preserved to reduce padding artifacts.
        """
        if filter_size == 0:
            return data
        win = signal.windows.hann(filter_size)
        filtered_data = signal.convolve(data, win, mode="same") / sum(win)
        filtered_data[:filter_size] = data[:filter_size]
        filtered_data[-filter_size:] = data[-filter_size:]

        return filtered_data

    def _linear_interpolation(
        self, data: np.ndarray, interpol_to_len: int
    ) -> tuple[np.ndarray, float]:
        """
        Interpolate a data segment to a target number of samples.

        Parameters
        ----------
        data : np.ndarray
            Input samples.
        interpol_to_len : int
            Desired output length.

        Returns
        -------
        tuple[np.ndarray, float]
            Interpolated data and the applied resampling factor.
        """
        x = np.arange(0, data.shape[0])
        x_interpol = np.linspace(0, data.shape[0], interpol_to_len)

        interpol_factor = len(x_interpol) / len(x)
        data_interpolated = np.interp(
            x_interpol, x, data, left=None, right=None, period=None
        )

        return data_interpolated, interpol_factor

    def __predict(self) -> None:
        """
        Run model inference on the trace using a sliding window.

        Raises
        ------
        ValueError
            If the derived stride is invalid.
        """
        # resample values for prediction:
        data = signal.resample(
            self.trace.data, round(len(self.trace.data) * self.resampling_factor)
        )

        # invert the trace if event_direction and training_direction are different.
        if self.event_direction != self.training_direction:
            data *= -1

        win_size = round(self.window_size * self.resampling_factor)
        stride = round(self.stride_length * self.resampling_factor)

        if stride <= 0 or stride > win_size:
            raise ValueError("Invalid stride")

        ds = keras.utils.timeseries_dataset_from_array(
            data=np.expand_dims(data, axis=1).astype(np.float32),
            targets=None,
            sequence_length=win_size,
            sequence_stride=stride,
            batch_size=None,
            shuffle=False,
        )

        ds = ds.map(minmax_scaling, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        self.prediction = tf.squeeze(
            self.model.predict(ds, verbose=self.verbose, callbacks=self.callbacks)
        )

    def _interpolate_prediction_trace(self) -> tuple[np.ndarray, float]:
        """
        Interpolate the prediction trace back onto raw-data coordinates.

        Returns
        -------
        tuple[np.ndarray, float]
            Interpolated prediction trace and the interpolation factor.
        """
        stride = round(self.stride_length * self.resampling_factor)
        pn = len(self.prediction) - 1
        pn_mapped = pn * stride
        pn_in_raw_data = round(pn_mapped / self.resampling_factor)
        resampled_prediction, interpol_factor = self._linear_interpolation(
            data=self.prediction, interpol_to_len=pn_in_raw_data
        )

        return resampled_prediction, interpol_factor

    def _get_prediction_peaks(
        self, peak_w: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Find candidate event regions in the prediction trace.

        Parameters
        ----------
        peak_w : int, default=10
            Minimum peak width used during prediction peak detection.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Start indices, end indices, and peak scores for candidate events.
        """
        filtered_prediction = maximum_filter1d(
            self.prediction, size=int(5 * self.interpol_factor), origin=-2
        )

        _, peak_properties = signal.find_peaks(
            x=filtered_prediction,
            height=self.model_threshold,
            prominence=self.model_threshold,
            width=peak_w * self.interpol_factor,
        )

        start_pnts = np.array(
            peak_properties["left_ips"] + self.window_size / 4, dtype=np.int64
        )
        # check if start_pnts are larger than right_ips and limit to this value minus buffer of a quarter of peak width
        boolean_indices = start_pnts > peak_properties["right_ips"]
        start_pnts[boolean_indices] = (
            peak_properties["right_ips"][boolean_indices]
            - peak_properties["widths"][boolean_indices] / 4
        )
        end_pnts = np.array(
            peak_properties["right_ips"] + self.window_size / 2, dtype=np.int64
        )
        scores = peak_properties["peak_heights"]

        return start_pnts, end_pnts, scores

    def _make_smth_gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate raw and smoothed first-derivative traces.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Raw gradient and smoothed gradient arrays.
        """
        # filter raw data trace, calculate gradient and filter first derivative trace
        if self.convolve_win > 0:
            trace_convolved = self.hann_filter(
                data=self.trace.data, filter_size=self.convolve_win * 2
            )
        else:
            trace_convolved = self.lowpass_filter(
                data=self.trace.data,
                cutoff=self.trace.sampling_rate / (self.filter_factor * 1.5),
                order=4,
            )
        trace_convolved *= self.event_direction  # (-1 = 'negative', 1 else)

        gradient = np.gradient(trace_convolved, self.trace.sampling)
        # gradient[:int(self.convolve_win * 1.5)] = 0
        # gradient[-int(self.convolve_win * 1.5):] = 0

        smth_gradient = self.hann_filter(
            data=gradient, filter_size=self.gradient_convolve_win
        )
        smth_gradient[: self.gradient_convolve_win] = 0
        smth_gradient[-self.gradient_convolve_win :] = 0

        return gradient, smth_gradient

    def _get_grad_threshold(
        self,
        gradient: np.ndarray,
        start_pnts: np.ndarray,
        end_pnts: np.ndarray,
        multiplier: float = 4.0,
    ) -> int:
        """
        Estimate a derivative threshold from event-free trace segments.

        Parameters
        ----------
        gradient : np.ndarray
            Gradient trace.
        start_pnts : np.ndarray
            Candidate event start indices.
        end_pnts : np.ndarray
            Candidate event end indices.
        multiplier : float, default=4.0
            Multiplier for calculating the threshold based on MAD.

        Returns
        -------
        int
            Gradient threshold derived from the standard deviation of event-free
            segments.
        """
        split_data = np.split(gradient, np.vstack((start_pnts, end_pnts)).ravel("F"))
        event_free_data = np.concatenate(split_data[::2]).ravel()

        grad_threshold = robust_noise_mad(event_free_data, multiplier=multiplier)[0]

        return grad_threshold

    def _find_event_locations(
        self, limit: int, scores: np.ndarray, rel_prom_cutoff: float = 0.25
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Refine approximate event positions from prediction peaks.

        Parameters
        ----------
        limit : int
            Right-edge limit used to reject border events.
        scores : np.ndarray
            Prediction values associated with candidate events.
        rel_prom_cutoff : float, default=0.25
            Minimum relative prominence used to keep overlapping derivative peaks.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Event locations and their corresponding prediction scores.
        """
        # Remove indices at left and right borders to prevent boundary issues.
        mask = (self.start_pnts > self.window_size) & (
            self.end_pnts < self.prediction.shape[0]
        )

        self.end_pnts = self.end_pnts[mask]
        self.start_pnts = self.start_pnts[mask]
        scores = scores[mask]

        event_locations, event_scores = [], []
        for i, _ in enumerate(self.start_pnts):
            peaks, peak_params = signal.find_peaks(
                x=self.smth_gradient[self.start_pnts[i] : self.end_pnts[i]],
                height=self.grad_threshold,
                prominence=self.grad_threshold,
            )

            if (
                peaks.shape[0] > 1
            ):  # If > 1 peak found; apply relative prominence cutoff
                rel_prom = peak_params["prominences"] / np.max(
                    peak_params["prominences"]
                )
                inds = np.argwhere(rel_prom >= rel_prom_cutoff).flatten()
                peaks = peaks[inds]
                for my_param in peak_params:
                    peak_params[my_param] = peak_params[my_param][inds]

            if not len(peaks):  # If no peak found: default argmax finding
                peaks = np.array(
                    [
                        np.argmax(
                            self.smth_gradient[self.start_pnts[i] : self.end_pnts[i]]
                        )
                    ]
                )

            for peak in peaks:
                if (self.start_pnts[i] + peak) >= (self.trace.data.shape[0] - limit):
                    continue
                if self.start_pnts[i] + peak not in event_locations:
                    event_locations.append(self.start_pnts[i] + peak)
                    event_scores.append(scores[i])

        return np.array(event_locations), np.array(event_scores)

    def _remove_duplicate_locations(self) -> None:
        """
        Remove duplicate or near-duplicate detected event locations.
        """
        unique_indices = np.unique(self.event_locations, return_index=True)[1]
        self.event_locations = self.event_locations[unique_indices]
        self.event_scores = self.event_scores[unique_indices]

        duplicate_indices = (
            np.argwhere(
                np.diff(self.event_locations) < self.window_size / 100
            ).flatten()
            + 1
        )
        self.event_locations = np.delete(self.event_locations, duplicate_indices)
        self.event_scores = np.delete(self.event_scores, duplicate_indices)
        self.event_locations = np.asarray(self.event_locations, dtype=np.int64)

    def _get_event_properties(
        self, filter: bool = True, use_legacy_baseline_method: bool = True
    ) -> None:
        """
        Extract detailed event properties required for downstream analysis.

        Parameters
        ----------
        filter : bool, default=True
            If True, derive properties from a filtered version of the trace.
        use_legacy_baseline_method : bool, default=True
            If True, use the legacy baseline estimator. Otherwise, use the newer
            baseline method.

        Raises
        ------
        ValueError
            If the requested extraction window exceeds the trace boundaries.
        """
        ### Prepare data
        diffs = np.diff(
            self.event_locations, append=self.trace.data.shape[0]
        )  # Difference in points between the event locations
        baseline_duration = int(
            self.window_size * 0.1
        )  # duration of baseline period before event peak
        after = self.window_size + self.add_points
        positions = self.event_locations

        ### Set parameters for charge calculation
        charge_factor = 4
        combined_charge_events = 1
        calculate_charge = False  # will be set to True in the loop if double event criteria are fulfilled; not a flag for charge

        if np.any(positions - self.add_points < 0) or np.any(
            positions + after >= self.trace.data.shape[0]
        ):
            raise ValueError("Cannot extract time windows exceeding input data size.")
        # Filter data if required
        if filter:
            if self.convolve_win > 0:
                mini_trace = self.hann_filter(
                    data=self.trace.data, filter_size=self.convolve_win
                )
            else:
                mini_trace = self.lowpass_filter(
                    data=self.trace.data,
                    cutoff=self.trace.sampling_rate / self.filter_factor,
                    order=4,
                )
        else:
            mini_trace = self.trace.data.copy()
        mini_trace *= self.event_direction

        # Initialize arrays to store event properties

        self.event_peak_locations = np.full_like(positions, -1, dtype=np.int64)
        self.bsl_starts = np.full_like(positions, -1, dtype=np.int64)
        self.bsl_ends = np.full_like(positions, -1, dtype=np.int64)
        self.event_start = np.full_like(positions, -1, dtype=np.int64)

        self.event_peak_values = np.full_like(positions, np.nan, dtype=np.float64)
        self.event_bsls = np.full_like(positions, np.nan, dtype=np.float64)
        self.event_bsl_durations = np.full_like(positions, np.nan, dtype=np.float64)
        self.decaytimes = np.full_like(positions, np.nan, dtype=np.float64)
        self.charges = np.full_like(positions, np.nan, dtype=np.float64)
        self.risetimes = np.full_like(positions, np.nan, dtype=np.float64)
        self.half_decay = np.full_like(positions, np.nan, dtype=np.float64)
        self.halfwidths = np.full_like(positions, np.nan, dtype=np.float64)
        self.rise_half_amp_times = np.full_like(positions, np.nan, dtype=np.float64)
        self.decay_half_amp_times = np.full_like(positions, np.nan, dtype=np.float64)

        self.min_positions_rise = np.full_like(positions, np.nan, dtype=np.float64)
        self.max_positions_rise = np.full_like(positions, np.nan, dtype=np.float64)
        self.min_values_rise = np.full_like(positions, np.nan, dtype=np.float64)
        self.max_values_rise = np.full_like(positions, np.nan, dtype=np.float64)

        for ix, position in enumerate(positions):
            indices = position + np.arange(-self.add_points, after)
            data = mini_trace[indices]

            event_peak_pos = get_event_peak(
                data=data,
                event_num=ix,
                add_points=self.add_points,
                window_size=self.window_size,
                diffs=diffs,
            )

            self.event_peak_locations[ix] = int(event_peak_pos)
            self.event_peak_values[ix] = np.mean(
                data[
                    event_peak_pos - self.peak_spacer : event_peak_pos
                    + self.peak_spacer
                ]
            )

            if use_legacy_baseline_method:
                baseline = legacy_get_event_baseline(
                    data=data,
                    duration=baseline_duration,
                    event_num=ix,
                    diffs=diffs,
                    add_points=self.add_points,
                    peak_positions=self.event_peak_locations,
                    positions=positions,
                )
            else:
                baseline = get_event_baseline(
                    data=data,
                    bsl_duration=baseline_duration,
                    event_num=ix,
                    relative_event_position=self.add_points,
                    positions=positions,
                )

            self.bsl_starts[ix] = baseline.start
            self.bsl_ends[ix] = baseline.end
            self.event_bsls[ix] = baseline.value
            self.event_bsl_durations[ix] = baseline.duration

            onset_position = get_event_onset(
                data=data,
                peak_position=event_peak_pos,
                baseline=baseline.value,
                baseline_var=baseline.var,
            )
            self.event_start[ix] = onset_position

            (
                risetime,
                min_position_rise,
                min_value_rise,
                max_position_rise,
                max_value_rise,
            ) = get_event_risetime(
                data=data[baseline.start : int(event_peak_pos)],
                sampling_rate=self.trace.sampling_rate,
                baseline=baseline.value,
                amplitude=self.event_peak_values[ix] - baseline.value,
            )
            self.risetimes[ix] = risetime
            self.min_positions_rise[ix] = min_position_rise
            self.min_values_rise[ix] = min_value_rise

            self.max_positions_rise[ix] = max_position_rise
            self.max_values_rise[ix] = max_value_rise

            half_amplitude_level = (
                baseline.value + (data[event_peak_pos] - baseline.value) / 2
            )
            if (
                diffs[ix] < self.add_points
            ):  # next event close; check if we can get halfdecay
                right_lim = (
                    diffs[ix] + self.add_points
                )  # Right limit is the onset of the next event
                test_arr = data[event_peak_pos:right_lim]
                if test_arr[
                    test_arr < half_amplitude_level
                ].shape[
                    0
                ]:  # means that event goes below 50% ampliude before max rise of the next event; 1/2 decay can be calculated
                    halfdecay_position, halfdecay_time = get_event_halfdecay_time(
                        data=data[0:right_lim],
                        peak_position=event_peak_pos,
                        baseline=baseline.value,
                    )
                else:
                    halfdecay_position, halfdecay_time = np.nan, np.nan
            else:
                halfdecay_position, halfdecay_time = get_event_halfdecay_time(
                    data=data, peak_position=event_peak_pos, baseline=baseline.value
                )

            self.half_decay[ix] = halfdecay_position
            self.decaytimes[ix] = halfdecay_time

            # Calculate halfwidth and related times
            current_amplitude = abs(self.event_peak_values[ix] - baseline.value)
            if (
                self.event_direction == -1
            ):  # event_peak_values and bsls are already inverted for negative events later, so use original direction for amplitude calc with data
                current_amplitude = abs(data[event_peak_pos] - baseline.value)

            halfwidth, t_rise_half, t_decay_half = get_event_halfwidth(
                event_data=data,
                peak_index=event_peak_pos,
                baseline=baseline.value,
                amplitude=current_amplitude,
                sampling_rate=self.trace.sampling_rate,
            )
            self.halfwidths[ix] = halfwidth
            self.rise_half_amp_times[ix] = t_rise_half
            self.decay_half_amp_times[ix] = t_decay_half

            # calculate charges
            ### For charge; multiple event check done outside function.
            if ix < positions.shape[0] - 1:
                if (
                    combined_charge_events == 1
                ):  # define onset position for charge calculation
                    onset_in_trace = position - (self.add_points - self.event_start[ix])

                if np.isnan(self.half_decay[ix]):
                    combined_charge_events += 1

                else:
                    ### Get distance from peak to next event location.
                    peak_in_trace = position + (
                        self.event_peak_locations[ix] - self.add_points
                    )
                    next_event_location = positions[ix + 1]
                    delta_peak_location = next_event_location - peak_in_trace

                    # determine end of area calculation based on event decay
                    endpoint = int(
                        self.event_peak_locations[ix]
                        + charge_factor
                        * (int(self.half_decay[ix]) - self.event_peak_locations[ix])
                    )
                    delta_peak_endpoint = endpoint - self.event_peak_locations[ix]

                    if (
                        delta_peak_location > delta_peak_endpoint
                    ):  # Next event_location further away than the charge window; calculate charge
                        calculate_charge = True
                    else:
                        combined_charge_events += 1

                if calculate_charge:
                    endpoint_in_trace = (
                        position
                        + (self.event_peak_locations[ix] - self.add_points)
                        + delta_peak_endpoint
                    )
                    charge = get_event_charge(
                        data=mini_trace,
                        start_point=onset_in_trace,
                        end_point=endpoint_in_trace,
                        baseline=baseline.value,
                        sampling=self.trace.sampling,
                    )

            else:  # Handle the last event
                if (
                    combined_charge_events == 1
                ):  # define onset position for charge calculation
                    onset_in_trace = position - (self.add_points - self.event_start[ix])

                peak_in_trace = position + (
                    self.event_peak_locations[ix] - self.add_points
                )
                endpoint = int(
                    self.event_peak_locations[ix]
                    + charge_factor
                    * (int(self.half_decay[ix]) - self.event_peak_locations[ix])
                )
                delta_peak_endpoint = endpoint - self.event_peak_locations[ix]
                endpoint_in_trace = (
                    position
                    + (self.event_peak_locations[ix] - self.add_points)
                    + delta_peak_endpoint
                )

                endpoint_in_trace = min(endpoint_in_trace, mini_trace.shape[0])

                charge = get_event_charge(
                    data=mini_trace,
                    start_point=onset_in_trace,
                    end_point=endpoint_in_trace,
                    baseline=baseline.value,
                    sampling=self.trace.sampling,
                )
                calculate_charge = True
            if calculate_charge:  # Charge was calculated; check how many potentially overlapping events contributed.
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
            self.event_peak_locations[ix] = int(
                self.event_peak_locations[ix]
                + self.event_locations[ix]
                - self.add_points
            )
            self.bsl_starts[ix] = int(
                self.bsl_starts[ix] + self.event_locations[ix] - self.add_points
            )
            self.bsl_ends[ix] = int(
                self.bsl_ends[ix] + self.event_locations[ix] - self.add_points
            )

            self.event_start[ix] = int(
                self.event_start[ix] + self.event_locations[ix] - self.add_points
            )
            self.rise_half_amp_times[ix] += (
                self.event_locations[ix] - self.add_points
            ) * self.trace.sampling
            self.decay_half_amp_times[ix] += (
                self.event_locations[ix] - self.add_points
            ) * self.trace.sampling
            self.min_positions_rise[ix] += self.bsl_starts[ix] * self.trace.sampling
            self.max_positions_rise[ix] += self.bsl_starts[ix] * self.trace.sampling

            if not np.isnan(self.half_decay[ix]):
                self.half_decay[ix] = int(
                    self.half_decay[ix] + self.event_locations[ix] - self.add_points
                )

    def _get_singular_event_indices(self) -> None:
        """
        Identify events that do not overlap neighboring events.
        """
        no_events_in_decay = np.where(
            np.diff(self.event_locations) > self.window_size * 1.5
        )[0]
        no_events_in_rise = (
            (np.where(np.diff(self.event_locations) > self.window_size * 0.5)[0]) + 1
        )
        self.singular_event_indices = np.intersect1d(
            no_events_in_rise,
            no_events_in_decay,
            assume_unique=False,
            return_indices=False,
        )

        # First and last event will be lost due to intersecting. Add manually if they qualify.
        if 0 in no_events_in_decay:
            self.singular_event_indices = np.insert(self.singular_event_indices, 0, 0)
        if len(self.event_locations) - 1 in no_events_in_rise:
            self.singular_event_indices = np.append(
                self.singular_event_indices, [len(self.event_locations) - 1]
            )

        # if all events are overlapping, use all of them.
        if not len(self.singular_event_indices):
            self.singular_event_indices = np.array(
                list(range(self.event_locations.shape[0]))
            )

    def _get_average_event_properties(
        self, use_legacy_baseline_method: bool = True
    ) -> dict:
        """
        Analyze the average waveform of non-overlapping events.

        Parameters
        ----------
        use_legacy_baseline_method : bool, default=True
            If True, use the legacy baseline estimator.

        Returns
        -------
        dict
            Summary properties extracted from the average event waveform.
        """
        diffs = [self.add_points * 10]  # Set right limit larger than window size
        charge_factor = 4  # Charge window is 4 * decay time
        data = (
            np.mean(self.events[self.singular_event_indices], axis=0)
            * self.event_direction
        )

        event_peak = get_event_peak(
            data=data,
            event_num=0,
            add_points=self.add_points,
            window_size=self.window_size,
            diffs=diffs,
        )
        event_peak_value = data[event_peak]
        if use_legacy_baseline_method:
            baseline = legacy_get_event_baseline(
                data=data,
                duration=int(self.window_size * 0.1),
                event_num=0,
                diffs=diffs,
                add_points=self.add_points,
                peak_positions=[event_peak],
                positions=[self.add_points],
            )
        else:
            baseline = get_event_baseline(
                data=data,
                bsl_duration=int(self.window_size * 0.1),
                event_num=0,
                relative_event_position=self.add_points,
                positions=[self.add_points],
            )
        onset_position = get_event_onset(
            data=data,
            peak_position=event_peak,
            baseline=baseline.value,
            baseline_var=baseline.var,
        )

        (
            risetime,
            min_position_rise,
            min_value_rise,
            max_position_rise,
            max_value_rise,
        ) = get_event_risetime(
            data=data[baseline.start : int(event_peak)],
            sampling_rate=self.trace.sampling_rate,
            baseline=baseline.value,
            amplitude=event_peak_value - baseline.value,
        )

        halfdecay_position, halfdecay_time = get_event_halfdecay_time(
            data=data, peak_position=event_peak, baseline=baseline.value
        )
        endpoint = int(event_peak + charge_factor * halfdecay_position)
        charge = get_event_charge(
            data=data,
            start_point=onset_position,
            end_point=endpoint,
            baseline=baseline.value,
            sampling=self.trace.sampling,
        )

        results = {
            "amplitude": event_peak_value - baseline.value,
            "baseline": baseline.value * self.event_direction,
            "risetime": risetime * self.trace.sampling,
            "halfdecay_time": halfdecay_time * self.trace.sampling,
            "charge": charge * self.event_direction,
            "event_peak": event_peak,
            "onset_position": onset_position,
            "min_position_rise": min_position_rise,
            "min_value_rise": min_value_rise * self.event_direction,
            "max_position_rise": max_position_rise,
            "max_value_rise": max_value_rise * self.event_direction,
            "halfdecay_position": halfdecay_position,
            "endpoint_charge": endpoint,
        }

        return results

    def detect_events(
        self,
        stride: int | None = None,
        eval: bool = False,
        resample_to_600: bool = True,
        peak_w: int = 5,
        rel_prom_cutoff: float = 0.25,
        filter_factor: float = 20.0,
        convolve_win: int = 0,
        gradient_convolve_win: int = 0,
        bsl_win: float = 0.33,
        use_legacy_baseline_method: bool = True,
    ) -> None:
        """
        Perform event detection, extraction, and optional evaluation.

        Parameters
        ----------
        stride : int | None, optional
            Prediction stride. If omitted, defaults to roughly one thirtieth of
            the window size.
        eval : bool, default=False
            Whether to evaluate detected events.
        resample_to_600 : bool, default=True
            Whether to resample the trace to the 600-sample model window.
        peak_w : int, default=5
            The minimum prediction peak width.
        rel_prom_cutoff : float, default=0.25
            Relative prominence cutoff used when separating overlapping events.
        filter_factor : float, default=20.0
            Low-pass filter factor expressed as a fraction of the sampling rate.
        convolve_win : int, default=0
            Hann window size used for event-analysis filtering.
        gradient_convolve_win : int, default=0
            Hann window size used to smooth the derivative.
        bsl_win : float, default=0.33
            Baseline window size as fraction of window size.
        use_legacy_baseline_method : bool, default=True
            If True, use the legacy baseline estimator.

        Raises
        ------
        ValueError
            If ``bsl_win`` is not positive.
        """
        self.peak_w = peak_w
        self.rel_prom_cutoff = rel_prom_cutoff
        self.filter_factor = filter_factor
        self.convolve_win = convolve_win
        if bsl_win <= 0:
            raise ValueError("Baseline window size must be greater than 0")
        self.add_points = int(
            self.window_size / np.round(1 / bsl_win, 1)
        )  # number of additional points to extract before and after the event window for analysis

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
        self.start_pnts, self.end_pnts, scores = self._get_prediction_peaks(
            peak_w=peak_w
        )
        self.gradient, self.smth_gradient = self._make_smth_gradient()
        self.grad_threshold = self._get_grad_threshold(
            grad=self.smth_gradient, start_pnts=self.start_pnts, end_pnts=self.end_pnts
        )
        self.event_locations, self.event_scores = self._find_event_locations(
            limit=self.window_size + self.add_points,
            scores=scores,
            rel_prom_cutoff=rel_prom_cutoff,
        )
        self._remove_duplicate_locations()

        self.slopes = self.smth_gradient[self.event_locations]

        if self.event_locations.shape[0] > 0:
            self._get_singular_event_indices()
            self.events = self.trace._extract_event_data(
                positions=self.event_locations,
                before=self.add_points,
                after=self.window_size + self.add_points,
            )

            self._get_event_properties(
                use_legacy_baseline_method=use_legacy_baseline_method
            )
            self.events = self.events - self.event_bsls[:, None]
            self.average_event_properties = self._get_average_event_properties(
                use_legacy_baseline_method=use_legacy_baseline_method
            )

            if eval:
                self._eval_events()

    def _get_average_event_decay(self) -> np.ndarray:
        """
        Fit a single-exponential decay to the average event waveform.

        Returns
        -------
        np.ndarray
            Fit parameters for the average event decay, or ``np.nan`` values when
            the fit fails.
        """
        events_for_avg = self.events[self.singular_event_indices]

        event_x = np.arange(0, events_for_avg.shape[1]) * self.trace.sampling
        event_avg = np.average(events_for_avg, axis=0) * self.event_direction
        if events_for_avg.shape[0] < 4:
            fit_start = np.argmax(
                np.convolve(event_avg, np.ones(5) / 5, mode="same")
            ) + int(0.01 * self.window_size)
        else:
            fit_start = np.argmax(event_avg) + int(0.01 * self.window_size)
        if fit_start > events_for_avg.shape[1] - int(
            0.2 * self.window_size
        ):  # not a valid starting point
            return np.full(3, np.nan)
        try:
            self.avg_decay_fit_start = fit_start
            fit, _ = curve_fit(
                exp_fit,
                event_x[fit_start:],
                event_avg[fit_start:],
                p0=[
                    np.amax(event_avg) + 1,
                    events_for_avg.shape[1] / 50 * self.trace.sampling,
                    0,
                ],
                bounds=([0, 0, -np.inf], [np.inf, 1e3, np.inf]),
            )
            return fit
        except RuntimeError:
            self.avg_decay_fit_start = np.nan
            return np.full(3, np.nan)

    def _eval_events(self) -> None:
        """
        Compute event statistics and derived timing metrics.

        Notes
        -----
        Results are stored on the instance, including ``event_stats``, event peak
        times, onset times, half-decay times, and inter-event intervals.
        """
        if not self.events_present():
            return
        if not len(self.singular_event_indices):
            self.singular_event_indices = np.array(
                list(range(self.event_locations.shape[0]))
            )

        self.avg_decay_fit = self._get_average_event_decay()
        self.event_stats = EventStats(
            amplitudes=self.event_peak_values - self.event_bsls,
            scores=self.event_scores,
            tau=self.avg_decay_fit[1],
            charges=self.charges,
            risetimes=self.risetimes,
            slopes=self.slopes,
            decaytimes=self.decaytimes,
            halfwidths=self.halfwidths,
            time=self.trace.total_time,
            unit=self.trace.y_unit,
        )

        self.event_peak_times = self.event_peak_locations * self.trace.sampling
        self.half_decay_times = self.half_decay * self.trace.sampling
        self.event_start_times = self.event_start * self.trace.sampling
        self.interevent_intervals = np.diff(self.event_peak_times, prepend=np.nan)

        if self.verbose:
            self.event_stats.print()

    def delete_events(
        self, event_indices: list[int] | None = None, eval: bool = True
    ) -> None:
        """
        Delete events by index and optionally recompute statistics.

        Parameters
        ----------
        event_indices : list[int] | None, optional
            Event indices to remove.
        eval : bool, default=True
            Whether to recompute event statistics after deletion.

        Raises
        ------
        ValueError
            If any requested event index does not exist.
        """
        event_indices = event_indices or []
        if not self.events_present():
            return

        if not hasattr(self, "event_stats"):
            self._eval_events()

        for event in event_indices:
            if event < 0 or event >= self.event_locations.shape[0]:
                raise ValueError(f"Event {event} does not exist.")

        num_events = self.event_locations.shape[0]
        blacklist = {}

        attrs_to_delete = []
        for attr_name, attr_val in self.__dict__.items():
            if attr_name in blacklist:
                continue
            if isinstance(attr_val, np.ndarray) and attr_val.shape[0] == num_events:
                attrs_to_delete.append(attr_name)

        for attr_name in attrs_to_delete:
            arr = getattr(self, attr_name)
            setattr(self, attr_name, np.delete(arr, event_indices, axis=0))

        self.deleted_events += len(event_indices)

        if eval:
            self._get_singular_event_indices()
            self._eval_events()

    def save_to_h5(self, filename: str, include_prediction: bool = False) -> None:
        """
        Save detection results to an HDF5 file.

        Parameters
        ----------
        filename : str
            Destination filename. The ``.h5`` suffix is added if needed.
        include_prediction : bool, default=False
            If True, include the prediction trace in the output file.
        """
        if not hasattr(self, "event_stats"):
            self._eval_events()
            if not hasattr(self, "event_stats"):
                print("Save error: No events found")
                return

        if not filename.endswith("h5"):
            filename += ".h5"

        with h5py.File(filename, "w", track_order=True) as f:
            f.create_dataset("events", data=np.array(self.events))
            f.create_dataset(
                "event_params/event_locations", data=np.array(self.event_locations)
            )
            f.create_dataset(
                "event_params/event_scores", data=np.array(self.event_scores)
            )
            f.create_dataset(
                "event_params/event_amplitudes", data=self.event_stats.amplitudes
            )
            f.create_dataset(
                "event_params/event_charges", data=self.event_stats.charges
            )
            f.create_dataset(
                "event_params/event_risetimes", data=self.event_stats.risetimes
            )
            f.create_dataset(
                "event_params/event_halfdecays", data=self.event_stats.halfdecays
            )
            f.create_dataset(
                "event_params/event_halfwidths", data=self.event_stats.halfwidths
            )
            f.create_dataset("event_params/event_bsls", data=np.array(self.event_bsls))
            f.create_dataset(
                "event_params/event_intervals", data=np.array(self.interevent_intervals)
            )
            f.create_dataset(
                "event_statistics/amplitude_average",
                data=self.event_stats.mean(self.event_stats.amplitudes),
            )
            f.create_dataset(
                "event_statistics/amplitude_stdev",
                data=self.event_stats.std(self.event_stats.amplitudes),
            )
            f.create_dataset(
                "event_statistics/amplitude_median",
                data=self.event_stats.median(self.event_stats.amplitudes),
            )
            f.create_dataset(
                "event_statistics/charge_mean",
                data=self.event_stats.mean(self.event_stats.charges),
            )
            f.create_dataset(
                "event_statistics/charge_median",
                data=self.event_stats.median(self.event_stats.charges),
            )
            f.create_dataset(
                "event_statistics/risetime_mean",
                data=self.event_stats.mean(self.event_stats.risetimes),
            )
            f.create_dataset(
                "event_statistics/risetime_median",
                data=self.event_stats.median(self.event_stats.risetimes),
            )
            f.create_dataset(
                "event_statistics/decaytime_mean",
                data=self.event_stats.mean(self.event_stats.halfdecays),
            )
            f.create_dataset(
                "event_statistics/decaytime_median",
                data=self.event_stats.median(self.event_stats.halfdecays),
            )
            f.create_dataset(
                "event_statistics/halfwidth_mean",
                data=self.event_stats.mean(self.event_stats.halfwidths),
            )
            f.create_dataset(
                "event_statistics/halfwidth_median",
                data=self.event_stats.median(self.event_stats.halfwidths),
            )
            f.create_dataset(
                "event_statistics/decay_from_fit", data=self.event_stats.avg_tau_decay
            )
            f.create_dataset(
                "event_statistics/frequency", data=self.event_stats.frequency()
            )
            f.create_dataset(
                "event_statistics/iei_mean",
                data=self.event_stats.mean(self.interevent_intervals),
            )
            f.create_dataset(
                "event_statistics/iei_median",
                data=self.event_stats.median(self.interevent_intervals),
            )

            f.attrs["amplitude_unit"] = self.trace.y_unit
            f.attrs["recording_time"] = self.trace.data.shape[0] * self.trace.sampling
            f.attrs["source_filename"] = self.trace.filename
            f.attrs["miniml_model"] = self.model_path
            f.attrs["miniml_model_threshold"] = self.model_threshold
            f.attrs["minimum_peak"] = self.peak_w
            f.attrs["stride"] = self.stride_length
            f.attrs["window"] = self.window_size
            f.attrs["event_direction"] = self.event_direction
            if self.convolve_win > 0:
                f.attrs["convolve_win"] = self.convolve_win
            else:
                f.attrs["filter_factor"] = self.filter_factor
            f.attrs["gradient_convolve_win"] = self.gradient_convolve_win
            f.attrs["relative_prominence"] = self.rel_prom_cutoff
            f.attrs["deleted_events"] = self.deleted_events

            if include_prediction:
                f.create_dataset("prediction", data=self.prediction)
        print(f"Events saved to {filename}")

    def save_to_csv(self, filename: str = "") -> None:
        """
        Save detection results to CSV files.

        Parameters
        ----------
        filename : str, default=""
            Output filename stem. Results are written to ``*_avgs.csv`` and
            ``*_individual.csv`` files.
        """
        if filename.endswith(".csv"):
            filename = filename.removesuffix(".csv")

        if not hasattr(self, "event_stats"):
            self._eval_events()
            if not hasattr(self, "event_stats"):
                print("Save error: No events found")
                return

        individual = np.stack(
            (
                np.array(self.event_locations),
                np.array(self.event_scores),
                self.event_stats.amplitudes,
                self.event_stats.charges,
                self.event_stats.risetimes,
                self.event_stats.halfdecays,
                self.event_stats.halfwidths,
                self.interevent_intervals,
            )
        )

        avgs = np.array(
            (
                self.event_stats.mean(self.event_stats.amplitudes),
                self.event_stats.std(self.event_stats.amplitudes),
                self.event_stats.median(self.event_stats.amplitudes),
                self.event_stats.mean(self.event_stats.charges),
                self.event_stats.mean(self.event_stats.risetimes),
                self.event_stats.mean(self.event_stats.halfdecays),
                self.event_stats.mean(self.event_stats.halfwidths),
                self.event_stats.avg_tau_decay,
                self.event_stats.frequency(),
                self.event_stats.mean(self.interevent_intervals),
            )
        )

        column_names = [f"event_{i}" for i in range(len(self.event_locations))]

        import csv

        row_labels_ind = [
            "location",
            "score",
            "amplitude",
            "charge",
            "risetime",
            "decaytime",
            "halfwidth",
            "interval",
        ]
        with open(f"{filename}_individual.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([""] + column_names)
            for label, row_data in zip(row_labels_ind, individual):
                writer.writerow([label] + list(row_data))

        row_labels_avg = [
            "amplitude mean",
            "amplitude std",
            "amplitude median",
            "charge mean",
            "risetime mean",
            "decaytime mean",
            "halfwidth mean",
            "tau_avg",
            "frequency",
            "iei mean",
        ]
        with open(f"{filename}_avgs.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for label, val in zip(row_labels_avg, avgs):
                writer.writerow([label, val])

        print(f"events saved to {filename}_avgs.csv and {filename}_individual.csv")

    def save_to_pickle(
        self,
        filename: str = "",
        include_prediction: bool = True,
        include_data: bool = True,
    ) -> None:
        """
        Save detection results to a pickle file.

        Parameters
        ----------
        filename : str, default=""
            Output filename, optionally including a directory.
        include_prediction : bool, default=True
            Include the prediction trace.
        include_data : bool, default=True
            Include the source trace data together with the analysis results.
        """
        if not hasattr(self, "event_stats"):
            self._eval_events()
            if not hasattr(self, "event_stats"):
                print("Save error: No events found")
                return

        if not filename.endswith("pickle"):
            filename += ".pickle"

        results = {
            "event_location_parameters": {
                "event_locations": np.array(self.event_locations),
                "event_scores": np.array(self.event_scores),
                "event_peak_locations": self.event_peak_locations,
                "event_baselines": self.event_bsls,
                "event_onset_locations": self.event_start,
                "min_positions_rise": self.min_positions_rise,
                "max_positions_rise": self.max_positions_rise,
                "min_values_rise": self.min_values_rise,
                "max_values_rise": self.max_values_rise,
                "half_decay_positions": self.half_decay,
            },
            "individual_values": {
                "amplitudes": self.event_stats.amplitudes,
                "charges": self.event_stats.charges,
                "risetimes": self.event_stats.risetimes,
                "half_decaytimes": self.event_stats.halfdecays,
                "event_intervals": self.interevent_intervals,
                "halfwidths": self.event_stats.halfwidths,
            },
            "average_values": {
                "amplitude mean": self.event_stats.mean(self.event_stats.amplitudes),
                "amplitude std": self.event_stats.std(self.event_stats.amplitudes),
                "amplitude median": self.event_stats.median(
                    self.event_stats.amplitudes
                ),
                "charge mean": self.event_stats.mean(self.event_stats.charges),
                "risetime mean": self.event_stats.mean(self.event_stats.risetimes),
                "half_decaytime mean": self.event_stats.mean(
                    self.event_stats.halfdecays
                ),
                "halfwidth mean": self.event_stats.mean(self.event_stats.halfwidths),
                "decay_tau": self.event_stats.avg_tau_decay * 1000,
                "frequency": self.event_stats.frequency(),
                "iei_mean": self.event_stats.mean(self.interevent_intervals),
                "iei_median": self.event_stats.median(self.interevent_intervals),
            },
            "average_event_properties": self.average_event_properties,
            "metadata": {
                ### trace information:
                "source_filename": self.trace.filename,
                "y_unit": self.trace.y_unit,
                "recording_time": self.trace.data.shape[0] * self.trace.sampling,
                "sampling": self.trace.sampling,
                "sampling_rate": self.trace.sampling_rate,
                ### miniML information
                "miniml_model": self.model_path,
                "miniml_model_threshold": self.model_threshold,
                ### event detection params:
                "window_size": self.window_size,
                "stride": self.stride_length,
                "add_points": self.add_points,
                "resampling_factor": self.resampling_factor,
                ### event analysis params:
                "convolve_win": self.convolve_win,
                "filter_factor": self.filter_factor,
                "gradient_convolve_win": self.gradient_convolve_win,
                "min_peak_w": self.peak_w,
                "rel_prom_cutoff": self.rel_prom_cutoff,
                "event_direction": self.event_direction,
                "deleted_events": self.deleted_events,
            },
            "events": self.events,
        }

        if include_prediction:
            results["prediction"] = self.prediction  # Save prediction as numpy array

        if include_data:
            results["mini_trace"] = self.trace.data

        with open(filename, "wb") as handle:
            pkl.dump(results, handle)
        print(f"events saved to {filename}")


class EventAnalysis(EventDetection):
    """
    Analyze events detected by an external method.

    Parameters
    ----------
    trace : MiniTrace
        Raw input trace.
    window_size : int
        Number of samples to extract for each individual event.
    event_direction : str
        Event polarity.
    verbose : int
        Verbosity level.
    event_positions : np.ndarray | list
        Positions of detected events.
    filter_factor : int
        Fraction of the sampling rate used to lowpass filter the data for analysis.
    convolve_win : int
        Hann window size used to filter the data for event analysis.
    gradient_convolve_win : int
        Hann window size used to smooth the derivative.
    resampling_factor : float
        Resampling factor applied during analysis.
    """

    def __init__(
        self,
        trace,
        window_size,
        event_direction,
        verbose,
        event_positions,
        filter_factor,
        convolve_win,
        gradient_convolve_win,
        resampling_factor,
    ) -> None:
        super().__init__(
            data=trace,
            window_size=window_size,
            event_direction=event_direction,
            verbose=verbose,
        )
        self.add_points = int(self.window_size / 3)
        self.event_direction = event_direction
        self.filter_factor = filter_factor
        self.convolve_win = convolve_win
        self.resampling_factor = resampling_factor

        self.event_locations = event_positions[
            np.logical_and(
                self.add_points < event_positions,
                event_positions
                < len(self.trace.data) - (self.window_size + self.add_points),
            )
        ]
        self.event_locations = self.event_locations.astype(np.int64)
        self.events = self.trace._extract_event_data(
            self.event_locations,
            before=self.add_points,
            after=self.window_size + self.add_points,
        )
        self.gradient_convolve_win = gradient_convolve_win

    def eval_events(self, filter: bool = True) -> None:
        """
        Evaluate externally detected events.

        Parameters
        ----------
        filter : bool, default=True
            If True, derive event properties from filtered trace data.
        """
        if self.event_locations.shape[0] > 0:
            super()._get_event_properties(filter=filter)
            self.events = self.events - self.event_bsls[:, None]
            super()._eval_events()
