import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import tempfile
import os
import h5py
import pickle as pkl
import keras

from miniml.core.trace import MiniTrace
from miniml.core.event import EventStats, EventDetection, EventAnalysis


class TestEventStats(unittest.TestCase):
    def test_stats_calculations(self):
        """Test statistical properties of EventStats with typical values."""
        amplitudes = np.array([10.0, 12.0, 14.0])
        scores = np.array([0.9, 0.95, 0.88])
        charges = np.array([2.0, 2.5, 3.0])
        risetimes = np.array([0.001, 0.0015, 0.002])
        slopes = np.array([5.0, 6.0, 7.0])
        decaytimes = np.array([0.003, 0.004, 0.005])
        halfwidths = np.array([0.002, 0.0025, 0.003])
        tau = 0.004
        time = 10.0
        unit = "pA"

        stats = EventStats(
            amplitudes=amplitudes,
            scores=scores,
            charges=charges,
            risetimes=risetimes,
            slopes=slopes,
            decaytimes=decaytimes,
            halfwidths=halfwidths,
            tau=tau,
            time=time,
            unit=unit,
        )

        self.assertEqual(stats.event_count, 3)
        self.assertAlmostEqual(stats.mean(stats.amplitudes), 12.0)
        self.assertAlmostEqual(stats.median(stats.amplitudes), 12.0)
        self.assertAlmostEqual(stats.std(stats.amplitudes), np.std(amplitudes, ddof=1))
        self.assertAlmostEqual(stats.cv(stats.amplitudes), np.std(amplitudes, ddof=1) / 12.0)
        self.assertAlmostEqual(stats.frequency(), 0.3)

    def test_stats_with_nan(self):
        """Test EventStats methods when NaN or empty arrays are passed."""
        stats = EventStats(
            amplitudes=np.array([np.nan, np.nan]),
            scores=np.array([0.9, np.nan]),
            charges=np.array([]),
            risetimes=np.array([]),
            slopes=np.array([]),
            decaytimes=np.array([]),
            halfwidths=np.array([]),
            tau=np.nan,
            time=10.0,
            unit="pA",
        )
        self.assertTrue(np.isnan(stats.mean(stats.amplitudes)))
        self.assertTrue(np.isnan(stats.median(stats.amplitudes)))
        self.assertTrue(np.isnan(stats.std(stats.amplitudes)))
        self.assertTrue(np.isnan(stats.cv(stats.amplitudes)))

        # Mean when mean is 0 should return NaN for cv
        stats_zero = EventStats(
            amplitudes=np.array([0.0, 0.0]),
            scores=np.array([0.9, np.nan]),
            charges=np.array([]),
            risetimes=np.array([]),
            slopes=np.array([]),
            decaytimes=np.array([]),
            halfwidths=np.array([]),
            tau=np.nan,
            time=10.0,
            unit="pA",
        )
        self.assertTrue(np.isnan(stats_zero.cv(stats_zero.amplitudes)))

    def test_print_stats(self):
        """Test the print() output executes without error."""
        stats = EventStats(
            amplitudes=np.array([10.0]),
            scores=np.array([0.9]),
            charges=np.array([2.0]),
            risetimes=np.array([0.001]),
            slopes=np.array([5.0]),
            decaytimes=np.array([0.003]),
            halfwidths=np.array([0.002]),
            tau=0.004,
            time=10.0,
            unit="pA",
        )
        with patch("builtins.print") as mock_print:
            stats.print()
            mock_print.assert_called()


class TestEventDetection(unittest.TestCase):
    def setUp(self):
        # Generate dummy trace data (with a synthetic event/peak in the middle)
        self.sampling_rate = 10000.0
        self.sampling_interval = 1.0 / self.sampling_rate
        self.trace_length = 5000
        self.data = np.zeros(self.trace_length, dtype=float)

        # Add a simulated negative event at index 2000
        # Fast rise, slow decay
        x = np.arange(1000)
        event_shape = -20.0 * (1 - np.exp(-x / 10.0)) * np.exp(-x / 50.0)
        self.data[2000:3000] += event_shape

        self.trace = MiniTrace(
            data=self.data,
            sampling_interval=self.sampling_interval,
            y_unit="pA",
            filename="mock_trace.abf",
        )
        # Instantiate EventDetection with no model initially
        self.detector = EventDetection(
            data=self.trace,
            window_size=600,
            event_direction="negative",
            training_direction="negative",
            verbose=0,
        )

    def test_direction_setters(self):
        """Test setting event and training directions."""
        self.detector.event_direction = "negative"
        self.assertEqual(self.detector.event_direction, -1)
        self.detector.event_direction = "positive"
        self.assertEqual(self.detector.event_direction, 1)

        self.detector.training_direction = "negative"
        self.assertEqual(self.detector.training_direction, -1)
        self.detector.training_direction = "positive"
        self.assertEqual(self.detector.training_direction, 1)

    def test_lowpass_filter(self):
        """Test Butterworth lowpass filter helper."""
        filtered = self.detector.lowpass_filter(self.data, cutoff=1000.0)
        self.assertEqual(filtered.shape, self.data.shape)

    def test_hann_filter(self):
        """Test Hann window smoothing helper."""
        smoothed = self.detector.hann_filter(self.data, filter_size=10)
        self.assertEqual(smoothed.shape, self.data.shape)
        # filter_size = 0 returns input unchanged
        self.assertTrue(np.array_equal(self.detector.hann_filter(self.data, 0), self.data))

    def test_linear_interpolation(self):
        """Test linear interpolation helper."""
        arr = np.array([0.0, 10.0, 20.0])
        interpolated, factor = self.detector._linear_interpolation(arr, 5)
        self.assertEqual(len(interpolated), 5)
        self.assertAlmostEqual(factor, 5.0 / 3.0)

    def test_bsl_win_validation(self):
        """Test detect_events raises ValueError for invalid bsl_win."""
        with self.assertRaises(ValueError):
            self.detector.detect_events(bsl_win=-0.5)

    def test_load_model_error(self):
        """Test load_model raises ValueError when file is not a valid Keras model."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name
        try:
            # Create a valid HDF5 file but without the keras_version attribute
            with h5py.File(temp_path, "w") as f_h5:
                f_h5.create_dataset("dummy", data=[1, 2, 3])

            with self.assertRaises(ValueError):
                self.detector.load_model(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_full_detection_flow(self):
        """Test the event detection pipeline by mocking prediction and thresholding."""
        # We will mock the prediction step: pretend the model predicts a peak near index 2000.
        # Window size is 600, resampling to 600 is True (resampling_factor=1).
        # stride_length = 20. Total predictions = (5000 - 600) / 20 = 220.
        # Let's create a simulated prediction trace.
        mock_pred = np.zeros(220, dtype=float)
        # Put a high probability peak at index 95 (corresponding to 95 * 20 = 1900 to 2000 in raw coordinates)
        mock_pred[90:100] = np.linspace(0.1, 0.95, 10)
        mock_pred[100:110] = np.linspace(0.95, 0.1, 10)

        # Mock the __predict method to assign this prediction to detector
        def mock_predict_fn(self_detector):
            self_detector.prediction = mock_pred

        with patch.object(EventDetection, "_EventDetection__predict", mock_predict_fn):
            # Run detection
            self.detector.detect_events(
                stride=20,
                eval=True,
                resample_to_600=True,
                peak_w=5,
                use_legacy_baseline_method=True,
            )

        # Verify that we found some event locations and events are present
        self.assertTrue(self.detector.events_present())
        self.assertGreater(len(self.detector.event_locations), 0)

        # Verify event stats are evaluated
        self.assertIsNotNone(self.detector.event_stats)
        self.assertEqual(len(self.detector.event_stats.amplitudes), len(self.detector.event_locations))

        # Test event deletion
        original_count = len(self.detector.event_locations)

        # Try to delete non-existent index (expect ValueError)
        with self.assertRaises(ValueError):
            self.detector.delete_events([999999])

        # Delete the first event
        self.detector.delete_events([0], eval=True)
        self.assertEqual(len(self.detector.event_locations), original_count - 1)
        self.assertEqual(self.detector.deleted_events, 1)

    def test_full_detection_non_legacy_baseline(self):
        """Test event detection pipeline with the non-legacy baseline method."""
        mock_pred = np.zeros(220, dtype=float)
        mock_pred[100] = 0.99

        def mock_predict_fn(self_detector):
            self_detector.prediction = mock_pred

        with patch.object(EventDetection, "_EventDetection__predict", mock_predict_fn):
            self.detector.detect_events(
                stride=20,
                eval=True,
                resample_to_600=True,
                peak_w=5,
                use_legacy_baseline_method=False,
            )
        self.assertTrue(self.detector.events_present())

    def test_serialization_methods(self):
        """Test saving detection results to H5, CSV, and Pickle files."""
        # Create a mock detection with some populated attributes
        self.detector.events = np.zeros((2, 1000))
        self.detector.event_locations = np.array([2000, 3000], dtype=np.int64)
        self.detector.event_scores = np.array([0.9, 0.95])
        self.detector.event_peak_locations = np.array([2050, 3050], dtype=np.int64)
        self.detector.event_peak_times = np.array([0.205, 0.305])
        self.detector.half_decay = np.array([2100, 3100])
        self.detector.half_decay_times = np.array([0.21, 0.31])
        self.detector.event_start = np.array([2000, 3000])
        self.detector.event_start_times = np.array([0.2, 0.3])
        self.detector.interevent_intervals = np.array([np.nan, 0.1])
        self.detector.event_bsls = np.array([0.0, 0.0])
        self.detector.prediction = np.zeros(220)
        self.detector.stride_length = 20
        self.detector.peak_w = 5
        self.detector.add_points = 200
        self.detector.resampling_factor = 1.0
        self.detector.filter_factor = 20.0
        self.detector.gradient_convolve_win = 0
        self.detector.rel_prom_cutoff = 0.25
        self.detector.avg_decay_fit = np.array([10.0, 0.004, 0.0])
        self.detector.average_event_properties = {"amplitude": 10.0}

        # Setup event stats
        self.detector.event_stats = EventStats(
            amplitudes=np.array([10.0, 12.0]),
            scores=self.detector.event_scores,
            charges=np.array([2.0, 2.5]),
            risetimes=np.array([0.001, 0.0012]),
            slopes=np.array([5.0, 6.0]),
            decaytimes=np.array([0.003, 0.0035]),
            halfwidths=np.array([0.002, 0.0022]),
            tau=0.004,
            time=self.trace.total_time,
            unit="pA",
        )
        self.detector.singular_event_indices = np.array([0, 1])

        # HDF5
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            h5_path = f.name
        try:
            self.detector.save_to_h5(h5_path, include_prediction=True)
            self.assertTrue(os.path.exists(h5_path))
            with h5py.File(h5_path, "r") as f_h5:
                self.assertIn("events", f_h5)
                self.assertIn("prediction", f_h5)
                self.assertIn("event_params/event_locations", f_h5)
                self.assertAlmostEqual(f_h5.attrs["recording_time"], self.trace.total_time)
        finally:
            if os.path.exists(h5_path):
                os.remove(h5_path)

        # CSV
        with tempfile.TemporaryDirectory() as tempdir:
            csv_stem = os.path.join(tempdir, "test_csv")
            self.detector.save_to_csv(csv_stem)
            self.assertTrue(os.path.exists(f"{csv_stem}_individual.csv"))
            self.assertTrue(os.path.exists(f"{csv_stem}_avgs.csv"))

        # Pickle
        with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as f:
            pickle_path = f.name
        try:
            self.detector.save_to_pickle(pickle_path, include_prediction=True, include_data=True)
            self.assertTrue(os.path.exists(pickle_path))
            with open(pickle_path, "rb") as f_pkl:
                res = pkl.load(f_pkl)
                self.assertIn("metadata", res)
                self.assertIn("prediction", res)
                self.assertIn("mini_trace", res)
        finally:
            if os.path.exists(pickle_path):
                os.remove(pickle_path)


class TestEventAnalysis(unittest.TestCase):
    def test_analysis_instantiation_and_eval(self):
        """Test EventAnalysis runs end-to-end evaluation with externally-supplied indices."""
        sampling_rate = 10000.0
        sampling_interval = 1.0 / sampling_rate
        data = np.zeros(4000)
        # Synthetic event
        data[2000:2500] = -15.0 * np.exp(-np.arange(500) / 100.0)

        trace = MiniTrace(data, sampling_interval, y_unit="pA")
        event_positions = np.array([2000])

        analysis = EventAnalysis(
            trace=trace,
            window_size=600,
            event_direction="negative",
            verbose=0,
            event_positions=event_positions,
            filter_factor=20.0,
            convolve_win=0,
            gradient_convolve_win=0,
            resampling_factor=1.0,
        )

        self.assertEqual(len(analysis.event_locations), 1)
        self.assertEqual(analysis.event_locations[0], 2000)

        # Run eval_events
        analysis.eval_events(filter=True)
        self.assertIsNotNone(analysis.event_stats)
        self.assertEqual(analysis.event_stats.event_count, 1)


if __name__ == "__main__":
    unittest.main()
