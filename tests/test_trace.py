import unittest
from unittest.mock import patch
import numpy as np
from miniml.core.trace import MiniTrace


class TestMiniTrace(unittest.TestCase):
    def setUp(self):
        # Create a simple trace: 1000 Hz, 1 second of data (1000 samples)
        self.sampling_rate = 1000.0
        self.sampling_interval = 1.0 / self.sampling_rate
        self.data = np.sin(2 * np.pi * 5 * np.linspace(0, 1, 1000))
        self.trace = MiniTrace(
            data=self.data,
            sampling_interval=self.sampling_interval,
            y_unit="pA",
            filename="test_trace.abf",
        )

    def test_initialization(self):
        """Test default initialization and parameter setting."""
        empty_trace = MiniTrace()
        self.assertEqual(len(empty_trace.data), 0)
        self.assertEqual(empty_trace.sampling, 1.0)
        self.assertEqual(empty_trace.y_unit, "")
        self.assertEqual(empty_trace.filename, "")

        self.assertTrue(np.array_equal(self.trace.data, self.data))
        self.assertEqual(self.trace.sampling, self.sampling_interval)
        self.assertEqual(self.trace.y_unit, "pA")
        self.assertEqual(self.trace.filename, "test_trace.abf")
        self.assertEqual(self.trace.data.dtype, np.float64)

    def test_sampling_validation(self):
        """Test validation for negative sampling interval."""
        with self.assertRaises(ValueError):
            self.trace.sampling = -0.5

    def test_properties(self):
        """Test dynamic properties of MiniTrace."""
        self.assertEqual(self.trace.sampling_rate, self.sampling_rate)

        # total_time
        self.assertAlmostEqual(self.trace.total_time, 1.0)

        # time_axis
        time_axis = self.trace.time_axis
        self.assertEqual(len(time_axis), len(self.data))
        self.assertAlmostEqual(time_axis[0], 0.0)
        self.assertAlmostEqual(time_axis[-1], 1.0 - self.sampling_interval)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.figure")
    def test_plot_trace(self, mock_figure, mock_show):
        """Test plot_trace function executes without crashing."""
        self.trace.plot_trace()
        mock_figure.assert_called()
        mock_show.assert_called_once()

    def test_detrend_linear(self):
        """Test detrending of trace."""
        # Add a linear trend to the data
        trend = np.linspace(0, 10, 1000)
        noisy_data = self.data + trend
        noisy_trace = MiniTrace(noisy_data, self.sampling_interval)

        detrended_trace = noisy_trace.detrend(detrend_type="linear")
        self.assertEqual(detrended_trace.data.shape, self.data.shape)
        # Detrended data should be close to original self.data (trend removed)
        np.testing.assert_allclose(detrended_trace.data, self.data, atol=0.5)

    def test_detrend_segments(self):
        """Test detrending with multiple segments."""
        noisy_trace = MiniTrace(self.data, self.sampling_interval)
        detrended_trace = noisy_trace.detrend(detrend_type="linear", num_segments=5)
        self.assertEqual(detrended_trace.data.shape, self.data.shape)

    def test_filter_highpass_lowpass(self):
        """Test highpass and lowpass Butter filtering."""
        highpass_filtered = self.trace.filter(highpass=10.0)
        self.assertEqual(highpass_filtered.data.shape, self.data.shape)

        lowpass_filtered = self.trace.filter(lowpass=50.0)
        self.assertEqual(lowpass_filtered.data.shape, self.data.shape)

    def test_filter_savgol_and_hann(self):
        """Test Savitzky-Golay and Hann window filters."""
        # Savgol filtering
        savgol_filtered = self.trace.filter(savgol=10.0) # 10 ms
        self.assertEqual(savgol_filtered.data.shape, self.data.shape)

        # Hann window
        hann_filtered = self.trace.filter(hann=10)
        self.assertEqual(hann_filtered.data.shape, self.data.shape)
        # Check edge preservation
        np.testing.assert_array_equal(hann_filtered.data[:10], self.trace.data[:10])
        np.testing.assert_array_equal(hann_filtered.data[-10:], self.trace.data[-10:])

    def test_filter_multiple_lowpass_warning(self):
        """Test warning output when both lowpass and savgol are specified."""
        with patch("builtins.print") as mock_print:
            filtered = self.trace.filter(lowpass=50.0, savgol=10.0)
            mock_print.assert_any_call(
                "Warning: Two lowpass filters selected, Savgol filter is ignored."
            )
            self.assertEqual(filtered.data.shape, self.data.shape)

    def test_filter_line_noise(self):
        """Test line-noise frequency filtering."""
        # Must raise ValueError if width is not specified
        with self.assertRaises(ValueError):
            self.trace.filter(line_freq=50.0)

        # Successful line-noise filtering
        filtered = self.trace.filter(line_freq=50.0, width=2.0)
        self.assertEqual(filtered.data.shape, self.data.shape)

    def test_resample(self):
        """Test resampling to a target frequency."""
        # None frequency returns same object
        self.assertIs(self.trace.resample(None), self.trace)

        # Resample from 1000 Hz to 500 Hz
        resampled = self.trace.resample(500.0)
        self.assertEqual(len(resampled.data), 500)
        self.assertAlmostEqual(resampled.sampling, 0.002)

    def test_extract_event_data_valid(self):
        """Test extracting valid event windows from the trace."""
        positions = np.array([100, 200, 300])
        before = 10
        after = 20
        extracted = self.trace._extract_event_data(positions, before, after)

        # Expected shape: (len(positions), before + after) = (3, 30)
        self.assertEqual(extracted.shape, (3, 30))
        # Verify first extracted window
        expected_window = self.data[100 - 10 : 100 + 20]
        np.testing.assert_array_equal(extracted[0], expected_window)

    def test_extract_event_data_out_of_bounds(self):
        """Test extract_event_data raises ValueError when out of bounds."""
        # Exceeds left bound
        positions_left = np.array([5, 200])
        with self.assertRaises(ValueError):
            self.trace._extract_event_data(positions_left, before=10, after=20)

        # Exceeds right bound
        positions_right = np.array([200, 990])
        with self.assertRaises(ValueError):
            self.trace._extract_event_data(positions_right, before=10, after=20)


if __name__ == "__main__":
    unittest.main()
