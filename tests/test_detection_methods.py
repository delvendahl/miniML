import unittest
import numpy as np

from miniml.threshold_detection import threshold_detection, DetectionResult
from miniml.deconvolution import deconvolution, make_template as deconv_make_template, DeconvolutionResult
from miniml.template_matching import template_matching, make_template as tm_make_template, TemplateMatchResult


class TestDetectionMethods(unittest.TestCase):
    def setUp(self):
        # Sampling rate of 20,000 Hz, 1000 samples (0.05 seconds of data)
        self.sampling = 5e-5 # 2e-5 is 50 kHz, 5e-5 is 20 kHz
        self.data = np.zeros(2000, dtype=float)

        # Add a simulated negative event at index 1000
        x = np.arange(500)
        event = -10.0 * (1 - np.exp(-x / 10.0)) * np.exp(-x / 50.0)
        self.data[1000:1500] += event

    def test_threshold_detection(self):
        """Test threshold-based event detection."""
        # Parameters: threshold, baseline duration, dt, peak_win
        result = threshold_detection(
            data=self.data,
            sampling=self.sampling,
            threshold=-2.0,
            baseline=0.005,
            dt=0.001,
            peak_win=0.002,
        )
        self.assertIsInstance(result, DetectionResult)
        self.assertGreater(len(result.indices), 0)
        # Check repr
        self.assertIn("DetectionResult", repr(result))

    def test_deconvolution(self):
        """Test deconvolution-based event detection."""
        kernel = deconv_make_template(
            t_rise=0.0005,
            t_decay=0.001,
            duration=0.005,
            sampling=self.sampling,
        )
        self.assertEqual(len(kernel), 100) # 0.005 / 5e-5 = 100 samples

        # Run deconvolution
        result = deconvolution(
            data=self.data,
            kernel=kernel,
            threshold=-3.0,
            sampling=self.sampling,
        )
        self.assertIsInstance(result, DeconvolutionResult)
        self.assertIn("DeconvolutionResult", repr(result))

    def test_template_matching(self):
        """Test Clements & Bekkers template matching method."""
        kernel = tm_make_template(
            t_rise=0.0005,
            t_decay=0.001,
            baseline=0.001,
            duration=0.004,
            sampling=self.sampling,
        )
        # Run template matching with positive threshold (e.g. 4)
        result = template_matching(
            data=self.data,
            kernel=kernel,
            threshold=4.0,
        )
        self.assertIsInstance(result, TemplateMatchResult)
        self.assertIn("TemplateMatchResult", repr(result))

    def test_template_matching_data_length_error(self):
        """Test template_matching raises an Exception if data is too short."""
        kernel = np.ones(50)
        short_data = np.ones(30)
        with self.assertRaises(Exception):
            template_matching(short_data, kernel, threshold=4.0)


if __name__ == "__main__":
    unittest.main()
