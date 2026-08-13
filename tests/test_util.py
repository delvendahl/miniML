import unittest
import numpy as np
import tensorflow as tf
from miniml.core.util import exp_fit, minmax_scaling, mEPSC_template, robust_noise_mad


class TestUtils(unittest.TestCase):
    def test_exp_fit(self):
        """Test the single-exponential decay curve evaluation."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        amp = 10.0
        tau = 2.0
        offset = 1.0
        # Formula: amp * np.exp(-(x - x[0]) / tau) + offset
        expected = amp * np.exp(-x / tau) + offset
        res = exp_fit(x, amp, tau, offset)
        np.testing.assert_allclose(res, expected)

    def test_minmax_scaling_tensor(self):
        """Test min-max normalization on a TensorFlow tensor."""
        x = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
        expected = tf.constant([0.0, 0.25, 0.5, 0.75, 1.0], dtype=tf.float32)
        res = minmax_scaling(x)
        np.testing.assert_allclose(res.numpy(), expected.numpy())

    def test_minmax_scaling_flat_tensor(self):
        """Test minmax_scaling when max == min."""
        x = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32)
        expected = tf.constant([0.0, 0.0, 0.0], dtype=tf.float32)
        res = minmax_scaling(x)
        np.testing.assert_allclose(res.numpy(), expected.numpy())

    def test_mEPSC_template(self):
        """Test mEPSC_template waveform generation."""
        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        amplitude = 5.0
        t_rise = 1.0
        t_decay = 2.0
        x0 = 2.0

        # Formula: amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay)
        # for x < x0, value is 0.0
        expected = np.zeros_like(x)
        mask = x >= x0
        expected[mask] = amplitude * (1 - np.exp(-(x[mask] - x0) / t_rise)) * np.exp(-(x[mask] - x0) / t_decay)

        res = mEPSC_template(x, amplitude, t_rise, t_decay, x0)
        np.testing.assert_allclose(res, expected)

    def test_robust_noise_mad(self):
        """Test robust noise threshold calculation using MAD."""
        gradient = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        # median_grad = 0.0
        # abs_dev = [2.0, 1.0, 0.0, 1.0, 2.0]
        # mad = median(abs_dev) = 1.0
        # robust_sigma = 1.4826 * 1.0 = 1.4826
        # threshold = 0.0 + 4.0 * 1.4826 = 5.9304
        expected_threshold = 4.0 * 1.4826
        expected_sigma = 1.4826

        threshold, sigma = robust_noise_mad(gradient, multiplier=4.0)
        self.assertAlmostEqual(sigma, expected_sigma, places=4)
        self.assertAlmostEqual(threshold, expected_threshold, places=4)


if __name__ == "__main__":
    unittest.main()
