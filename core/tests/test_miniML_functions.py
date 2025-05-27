import unittest
import numpy as np
# Assuming the test file is core/tests/test_miniML_functions.py
# and miniML_functions.py is in core/
from ..miniML_functions import get_event_halfwidth

class TestGetEventHalfwidth(unittest.TestCase):
    def setUp(self):
        self.sampling_rate = 10000.0  # Hz
        self.sampling_interval = 1.0 / self.sampling_rate

    def test_typical_event(self):
        """Test with a well-behaved, symmetrical event."""
        baseline = 0.0
        amplitude = 10.0
        # Triangular pulse: Peak at index 10 (value 10)
        # Rise: 0-10 over 10 samples. Decay: 10-0 over 10 samples.
        event_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        peak_index = 10
        
        # Half amplitude = 5.0
        # Rise crosses 5.0 at index 5. Time = 5 * 0.0001 = 0.0005 s
        # Decay crosses 5.0 at index 15. Time = 15 * 0.0001 = 0.0015 s
        # Half-width: 0.0015 - 0.0005 = 0.0010 s
        expected_t_rise_half = 5.0 * self.sampling_interval
        expected_t_decay_half = 15.0 * self.sampling_interval
        expected_half_width = expected_t_decay_half - expected_t_rise_half

        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6, msg="Typical event: t_rise_half")
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6, msg="Typical event: t_decay_half")
        self.assertAlmostEqual(half_width, expected_half_width, places=6, msg="Typical event: half_width")

    def test_event_too_short_or_flat(self):
        """Test an event that doesn't rise sufficiently or is flat."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        
        # Event rises to 4.0 (below half_amp), then decays.
        event_data_too_low = np.array([0, 1, 2, 3, 4, 3, 2, 1, 0], dtype=float)
        peak_index_too_low = 4

        hw, tr, td = get_event_halfwidth(
            event_data_too_low, peak_index_too_low, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(hw), "Too short (low): half_width")
        self.assertTrue(np.isnan(tr), "Too short (low): t_rise_half")
        self.assertTrue(np.isnan(td), "Too short (low): t_decay_half")

        # Event is just flat at baseline
        event_data_flat = np.zeros(20)
        peak_index_flat = 10
        hw_flat, tr_flat, td_flat = get_event_halfwidth(
            event_data_flat, peak_index_flat, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(hw_flat), "Flat event: half_width")
        self.assertTrue(np.isnan(tr_flat), "Flat event: t_rise_half")
        self.assertTrue(np.isnan(td_flat), "Flat event: t_decay_half")
        
        # Event rises above half_amp, but no points strictly below half_amp for rise phase.
        # (i.e. event_data starts at or above half_amp_level)
        event_data_starts_high = np.array([6,7,8,9,10,9,8,7,6,5,4], dtype=float) # half_amp=5
        peak_idx_sh = 4 # peak value 10
        hw_sh, tr_sh, td_sh = get_event_halfwidth(
            event_data_starts_high, peak_idx_sh, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(tr_sh), "Starts high: t_rise_half should be NaN")
        self.assertTrue(np.isnan(hw_sh), "Starts high: half_width should be NaN")
        # Decay: from 10 (idx 4) down to 4 (idx 10). Crosses 5 at index 9.
        expected_td_sh = 9.0 * self.sampling_interval
        self.assertAlmostEqual(td_sh, expected_td_sh, places=6, msg="Starts high: t_decay_half")


    def test_event_does_not_decay_to_baseline(self):
        """Test event where decay doesn't go below 50% amp."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Rise: 0 to 10. Decay: 10 down to 6 (never crosses 5.0 on decay)
        event_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6], dtype=float)
        peak_index = 10
        
        expected_t_rise_half = 5.0 * self.sampling_interval

        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6, msg="Not decay to baseline: t_rise_half")
        self.assertTrue(np.isnan(t_decay_half), "Not decay to baseline: t_decay_half")
        self.assertTrue(np.isnan(half_width), "Not decay to baseline: half_width")

    def test_event_starts_above_half_amp(self):
        """Test event snippet starting above 50% amplitude."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Data starts at 6.0, rises to 10, decays to 0
        event_data = np.array([6, 7, 8, 9, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        peak_index = 4 # Peak value 10

        # Decay crosses 5 at index 9. Time = 9 * 0.0001 = 0.0009 s
        expected_t_decay_half = 9.0 * self.sampling_interval

        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(t_rise_half), "Starts above half: t_rise_half")
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6, msg="Starts above half: t_decay_half")
        self.assertTrue(np.isnan(half_width), "Starts above half: half_width")

    def test_half_amp_on_sample_point(self):
        """Test when 50% amplitude falls exactly on sample points."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Rise: 0, 2.5, 5 (idx 2), 7.5, 10 (idx 4)
        # Decay: 10, 7.5, 5 (idx 6), 2.5, 0
        event_data = np.array([0, 2.5, 5, 7.5, 10, 7.5, 5, 2.5, 0], dtype=float)
        peak_index = 4
        
        expected_t_rise_half = 2.0 * self.sampling_interval
        expected_t_decay_half = 6.0 * self.sampling_interval
        expected_half_width = expected_t_decay_half - expected_t_rise_half
        
        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6, msg="On sample point: t_rise_half")
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6, msg="On sample point: t_decay_half")
        self.assertAlmostEqual(half_width, expected_half_width, places=6, msg="On sample point: half_width")

    def test_half_amp_requires_interpolation(self):
        """Test when 50% amplitude requires interpolation."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Rise: 0 (idx 0), 4 (idx 1), 8 (idx 2), 10 (idx 3, peak)
        # Interpolated rise for 5: between idx 1 (4) and 2 (8). t = (1 + (5-4)/(8-4)) * si = 1.25 * si
        # Decay: 10 (idx 3), 8 (idx 4), 4 (idx 5), 0 (idx 6)
        # Interpolated decay for 5: between idx 4 (8) and 5 (4). t = (4 + (5-8)/(4-8)) * si = 4.75 * si
        event_data = np.array([0, 4, 8, 10, 8, 4, 0], dtype=float)
        peak_index = 3

        expected_t_rise_half = (1.0 + (5.0-4.0)/(8.0-4.0)) * self.sampling_interval 
        expected_t_decay_half = (4.0 + (5.0-8.0)/(4.0-8.0)) * self.sampling_interval
        expected_half_width = expected_t_decay_half - expected_t_rise_half

        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6, msg="Interpolation: t_rise_half")
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6, msg="Interpolation: t_decay_half")
        self.assertAlmostEqual(half_width, expected_half_width, places=6, msg="Interpolation: half_width")

    def test_peak_at_start_or_end(self):
        """Test when peak_index is at the start or end of data."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        event_data_peak_start = np.array([10, 8, 6, 4, 2, 0], dtype=float) # Peak at idx 0
        
        # Peak at start
        peak_index_start = 0
        hw_s, tr_s, td_s = get_event_halfwidth(
            event_data_peak_start, peak_index_start, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(tr_s), "Peak at start: t_rise_half")
        self.assertTrue(np.isnan(hw_s), "Peak at start: half_width")
        # Decay: 10 (idx 0), 8 (idx 1), 6 (idx 2), 4 (idx 3)
        # Interpolated decay for 5: between idx 2 (6) and 3 (4). t = (2 + (5-6)/(4-6)) * si = 2.5 * si
        expected_td_s = (2.0 + (5.0-6.0)/(4.0-6.0)) * self.sampling_interval
        self.assertAlmostEqual(td_s, expected_td_s, places=6, msg="Peak at start: t_decay_half")

        # Peak at end
        event_data_peak_end = np.array([0, 2, 4, 6, 8, 10], dtype=float) # Peak at idx 5
        peak_index_end = len(event_data_peak_end) - 1
        hw_e, tr_e, td_e = get_event_halfwidth(
            event_data_peak_end, peak_index_end, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(td_e), "Peak at end: t_decay_half")
        self.assertTrue(np.isnan(hw_e), "Peak at end: half_width")
        # Rise: 0 (idx 0), 2 (idx 1), 4 (idx 2), 6 (idx 3), 8 (idx 4), 10 (idx 5)
        # Interpolated rise for 5: between idx 2 (4) and 3 (6). t = (2 + (5-4)/(6-4)) * si = 2.5 * si
        expected_tr_e = (2.0 + (5.0-4.0)/(6.0-4.0)) * self.sampling_interval
        self.assertAlmostEqual(tr_e, expected_tr_e, places=6, msg="Peak at end: t_rise_half")

    def test_zero_amplitude_event(self):
        """Test with zero amplitude."""
        baseline = 0.0
        amplitude = 0.0 # This is key
        event_data = np.zeros(10, dtype=float)
        peak_index = 5

        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(half_width), "Zero amplitude: half_width")
        self.assertTrue(np.isnan(t_rise_half), "Zero amplitude: t_rise_half")
        self.assertTrue(np.isnan(t_decay_half), "Zero amplitude: t_decay_half")

    def test_invalid_inputs(self):
        """Test invalid inputs like negative sampling rate or out-of-bounds peak."""
        baseline = 0.0
        amplitude = 10.0
        event_data = np.array([0,1,2,3,4,5,6,7,8,9,10,9,8,7,6,5,4,3,2,1,0], dtype=float)
        peak_index = 10

        # Negative sampling rate
        hw, tr, td = get_event_halfwidth(event_data, peak_index, baseline, amplitude, -10000.0)
        self.assertTrue(np.isnan(hw) and np.isnan(tr) and np.isnan(td), "Negative sampling rate")
        
        # Zero sampling rate
        hw, tr, td = get_event_halfwidth(event_data, peak_index, baseline, amplitude, 0.0)
        self.assertTrue(np.isnan(hw) and np.isnan(tr) and np.isnan(td), "Zero sampling rate")

        # Peak index out of bounds
        hw, tr, td = get_event_halfwidth(event_data, -1, baseline, amplitude, self.sampling_rate)
        self.assertTrue(np.isnan(hw) and np.isnan(tr) and np.isnan(td), "Peak index -1")
        hw, tr, td = get_event_halfwidth(event_data, len(event_data), baseline, amplitude, self.sampling_rate)
        self.assertTrue(np.isnan(hw) and np.isnan(tr) and np.isnan(td), "Peak index len(data)")
        
        # Empty event_data
        hw, tr, td = get_event_halfwidth(np.array([]), 0, baseline, amplitude, self.sampling_rate)
        self.assertTrue(np.isnan(hw) and np.isnan(tr) and np.isnan(td), "Empty data")

    def test_flat_peak_interpolation(self):
        """ Test event with a flat peak, ensure interpolation points are chosen correctly."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Rise: 0,4,8,10. Interpolated rise for 5: 1.25 * si
        # Flat peak: 10,10,10 (indices 3,4,5)
        # Decay: 10,8,4,0. Interpolated decay for 5 (from start of data):
        #   peak_index=4. Decay starts effectively from event_data[5] (last peak point).
        #   Values for decay consideration: event_data[4:] = [10, 10, 8, 4, 0]
        #   Half amp = 5. Points >= 5 are [10 (idx4), 10 (idx5), 8 (idx6)]. Last is idx6.
        #   Points < 5 are [4 (idx7), 0 (idx8)]. First is idx7.
        #   Interpolate between event_data[6]=8 and event_data[7]=4.
        #   t = (6 + (5-8)/(4-8)) * si = 6.75 * si
        event_data = np.array([0, 4, 8, 10, 10, 10, 8, 4, 0], dtype=float)
        peak_index = 4 # Middle of the flat peak

        expected_t_rise_half = (1.0 + (5.0-4.0)/(8.0-4.0)) * self.sampling_interval # 1.25 * si
        expected_t_decay_half = (6.0 + (5.0-8.0)/(4.0-8.0)) * self.sampling_interval # 6.75 * si
        expected_half_width = expected_t_decay_half - expected_t_rise_half
        
        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6, msg="Flat peak: t_rise_half")
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6, msg="Flat peak: t_decay_half")
        self.assertAlmostEqual(half_width, expected_half_width, places=6, msg="Flat peak: half_width")

if __name__ == '__main__':
    unittest.main()
