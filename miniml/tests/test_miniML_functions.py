import unittest
import numpy as np
import sys
import os

from miniml.miniML_functions import (
    get_event_halfwidth,
    get_event_peak,
    get_event_baseline,
    get_event_onset,
    get_event_risetime,
    get_event_halfdecay_time,
    get_event_charge
)

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
        event_data_starts_high = np.array([6,7,8,9,10,9,8,7,6,5,4], dtype=float) # half_amp=5
        peak_idx_sh = 4 # peak value 10
        hw_sh, tr_sh, td_sh = get_event_halfwidth(
            event_data_starts_high, peak_idx_sh, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(tr_sh), "Starts high: t_rise_half should be NaN")
        self.assertTrue(np.isnan(hw_sh), "Starts high: half_width should be NaN")
        self.assertTrue(np.isnan(td_sh), "Starts high: t_decay_half should be NaN")

    def test_event_does_not_decay_to_baseline(self):
        """Test event where decay doesn't go below 50% amp."""
        baseline = 0.0
        amplitude = 10.0 # Half amp = 5.0
        # Rise: 0 to 10. Decay: 10 down to 6 (never crosses 5.0 on decay)
        event_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 9.5, 9, 8.5, 8, 7.5, 7, 6.5, 6], dtype=float)
        peak_index = 10
        
        half_width, t_rise_half, t_decay_half = get_event_halfwidth(
            event_data, peak_index, baseline, amplitude, self.sampling_rate
        )
        self.assertTrue(np.isnan(t_rise_half), "Not decay to baseline: t_rise_half")
        self.assertTrue(np.isnan(t_decay_half), "Not decay to baseline: t_decay_half")
        self.assertTrue(np.isnan(half_width), "Not decay to baseline: half_width")

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
        self.assertAlmostEqual(t_rise_half, expected_t_rise_half, places=6)
        self.assertAlmostEqual(t_decay_half, expected_t_decay_half, places=6)
        self.assertAlmostEqual(half_width, expected_half_width, places=6)

class TestOtherMiniMLFunctions(unittest.TestCase):
    def test_get_event_peak(self):
        data = np.zeros(20)
        data[5] = 10
        event_num = 0
        add_points = 2
        window_size = 10
        diffs = np.array([20])
        # right_window_limit = 20/5 = 4
        # argmax(data[2:6]) + 2 = argmax([0, 0, 0, 10]) + 2 = 3 + 2 = 5
        peak_pos = get_event_peak(data, event_num, add_points, window_size, diffs)
        self.assertEqual(peak_pos, 5)

    def test_get_event_onset(self):
        data = np.array([0, 0, 0, 1, 5, 10, 8, 6, 4, 2, 0], dtype=float)
        peak_position = 5
        baseline = 0.0
        baseline_var = 2.0
        # bsl_thresh = 0 + 0.25 * 2 = 0.5
        # arr = [0, 0, 0, 1, 5]
        # arr[::-1] = [5, 1, 0, 0, 0]
        # below_threshold = [False, False, True, True, True]
        # argmax(below_threshold) = 2
        # onset_position = 5 - 2 = 3
        onset_pos = get_event_onset(data, peak_position, baseline, baseline_var)
        self.assertEqual(onset_pos, 3)

    def test_get_event_risetime(self):
        # Linear rise from 0 to 10 over 10 samples
        data = np.linspace(0, 10, 11)
        sampling_rate = 10000
        baseline = 0.0
        risetime, min_pos, min_val, max_pos, max_val = get_event_risetime(data, sampling_rate, baseline)
        self.assertGreater(risetime, 0)
        self.assertAlmostEqual(min_val, 1.0, delta=0.5)
        self.assertAlmostEqual(max_val, 9.0, delta=0.5)

    def test_get_event_halfdecay_time(self):
        data = np.array([0, 10, 8, 6, 5, 4, 2, 0], dtype=float)
        peak_position = 1
        baseline = 0.0
        # level = 5.0
        # data[1:] = [10, 8, 6, 5, 4, 2, 0]
        # data[1:] < 5 = [F, F, F, F, T, T, T] -> argmax = 4
        # pos = 1 + 4 = 5
        pos, time = get_event_halfdecay_time(data, peak_position, baseline)
        self.assertEqual(pos, 5)
        self.assertEqual(time, 4)

    def test_get_event_charge(self):
        data = np.array([0, 1, 1, 1, 0], dtype=float)
        start_point = 1
        end_point = 4
        baseline = 0.0
        sampling = 1.0
        charge = get_event_charge(data, start_point, end_point, baseline, sampling)
        self.assertEqual(charge, 2.0)

    def test_get_event_baseline(self):
        data = np.zeros(100)
        data[50:] = 10 # event starts at 50
        duration = 20
        event_num = 0
        add_points = 50
        diffs = np.array([100])
        peak_positions = np.array([60])
        positions = np.array([0])
        
        # previous_peak_in_trace = peak_positions[-1] + positions[-1] - add_points = 60 + 0 - 50 = 10
        # steepest_rise_in_trace = positions[0] = 0
        # Wait, event_num=0 is a special case.
        # else: bsl_duration = duration = 20
        # bsl_end = (50 - (60-50)*3) = 50 - 30 = 20.
        # bsl_start = 20 - 20 = 0.
        # bsl from 0 to 20. All zeros.
        res = get_event_baseline(data, duration, event_num, add_points, diffs, peak_positions, positions)
        self.assertEqual(res.value, 0.0)
        self.assertEqual(res.start, 0)
        self.assertEqual(res.end, 20)

if __name__ == '__main__':
    unittest.main()
