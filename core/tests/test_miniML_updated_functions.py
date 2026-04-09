import unittest
import numpy as np
import sys
import os

# Adjust path to import miniML_updated_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from miniML_updated_functions import (
    get_segment_stats,
    get_steepest_rise_position,
    baseline_score,
    get_event_baseline_v2,
    get_event_baseline_new
)

class TestMiniMLUpdatedFunctions(unittest.TestCase):
    def test_get_segment_stats(self):
        data = np.array([1, 1, 1, 2, 2, 2, 4, 6, 8], dtype=float)
        breakpoints = [3, 6, 9]
        values, variances, slopes = get_segment_stats(breakpoints, data)
        # Segment 1: data[1:2] = [1] -> median=1, var=0, slope=0
        # Segment 2: data[4:5] = [2] -> median=2, var=0, slope=0
        # Segment 3: data[7:8] = [6] -> median=6, var=0, slope=2 (actually Polynomial.fit on [7,8] with [6,8])
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0], 1.0)
        self.assertEqual(values[1], 2.0)
        self.assertEqual(values[2], 6.0)

    def test_get_steepest_rise_position(self):
        data = np.zeros(100)
        data[50:] = np.arange(50) # Linear rise starting at 50
        # steepest rise should be around 50
        pos = get_steepest_rise_position(data, filter_win=5)
        self.assertAlmostEqual(pos, 50, delta=5)

    def test_baseline_score(self):
        positions = [10, 20, 30]
        median_values = [0, 5, 10]
        slope_values = [0, 1, 2]
        variance_values = [0, 1, 2]
        steepest_rise = 40
        score = baseline_score(positions, median_values, slope_values, variance_values, steepest_rise)
        self.assertEqual(len(score), 3)

    def test_get_event_baseline_v2(self):
        # We need enough data for rpt.KernelCPD
        data = np.zeros(200)
        data[100:] = 10
        bsl_duration = 20
        event_num = 0
        relative_event_position = 100
        positions = np.array([500]) # Absolute position in trace

        # This might fail due to the np.trace bug if it hits that branch
        try:
            res = get_event_baseline_v2(data, bsl_duration, event_num, relative_event_position, positions)
            self.assertIsNotNone(res.value)
        except Exception as e:
            self.fail(f"get_event_baseline_v2 raised {type(e).__name__}: {e}")

    def test_get_event_baseline_new(self):
        data = np.zeros(500)
        data[300:] = 10
        bsl_duration = 20
        event_num = 0
        add_points = 200
        peak_position = 350
        positions = np.array([500])

        try:
            res = get_event_baseline_new(data, bsl_duration, event_num, add_points, peak_position, positions)
            self.assertIsNotNone(res.value)
        except Exception as e:
            self.fail(f"get_event_baseline_new raised {type(e).__name__}: {e}")

if __name__ == '__main__':
    unittest.main()
