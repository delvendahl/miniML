import unittest
import numpy as np

from miniml.core.event import Event
from miniml.core.detection import EventDetection
from miniml.core.trace import MiniTrace


class TestEventClass(unittest.TestCase):
    def test_initialization_and_properties(self):
        """Test that Event class initializes with all properties and a default excluded=False flag."""
        ev = Event(
            location=123,
            score=0.95,
            waveform=np.array([1.0, 2.0, 3.0]),
            peak_location=125,
            peak_value=2.5,
            bsl_start=100,
            bsl_end=110,
            bsl_value=0.5,
            bsl_duration=10.0,
            onset_location=115,
            decaytime=0.005,
            charge=1.2,
            risetime=0.001,
            half_decay=130.0,
            halfwidth=0.002,
            rise_half_amp_time=0.0015,
            decay_half_amp_time=0.0035,
            min_position_rise=116.0,
            max_position_rise=124.0,
            min_value_rise=0.6,
            max_value_rise=2.4,
            slope=10.0,
        )

        self.assertEqual(ev.location, 123)
        self.assertEqual(ev.score, 0.95)
        np.testing.assert_array_equal(ev.waveform, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(ev.peak_location, 125)
        self.assertEqual(ev.peak_value, 2.5)
        self.assertEqual(ev.bsl_start, 100)
        self.assertEqual(ev.bsl_end, 110)
        self.assertEqual(ev.bsl_value, 0.5)
        self.assertEqual(ev.bsl_duration, 10.0)
        self.assertEqual(ev.onset_location, 115)
        self.assertEqual(ev.decaytime, 0.005)
        self.assertEqual(ev.charge, 1.2)
        self.assertEqual(ev.risetime, 0.001)
        self.assertEqual(ev.half_decay, 130.0)
        self.assertEqual(ev.halfwidth, 0.002)
        self.assertEqual(ev.rise_half_amp_time, 0.0015)
        self.assertEqual(ev.decay_half_amp_time, 0.0035)
        self.assertEqual(ev.min_position_rise, 116.0)
        self.assertEqual(ev.max_position_rise, 124.0)
        self.assertEqual(ev.min_value_rise, 0.6)
        self.assertEqual(ev.max_value_rise, 2.4)
        self.assertEqual(ev.slope, 10.0)
        self.assertFalse(ev.excluded)

        # Toggle exclusion flag
        ev.excluded = True
        self.assertTrue(ev.excluded)


class TestEventDetectionProperties(unittest.TestCase):
    def test_property_arrays_with_detected_events(self):
        """Test that EventDetection property getters/setters map correctly to detected_events."""
        trace = MiniTrace(data=np.zeros(1000), sampling_interval=1.0/10000.0)
        detection = EventDetection(data=trace)

        # Create dummy events
        ev1 = Event(location=100, score=0.9, peak_location=105, bsl_value=0.1, slope=1.5, excluded=False)
        ev2 = Event(location=200, score=0.8, peak_location=205, bsl_value=0.2, slope=2.5, excluded=True)

        detection.detected_events = [ev1, ev2]

        # Verify property arrays
        np.testing.assert_array_equal(detection.event_locations, np.array([100, 200], dtype=np.int64))
        np.testing.assert_array_equal(detection.event_scores, np.array([0.9, 0.8], dtype=np.float64))
        np.testing.assert_array_equal(detection.event_peak_locations, np.array([105, 205], dtype=np.int64))
        np.testing.assert_array_equal(detection.event_bsls, np.array([0.1, 0.2], dtype=np.float64))
        np.testing.assert_array_equal(detection.slopes, np.array([1.5, 2.5], dtype=np.float64))
        np.testing.assert_array_equal(detection.exclude_events, np.array([0, 1], dtype=np.int64))

        # Test setter updates
        detection.event_locations = np.array([110, 210])
        self.assertEqual(detection.detected_events[0].location, 110)
        self.assertEqual(detection.detected_events[1].location, 210)

        detection.exclude_events = np.array([1, 0])
        self.assertTrue(detection.detected_events[0].excluded)
        self.assertFalse(detection.detected_events[1].excluded)

    def test_property_arrays_backing_fallback(self):
        """Test that EventDetection falls back to backing variables when detected_events is empty."""
        trace = MiniTrace(data=np.zeros(1000), sampling_interval=1.0/10000.0)
        detection = EventDetection(data=trace)

        self.assertEqual(len(detection.event_locations), 0)
        detection.event_locations = np.array([10, 20])
        np.testing.assert_array_equal(detection.event_locations, np.array([10, 20]))


if __name__ == "__main__":
    unittest.main()
