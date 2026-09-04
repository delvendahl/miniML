import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import h5py
import numpy as np

from miniml.fileio.trace_loader import TraceLoader


class TestTraceLoader(unittest.TestCase):
    def test_load_trace_from_file_unsupported(self):
        """Test that load_trace_from_file raises ValueError on unsupported types."""
        with self.assertRaises(ValueError):
            TraceLoader.load_trace_from_file("txt", {})

    def test_from_h5_file_success(self):
        """Test loading trace from HDF5 file successfully."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Create h5 dataset
            dummy_data = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
            with h5py.File(temp_path, "w") as f_h5:
                # mini_data dataset inside group
                grp = f_h5.create_group("subgroup")
                grp.create_dataset("mini_data", data=dummy_data)

            trace = TraceLoader.from_h5_file(
                filename=temp_path,
                tracename="mini_data",
                scaling=1.0,
                sampling=0.001,
                unit="pA",
            )
            self.assertEqual(trace.y_unit, "pA")
            self.assertEqual(trace.sampling, 0.001)
            np.testing.assert_allclose(trace.data, dummy_data)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_from_h5_file_not_found(self):
        """Test that loading a non-existent dataset raises FileNotFoundError."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            with h5py.File(temp_path, "w") as f_h5:
                f_h5.create_dataset("different_name", data=[1.0, 2.0])

            with self.assertRaises(FileNotFoundError):
                TraceLoader.from_h5_file(temp_path, tracename="mini_data")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_from_axon_file_invalid_extension(self):
        """Test that from_axon_file raises ValueError for non-abf file."""
        with self.assertRaises(ValueError):
            TraceLoader.from_axon_file("test.txt")

    @patch("pyabf.ABF")
    def test_from_axon_file_success(self, mock_abf_class):
        """Test that from_axon_file loads data successfully from mock ABF."""
        mock_abf = MagicMock()
        mock_abf.channelList = [0, 1]
        mock_abf.data = [np.array([10.0, 20.0, 30.0]), np.array([1.0, 2.0, 3.0])]
        mock_abf.adcUnits = ["pA", "mV"]
        mock_abf.sampleRate = 10000.0
        mock_abf_class.return_value = mock_abf

        trace = TraceLoader.from_axon_file("test.abf", channel=0, scaling=2.0)
        np.testing.assert_allclose(trace.data, np.array([20.0, 40.0, 60.0]))
        self.assertEqual(trace.y_unit, "pA")
        self.assertAlmostEqual(trace.sampling, 0.0001)

    @patch("pyabf.ABF")
    def test_from_axon_file_index_error(self, mock_abf_class):
        """Test that from_axon_file raises IndexError for missing channel."""
        mock_abf = MagicMock()
        mock_abf.channelList = [0]
        mock_abf_class.return_value = mock_abf

        with self.assertRaises(IndexError):
            TraceLoader.from_axon_file("test.abf", channel=1)

    def test_from_heka_file_invalid_extension(self):
        """Test that from_heka_file raises ValueError for non-dat file."""
        with self.assertRaises(ValueError):
            TraceLoader.from_heka_file("test.txt", rectype="mEPSC")

    @patch("miniml.fileio.heka_reader.Bundle")
    def test_from_heka_file_success(self, mock_bundle_class):
        """Test loading from HEKA dat file with mocked Bundle contents."""
        mock_bundle = MagicMock()
        # pul has hierarchy: pul[group_index].children returns Series list
        # pul[group_index][series_index] returns sweep list
        # pul[group_index][series_index][sweep_index][channel_index].GSeries returns float
        # pul[group_index][series_index][sweep_index][channel_index].YUnit returns str

        # Mock group
        mock_series_record = MagicMock()
        mock_series_record.Label = "mEPSC"
        mock_pul_group = MagicMock()
        mock_pul_group.children = [mock_series_record]

        # Mock sweeps in pul[0][0]
        mock_sweep_record = MagicMock()
        mock_channel_record = MagicMock()
        mock_channel_record.GSeries = 1e6  # 1 MOhm series resistance => 1e-6 Siemens
        mock_channel_record.YUnit = "A"
        mock_sweep_record.__getitem__.return_value = mock_channel_record
        mock_pul_group.__getitem__.return_value = [mock_sweep_record]

        mock_bundle.pul.children = [mock_pul_group]
        mock_bundle.pul.__getitem__.return_value = mock_pul_group

        # Mock pgf SampleInterval
        mock_pgf_record = MagicMock()
        mock_pgf_record.SampleInterval = 0.0001
        mock_bundle.pgf = [mock_pgf_record]

        # Mock data array
        # bundle.data[group, series, sweep, channel]
        mock_bundle.data = MagicMock()
        mock_bundle.data.__getitem__.return_value = np.array([1.0, 2.0, 3.0])

        mock_bundle_class.return_value = mock_bundle

        # Test loading
        trace = TraceLoader.from_heka_file(
            filename="test.dat",
            rectype="mEPSC",
            group=0,
            scaling=1e12,  # Scale from A to pA
            unit="pA",
            resample=True,
        )

        self.assertEqual(trace.y_unit, "pA")
        self.assertAlmostEqual(trace.sampling, 0.0001)
        np.testing.assert_allclose(trace.data, np.array([1e12, 2e12, 3e12]))

    @patch("miniml.fileio.heka_reader.Bundle")
    def test_from_heka_file_group_out_of_range(self, mock_bundle_class):
        """Test that from_heka_file raises IndexError for out-of-range group."""
        mock_bundle = MagicMock()
        mock_bundle.pul.children = []  # No groups
        mock_bundle_class.return_value = mock_bundle

        with self.assertRaises(IndexError):
            TraceLoader.from_heka_file("test.dat", rectype="mEPSC", group=1)


if __name__ == "__main__":
    unittest.main()
