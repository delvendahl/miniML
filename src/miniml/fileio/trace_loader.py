from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pyabf

from miniml.core.trace import MiniTrace


class TraceLoader:
    """
    Encapsulates file loading logic for MiniTrace objects.
    """

    @staticmethod
    def load_trace_from_file(file_type: str, file_args: dict) -> MiniTrace:
        """
        Dispatch loading by GUI file type and return a MiniTrace.
        """
        file_loader = {
            "HEKA DAT": TraceLoader.from_heka_file,
            "AXON ABF": TraceLoader.from_axon_file,
            "HDF5": TraceLoader.from_h5_file,
        }.get(file_type, None)

        if file_loader is None:
            raise ValueError("Unsupported file type.")

        return file_loader(**file_args)

    @staticmethod
    def from_h5_file(
        filename: str,
        tracename: str = "mini_data",
        scaling: float = 1e12,
        sampling: float = 2e-5,
        unit: str = "pA",
    ) -> MiniTrace:
        """
        Load data from an HDF5 file and return a MiniTrace.
        """
        with h5py.File(filename, "r") as f:
            path = f.visit(
                lambda key: (
                    key
                    if isinstance(f[key], h5py.Dataset)
                    and key.split("/")[-1] == tracename
                    else None
                )
            )
            if path is None:
                raise FileNotFoundError("Trace not found in file")
            data = f[path][:] * scaling

        return MiniTrace(
            data=data,
            sampling_interval=sampling,
            y_unit=unit,
            filename=Path(filename).name,
        )

    @staticmethod
    def from_heka_file(
        filename: str,
        rectype: str,
        group: int = 0,
        load_series: list[int] | None = None,
        exclude_series: list[int] | None = None,
        exclude_sweeps: dict[int, list[int]] | None = None,
        scaling: float = 1,
        unit: str = "",
        resample: bool = True,
    ) -> MiniTrace:
        """
        Load data from a HEKA DAT file and return a MiniTrace.
        """
        if not Path(filename).suffix.lower() == ".dat":
            raise ValueError("Incompatible file type. Method only loads .dat files.")

        from miniml.fileio import heka_reader as heka

        bundle = heka.Bundle(filename)

        if group < 0 or group > len(bundle.pul.children) - 1:
            raise IndexError("Group index out of range")

        bundle_series = {}
        for i, SeriesRecord in enumerate(bundle.pul[group].children):
            bundle_series.update({i: SeriesRecord.Label})

        if exclude_series is None:
            exclude_series = []

        if exclude_sweeps is None:
            exclude_sweeps = {}

        if not load_series:
            series_list = [
                series_number
                for series_number, record_type in bundle_series.items()
                if record_type == rectype and series_number not in exclude_series
            ]
        else:
            load_series = [x for x in load_series if x not in exclude_series]
            series_list = [
                series_number
                for series_number, record_type in bundle_series.items()
                if record_type == rectype and series_number in load_series
            ]

        series_data = []
        series_resistances = []
        for series in series_list:
            sweep_data = []
            for sweep in range(len(bundle.pul[group][series])):
                if series not in exclude_sweeps:
                    sweep_data.append(bundle.data[group, series, sweep, 0])
                else:
                    if sweep not in exclude_sweeps[int(series)]:
                        sweep_data.append(bundle.data[group, series, sweep, 0])
            pgf_series_index = (
                sum(len(bundle.pul[i].children) for i in range(group)) + series
            )
            series_data.append(
                (
                    np.array(sweep_data).flatten(),
                    bundle.pgf[pgf_series_index].SampleInterval,
                )
            )
            series_resistances.append(
                (1 / bundle.pul[group][series][0][0].GSeries) * 1e-6
            )

        max_sampling_interval = max([el[1] for el in series_data])
        data = np.array([], dtype=np.float64)
        for i, dat in enumerate(series_data):
            if dat[1] < max_sampling_interval:
                if not resample:
                    raise ValueError(
                        f"Sampling interval of series {i} is smaller than maximum sampling interval of all series"
                    )
                step = int(max_sampling_interval / dat[1])
                data = np.append(data, dat[0][::step])
            else:
                data = np.append(data, dat[0])

        data_unit = unit if unit else bundle.pul[group][series_list[0]][0][0].YUnit

        MiniTrace.excluded_sweeps = exclude_sweeps
        MiniTrace.excluded_series = exclude_series
        MiniTrace.Rseries = series_resistances

        bundle.close()

        return MiniTrace(
            data=data * scaling,
            sampling_interval=max_sampling_interval,
            y_unit=data_unit,
            filename=Path(filename).name,
        )

    @staticmethod
    def from_axon_file(
        filename: str, channel: int = 0, scaling: float = 1.0, unit: str = ""
    ) -> MiniTrace:
        """
        Load data from an AXON ABF file and return a MiniTrace.
        """
        if not Path(filename).suffix.lower() == ".abf":
            raise ValueError("Incompatible file type. Method only loads .abf files.")

        abf_file = pyabf.ABF(filename)
        if channel not in abf_file.channelList:
            raise IndexError("Selected channel does not exist.")

        data_unit = unit if unit else abf_file.adcUnits[channel]

        return MiniTrace(
            data=abf_file.data[channel] * scaling,
            sampling_interval=1 / abf_file.sampleRate,
            y_unit=data_unit,
            filename=Path(filename).name,
        )
