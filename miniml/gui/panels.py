from pathlib import Path

import numpy as np
import pyabf
import pyqtgraph as pg
from PyQt5.QtCore import QSize, Qt, pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIntValidator, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from miniml.fileio import heka_reader as heka
from miniml.gui.util import get_available_models, get_hdf_keys, get_icon_file_path


def finalize_dialog_window(
    window: QDialog, title: str = "new window", cancel: bool = True
) -> None:
    """
    Finalizes a dialog window by adding a OK/Cancel button box to it and setting the window title.

    Args:
        window (QDialog): The dialog window to finalize.
        title (str, optional): The title of the window. Defaults to 'new window'.
        cancel (bool, optional): Whether to include a cancel button. Defaults to True.

    Returns:
        None
    """
    QBtn = (
        (QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        if cancel
        else QDialogButtonBox.Close
    )
    window.buttonBox = QDialogButtonBox(QBtn)
    if cancel:
        window.buttonBox.accepted.connect(window.accept)
        window.buttonBox.rejected.connect(window.reject)
    else:
        window.buttonBox.clicked.connect(window.accept)

    layout = window.layout()
    if isinstance(layout, QFormLayout):
        layout.addRow(window.buttonBox)
    window.setWindowTitle(title)
    window.setWindowModality(pg.QtCore.Qt.WindowModality.ApplicationModal)


class LoadAbfPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.abf_file = pyabf.ABF(parent.filename)

        self.channel = QComboBox()
        self.channel.addItems([str(channel) for channel in self.abf_file.channelList])
        self.channel.setMinimumWidth(150)
        self.channel.currentIndexChanged[str].connect(
            self.on_comboBoxParent_currentChannelChanged
        )

        self.scale = QLineEdit("1")
        self.unit = QLineEdit(self.abf_file.adcUnits[0])
        self.protocol = QLineEdit(self.abf_file.protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setMinimumWidth(300)

        layout = QFormLayout(self)
        layout.addRow("Recording channel:", self.channel)
        layout.addRow("Scaling factor:", self.scale)
        layout.addRow("Data unit:", self.unit)
        layout.addRow("Protocol:", self.protocol)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load AXON .abf file")

    @pyqtSlot(str)
    def on_comboBoxParent_currentChannelChanged(self, index):
        self.unit.clear()
        self.unit.setText(self.abf_file.adcUnits[int(index)])


class LoadHdfPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.e1 = QComboBox()
        self.e1.setMinimumWidth(200)
        self.e1.addItems(get_hdf_keys(parent.filename))
        self.e2 = QLineEdit("2e-5")
        self.e2.setMinimumWidth(200)
        self.e3 = QLineEdit("1e12")
        self.e3.setMinimumWidth(200)
        self.e4 = QLineEdit("pA")
        self.e4.setMinimumWidth(200)

        layout = QFormLayout(self)
        layout.addRow("Dataset name:", self.e1)
        layout.addRow("Sampling interval (s):", self.e2)
        layout.addRow("Scaling factor:", self.e3)
        layout.addRow("Data unit:", self.e4)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load HDF .h5 file")


class LoadDatPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.bundle = heka.Bundle(parent.filename)

        group_series = []
        for i, GroupRecord in enumerate(self.bundle.pul.children):
            group_series.append(str(i + 1) + " - " + GroupRecord.Label)
        self.group = QComboBox()
        self.group.addItems(group_series)
        self.group.setMinimumWidth(150)
        self.group.currentIndexChanged[str].connect(
            self.on_comboBoxParent_currentIndexChanged
        )

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[0].children):
            bundle_series.append(str(i + 1) + " - " + SeriesRecord.Label)
        self.series = QComboBox()
        self.series.addItems(bundle_series)
        self.series.setMinimumWidth(300)
        self.load_option = QCheckBox("")
        self.e1 = QLineEdit("")
        self.e2 = QLineEdit("1e12")
        self.e3 = QLineEdit("pA")

        layout = QFormLayout(self)
        layout.addRow("Import group:", self.group)
        layout.addRow("Import series:", self.series)
        layout.addRow("Import all series of this type:", self.load_option)
        layout.addRow("Exclude selected series:", self.e1)
        layout.addRow("Scaling factor:", self.e2)
        layout.addRow("Data unit:", self.e3)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Load HEKA .dat file")
        self.finished.connect(self.on_dialog_finished)

    @pyqtSlot(str)
    def on_comboBoxParent_currentIndexChanged(self, index):
        group_no, _ = index.split(" - ")

        bundle_series = []
        for i, SeriesRecord in enumerate(self.bundle.pul[int(group_no) - 1].children):
            bundle_series.append(str(i + 1) + " - " + SeriesRecord.Label)

        self.series.clear()
        self.series.addItems(bundle_series)

    @pyqtSlot()
    def on_dialog_finished(self):
        self.bundle.close()


class FileInfoPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.filename = QLineEdit(parent.trace.filename)
        self.filename.setReadOnly(True)
        self.filename.setFixedWidth(300)
        self.format = QLineEdit(parent.filetype)
        self.format.setReadOnly(True)
        self.length = QLineEdit(f"{parent.trace.total_time:.2f}")
        self.length.setReadOnly(True)
        self.unit = QLineEdit(parent.trace.y_unit)
        self.unit.setReadOnly(True)
        self.mode = QLineEdit(parent.recording_mode)
        self.mode.setReadOnly(True)
        self.sampling = QLineEdit(str(np.round(parent.trace.sampling_rate)))
        self.sampling.setReadOnly(True)
        self.protocol = QLineEdit(parent.protocol)
        self.protocol.setReadOnly(True)
        self.protocol.setFixedWidth(300)

        layout = QFormLayout(self)
        layout.addRow("Filename:", self.filename)
        layout.addRow("File format:", self.format)
        layout.addRow("Recording duration (s):", self.length)
        layout.addRow("Data unit", self.unit)
        layout.addRow("Recording mode:", self.mode)
        layout.addRow("Sampling rate (Hz):", self.sampling)
        layout.addRow("Protocol:", self.protocol)
        self.setLayout(layout)

        finalize_dialog_window(self, title="File info", cancel=False)


class AboutPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QFormLayout(self)

        logo = QLabel()
        logo.setPixmap(
            QPixmap(str(Path(__file__).parent / "minML_icon.png")).scaled(
                QSize(100, 100)
            )
        )
        layout.addRow(logo)

        self.version = QLabel("miniML version 1.0.0")
        layout.addRow(self.version)

        self.author = QLabel(
            "Authors: Philipp O'Neill, Martin Baccino Calace, Igor Delvendahl"
        )
        layout.addRow(self.author)

        self.website = QLabel(
            'Website: <a href="https://github.com/delvendahl/miniML">miniML GitHub repository</a>'
        )
        self.website.setOpenExternalLinks(True)
        layout.addRow(self.website)

        self.paper = QLabel(
            'Publication: <a href="https://doi.org/10.7554/eLife.98485.3">miniML eLife paper 2025</a>'
        )
        self.paper.setOpenExternalLinks(True)
        layout.addRow(self.paper)

        self.setLayout(layout)

        finalize_dialog_window(self, title="About miniML", cancel=False)


class SummaryPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.populate_fields(parent)
        layout = QFormLayout(self)
        layout.addRow("Filename:", self.filename)
        layout.addRow("Events found:", self.event_count)
        layout.addRow("Events deleted:", self.deleted_event_count)
        layout.addRow("Event frequency (Hz):", self.event_frequency)
        layout.addRow("Average score:", self.average_score)
        layout.addRow(
            f"Average amplitude ({parent.detection.trace.y_unit}):",
            self.average_amplitude,
        )
        layout.addRow(
            f"Median amplitude ({parent.detection.trace.y_unit}):",
            self.median_amplitude,
        )
        layout.addRow("Coefficient of variation:", self.amplitude_cv)
        layout.addRow(
            f"Average area ({parent.detection.trace.y_unit}*s):", self.average_area
        )
        layout.addRow("Average risetime (ms):", self.average_rise_time)
        layout.addRow("Average rise slope (pA/ms):", self.average_slope)
        layout.addRow("Average 50% decay time (ms):", self.average_decay_time)
        layout.addRow("Average halfwidth (ms):", self.average_halfwidth)
        layout.addRow("Decay time constant (ms):", self.decay_tau)
        self.setLayout(layout)

        finalize_dialog_window(self, title="Summary", cancel=False)

    def populate_fields(self, parent):
        self.filename = QLineEdit(parent.trace.filename)
        self.filename.setReadOnly(True)
        self.event_count = QLineEdit(str(parent.detection.event_stats.event_count))
        self.event_count.setReadOnly(True)
        self.deleted_event_count = QLineEdit(str(parent.detection.deleted_events))
        self.deleted_event_count.setReadOnly(True)
        self.event_frequency = QLineEdit(
            f"{parent.detection.event_stats.frequency():.5f}"
        )
        self.event_frequency.setReadOnly(True)
        self.average_score = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.event_scores):.5f}"
        )
        self.average_score.setReadOnly(True)
        self.average_amplitude = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.amplitudes):.5f}"
        )
        self.average_amplitude.setReadOnly(True)
        self.median_amplitude = QLineEdit(
            f"{parent.detection.event_stats.median(parent.detection.event_stats.amplitudes):.5f}"
        )
        self.median_amplitude.setReadOnly(True)
        self.amplitude_cv = QLineEdit(
            f"{parent.detection.event_stats.cv(parent.detection.event_stats.amplitudes):.5f}"
        )
        self.amplitude_cv.setReadOnly(True)
        self.average_area = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.charges):.5f}"
        )
        self.average_area.setReadOnly(True)
        self.average_rise_time = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.risetimes) * 1e3:.5f}"
        )
        self.average_rise_time.setReadOnly(True)
        self.average_slope = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.slopes) * 1e-3:.5f}"
        )
        self.average_slope.setReadOnly(True)
        self.average_decay_time = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.halfdecays) * 1e3:.5f}"
        )
        self.average_decay_time.setReadOnly(True)
        self.average_halfwidth = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.halfwidths) * 1e3:.5f}"
        )
        self.average_halfwidth.setReadOnly(True)
        self.decay_tau = QLineEdit(
            f"{parent.detection.event_stats.mean(parent.detection.event_stats.avg_tau_decay) * 1e3:.5f}"
        )
        self.decay_tau.setReadOnly(True)


class SettingsPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.stride = QLineEdit(str(parent.settings.stride))
        self.ev_len = QLineEdit(str(parent.settings.event_window))
        self.thresh = QLineEdit(str(parent.settings.event_threshold))
        validator = QDoubleValidator(0.0, 1.0, 3)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.thresh.setValidator(validator)

        self.peak_w = QLineEdit(str(parent.settings.minimum_peak_width))
        self.peak_w.setValidator(QIntValidator(1, 1000))

        self.model = QComboBox()
        self.model.addItems(get_available_models())
        index = self.model.findText(parent.settings.model_name)
        if index >= 0:
            self.model.setCurrentIndex(index)
        self.model.setFixedWidth(200)
        self.direction = QComboBox()
        self.direction.addItems(["negative", "positive"])
        if parent.settings.direction == "negative":
            self.direction.setCurrentIndex(0)
        else:
            self.direction.setCurrentIndex(1)
        self.direction.setFixedWidth(200)
        self.batchsize = QLineEdit(str(parent.settings.batch_size))

        self.filter_factor = QLineEdit(str(parent.settings.filter_factor))
        self.filter_factor.setValidator(QDoubleValidator(1.0, 1000.0, 1))

        self.gradient_convolve_window = QLineEdit(
            str(parent.settings.gradient_convolve_win)
        )
        self.gradient_convolve_window.setValidator(QIntValidator(1, 10000))

        layout = QFormLayout(self)
        layout.addRow("Stride length (samples)", self.stride)
        layout.addRow("Event length (samples)", self.ev_len)
        layout.addRow("Min. peak height (0-1)", self.thresh)
        layout.addRow("Min. peak width (samples)", self.peak_w)
        layout.addRow("Model", self.model)
        layout.addRow("Event direction", self.direction)
        layout.addRow("Batch size", self.batchsize)
        layout.addRow("Filter factor", self.filter_factor)
        layout.addRow("Gradient filter window", self.gradient_convolve_window)
        self.setLayout(layout)

        finalize_dialog_window(self, title="miniML settings")


class CustomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwds):
        kwds["enableMenu"] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    ## reimplement right-click to zoom out
    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton:
            self.autoRange()

    ## reimplement mouseDragEvent to disable continuous axis zoom
    def mouseDragEvent(self, ev, axis=None):
        if axis is not None and ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)


class CutPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        vb = CustomViewBox()

        self.tracePlot = pg.PlotWidget(viewBox=vb)
        # self.plotData = self.tracePlot.plot(parent.trace.time_axis, parent.trace.data, pen=pg.mkPen(color='#1982C4', width=1), clear=True)
        self.plotData = self.tracePlot.plot(
            parent.time_ax_display,
            parent.data_display,
            pen=pg.mkPen(color="#1982C4", width=1),
            clear=True,
        )

        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", parent.trace.y_unit)

        self.region = pg.LinearRegionItem(
            brush=(138, 201, 38, 50),
            hoverBrush=(138, 201, 38, 100),
            pen=(46, 64, 87, 50),
            hoverPen=(0, 0, 0, 255),
            bounds=[0, parent.trace.total_time],
            swapMode="block",
        )
        self.region.setZValue(-1)
        self.region.setRegion([0, parent.trace.total_time])
        self.tracePlot.addItem(self.region)

        def update_positions():
            x1, x2 = self.region.getRegion()
            self.start.setText(str(np.round(x1, 5)))
            self.end.setText(str(np.round(x2, 5)))

        self.region.sigRegionChanged.connect(update_positions)

        def updateRegion():
            self.region.setRegion([float(self.start.text()), float(self.end.text())])

        def toggle_region_brush():
            if self.switch.isChecked():
                self.region.setBrush((255, 89, 94, 50))
                self.region.setHoverBrush((255, 89, 94, 50))
            else:
                self.region.setBrush((138, 201, 38, 50))
                self.region.setHoverBrush((138, 201, 38, 50))
            self.tracePlot.update()

        trace_validator = QDoubleValidator(0.0, parent.trace.total_time, 5)
        start_label = QLabel("Position 1 (s)")
        start_label.setStyleSheet("font-weight: bold;")
        self.start = QLineEdit("0.0")
        self.start.setMinimumWidth(80)
        self.start.setValidator(trace_validator)
        self.start.setStyleSheet("font-weight: bold;")
        self.start.editingFinished.connect(updateRegion)

        end_label = QLabel("Position 2 (s)")
        end_label.setStyleSheet("font-weight: bold;")
        self.end = QLineEdit(str(np.round(parent.trace.total_time)))
        self.end.setMinimumWidth(80)
        self.end.setValidator(trace_validator)
        self.end.setStyleSheet("font-weight: bold;")
        self.end.editingFinished.connect(updateRegion)

        self.toggle_label1 = QLabel("Cut between cursors")
        self.toggle_label1.setStyleSheet("font-weight: bold;")
        self.switch = QCheckBox()
        self.switch.setChecked(False)
        icon_toggle_off = get_icon_file_path("toggle_off_24px.svg")
        icon_toggle_on = get_icon_file_path("toggle_on_24px.svg")
        self.switch.setStyleSheet(f"""
            QCheckBox::indicator:unchecked {{
                image: url({icon_toggle_off});
                width: 48;
                height: 48;
            }}
            QCheckBox::indicator:checked {{
                image: url({icon_toggle_on});
                width: 48;
                height: 48;
            }}
        """)
        self.switch.stateChanged.connect(toggle_region_brush)

        lower_layout = QHBoxLayout()

        icon_first_page = get_icon_file_path("first_page_24dp.svg")
        icon_last_page = get_icon_file_path("last_page_24dp.svg")
        cursor1_icon = QLabel()
        cursor1_icon.setPixmap(QPixmap(icon_first_page))
        cursor1_icon.setFixedSize(36, 36)
        cursor2_icon = QLabel()
        cursor2_icon.setPixmap(QPixmap(icon_last_page))
        cursor2_icon.setFixedSize(36, 36)
        lower_layout.addWidget(cursor1_icon)
        lower_layout.addWidget(start_label)
        lower_layout.addWidget(self.start)
        lower_layout.addStretch()
        lower_layout.addWidget(cursor2_icon)
        lower_layout.addWidget(end_label)
        lower_layout.addWidget(self.end)
        lower_layout.addWidget(self.toggle_label1)
        lower_layout.addWidget(self.switch)

        layout.addWidget(self.tracePlot)
        layout.addLayout(lower_layout)
        self.setLayout(layout)

        def custom_accept():
            if self.buttonBox.button(QDialogButtonBox.Ok).underMouse():
                self.accept()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Cancel | QDialogButtonBox.Ok)
        self.buttonBox.accepted.connect(custom_accept)
        self.buttonBox.rejected.connect(self.reject)
        layout.addWidget(self.buttonBox)
        self.setWindowTitle("Cut trace")
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.resize(600, 400)


class FilterPanel(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.tracePlot = pg.PlotWidget()
        self.plotData = self.tracePlot.plot(
            parent.time_ax_display,
            parent.data_display,
            pen=pg.mkPen(color="grey", width=1),
            clear=True,
        )
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", parent.trace.y_unit)
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)

        self.filtered_trace_plot = pg.PlotDataItem(
            parent.time_ax_display,
            parent.data_display,
            pen=pg.mkPen(color="grey", width=1),
        )
        self.tracePlot.addItem(self.filtered_trace_plot)

        layout.addWidget(self.tracePlot)

        def comboBoxIndexChanged(index):
            if index == 1:
                self.time_window.setEnabled(True)
                self.low.setEnabled(False)
                self.order.setEnabled(True)
                self.hann_window.setEnabled(False)
            elif index == 2:
                self.time_window.setEnabled(False)
                self.low.setEnabled(False)
                self.order.setEnabled(False)
                self.hann_window.setEnabled(True)
            else:
                self.time_window.setEnabled(False)
                self.low.setEnabled(True)
                self.order.setEnabled(True)
                self.hann_window.setEnabled(False)
            filter_toggled()

        def filter_toggled():
            if not np.any(
                [
                    self.highpass.isChecked(),
                    self.lowpass.isChecked(),
                    self.line_noise.isChecked(),
                    self.detrend.isChecked(),
                ]
            ):
                self.filtered_trace_plot.setData(
                    parent.time_ax_display,
                    parent.trace.data,
                    pen=pg.mkPen(color="grey", width=1),
                )

                self.filtered_trace_plot.setPen(pg.mkPen(color="grey", width=1))
                return

            self.filtered_trace = parent.trace
            if self.detrend.isChecked():
                self.filtered_trace = self.filtered_trace.detrend(
                    num_segments=int(self.num_segments.text())
                )
            if self.highpass.isChecked():
                self.filtered_trace = self.filtered_trace.filter(
                    highpass=float(self.high.text()), order=int(self.order.text())
                )
            if self.line_noise.isChecked():
                self.filtered_trace = self.filtered_trace.filter(
                    line_freq=float(self.line_freq.text()),
                    width=float(self.notch_width.text()),
                )
            if self.lowpass.isChecked():
                if self.filter_type.currentText() == "Butterworth":
                    self.filtered_trace = self.filtered_trace.filter(
                        lowpass=float(self.low.text()), order=int(self.order.text())
                    )
                elif self.filter_type.currentText() == "Savitzky-Golay":
                    self.filtered_trace = self.filtered_trace.filter(
                        savgol=float(self.time_window.text()),
                        order=int(self.order.text()),
                    )
                else:
                    self.filtered_trace = self.filtered_trace.filter(
                        hann=int(self.hann_window.text())
                    )

            self.data_display, self.time_ax_display = parent.resample_for_display(
                data=self.filtered_trace.data, time_axis=self.filtered_trace.time_axis
            )
            self.filtered_trace_plot.setData(self.time_ax_display, self.data_display)
            self.filtered_trace_plot.setPen(pg.mkPen(color="#ffca3a", width=1))

        self.detrend = QCheckBox("")
        self.detrend.stateChanged.connect(filter_toggled)
        self.num_segments = QLineEdit("1")
        self.num_segments.setValidator(QIntValidator(1, 9999))
        self.highpass = QCheckBox("")
        self.highpass.stateChanged.connect(filter_toggled)
        self.high = QLineEdit("0.5")
        self.high.setValidator(QDoubleValidator(0.01, 99.99, 2))
        self.line_noise = QCheckBox("")
        self.line_noise.stateChanged.connect(filter_toggled)
        self.line_freq = QLineEdit("50")
        self.line_freq.setValidator(QDoubleValidator(0.9, 9999.9, 1))
        self.notch_width = QLineEdit("3.0")
        self.notch_width.setValidator(QDoubleValidator(0.01, 9999.9, 2))
        self.lowpass = QCheckBox("")
        self.lowpass.stateChanged.connect(filter_toggled)
        self.low = QLineEdit("750")
        self.low.setValidator(QDoubleValidator(0.9, 99999.9, 1))
        self.low.editingFinished.connect(filter_toggled)
        self.filter_type = QComboBox()
        self.filter_type.addItems(["Butterworth", "Savitzky-Golay", "Hann window"])
        self.filter_type.currentIndexChanged.connect(comboBoxIndexChanged)
        self.filter_type.setFixedWidth(200)
        self.time_window = QLineEdit("5.0")
        self.time_window.setValidator(QDoubleValidator(0.001, 999.9, 3))
        self.time_window.setEnabled(False)
        self.time_window.editingFinished.connect(filter_toggled)
        self.order = QLineEdit("4")
        self.order.setValidator(QIntValidator(1, 9))
        self.order.editingFinished.connect(filter_toggled)
        self.hann_window = QLineEdit("20")
        self.hann_window.setValidator(QIntValidator(3, 1000))
        self.hann_window.setEnabled(False)
        self.hann_window.editingFinished.connect(filter_toggled)

        controls1 = QFormLayout()
        controls1.addRow("Detrend data", self.detrend)
        controls1.addRow("Number of segments", self.num_segments)
        controls1.addRow("High-pass filter", self.highpass)
        controls1.addRow("High-pass (Hz)", self.high)
        controls1.addRow("Line noise filter", self.line_noise)
        controls1.addRow("Line noise frequency (Hz)", self.line_freq)
        controls1.addRow("Line noise width (Hz)", self.notch_width)

        controls2 = QFormLayout()
        controls2.addRow("Lowpass filter", self.lowpass)
        controls2.addRow("Filter type", self.filter_type)
        controls2.addRow("Low-pass (Hz)", self.low)
        controls2.addRow("Filter order", self.order)
        controls2.addRow("Window (ms)", self.time_window)
        controls2.addRow("Hann window size", self.hann_window)

        lower_layout = QHBoxLayout()
        lower_layout.addLayout(controls1)
        lower_layout.addLayout(controls2)
        layout.addLayout(lower_layout)

        def custom_accept():
            if hasattr(self, "filtered_trace"):
                parent.trace = self.filtered_trace
            self.accept()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(custom_accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.resize(600, 500)
        self.setWindowTitle("Filter data")
