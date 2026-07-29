import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
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

from miniml.resources.util import get_icon_file_path


class CustomViewBox(pg.ViewBox):
    """
    Plot view box with constrained mouse interaction behavior.
    """

    def __init__(self, *args, **kwds):
        """
        Disable the context menu and enable rectangle zoom mode.
        """
        kwds["enableMenu"] = False
        pg.ViewBox.__init__(self, *args, **kwds)
        self.setMouseMode(self.RectMode)

    def mouseClickEvent(self, ev):
        """
        Reset the view range on right-click.
        """
        if ev.button() == Qt.MouseButton.RightButton:
            self.autoRange()

    def mouseDragEvent(self, ev, axis=None):
        """
        Ignore right-button drags while preserving other drag actions.
        """
        if axis is not None and ev.button() == Qt.MouseButton.RightButton:
            ev.ignore()
        else:
            pg.ViewBox.mouseDragEvent(self, ev, axis=axis)


class CutPanel(QDialog):
    """
    Dialog for selecting and confirming trace cut boundaries.
    """

    def __init__(
        self,
        *,
        time_ax_display,
        data_display,
        y_unit: str,
        total_time: float,
        parent=None,
    ):
        """
        Build the cut UI and bind region and form interactions.
        """
        super().__init__(parent)

        layout = QVBoxLayout(self)
        vb = CustomViewBox()

        self.tracePlot = pg.PlotWidget(viewBox=vb)
        self.plotData = self.tracePlot.plot(
            time_ax_display,
            data_display,
            pen=pg.mkPen(color="#1982C4", width=1),
            clear=True,
        )

        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", y_unit)

        self.region = pg.LinearRegionItem(
            brush=(138, 201, 38, 50),
            hoverBrush=(138, 201, 38, 100),
            pen=(46, 64, 87, 50),
            hoverPen=(0, 0, 0, 255),
            bounds=[0, total_time],
            swapMode="block",
        )
        self.region.setZValue(-1)
        self.region.setRegion([0, total_time])
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

        trace_validator = QDoubleValidator(0.0, total_time, 5)
        start_label = QLabel("Position 1 (s)")
        start_label.setStyleSheet("font-weight: bold;")
        self.start = QLineEdit("0.0")
        self.start.setMinimumWidth(80)
        self.start.setValidator(trace_validator)
        self.start.setStyleSheet("font-weight: bold;")
        self.start.editingFinished.connect(updateRegion)

        end_label = QLabel("Position 2 (s)")
        end_label.setStyleSheet("font-weight: bold;")
        self.end = QLineEdit(str(np.round(total_time)))
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
    """
    Dialog for configuring and previewing trace filters.
    """

    def __init__(
        self,
        *,
        trace,
        time_ax_display,
        data_display,
        resample_for_display,
        parent=None,
    ):
        """
        Build filter controls and live preview plotting.
        """
        super().__init__(parent)

        self.source_trace = trace
        self.resample_for_display_fn = resample_for_display

        layout = QVBoxLayout(self)

        self.tracePlot = pg.PlotWidget()
        self.plotData = self.tracePlot.plot(
            time_ax_display,
            data_display,
            pen=pg.mkPen(color="grey", width=1),
            clear=True,
        )
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", trace.y_unit)
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)

        self.filtered_trace_plot = pg.PlotDataItem(
            time_ax_display,
            data_display,
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
                    time_ax_display,
                    self.source_trace.data,
                    pen=pg.mkPen(color="grey", width=1),
                )

                self.filtered_trace_plot.setPen(pg.mkPen(color="grey", width=1))
                return

            self.filtered_trace = self.source_trace
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

            self.data_display, self.time_ax_display = self.resample_for_display_fn(
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
            self.accept()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(custom_accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.resize(600, 500)
        self.setWindowTitle("Filter data")
