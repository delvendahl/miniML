import keras
import numpy as np
import pyqtgraph as pg
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)
from scipy.signal import convolve, find_peaks, resample
from scipy.signal.windows import hann
from sklearn.preprocessing import minmax_scale, scale

from miniml.gui.dialogs.common import finalize_dialog_window
from miniml.resources.util import get_available_models


def hex_to_rgb(hexa):
    """
    Convert a hex color code to a tuple of RGB values.
    """
    return tuple(int(hexa[i : i + 2], 16) for i in (1, 3, 5))


class SettingsPanel(QDialog):
    """
    Dialog for editing core analysis settings.
    """

    def __init__(self, *, settings, parent=None):
        """
        Build controls and initialize them from current settings.
        """
        super().__init__(parent)

        self.stride = QLineEdit(str(settings.stride))
        self.ev_len = QLineEdit(str(settings.event_window))
        self.thresh = QLineEdit(str(settings.event_threshold))
        validator = QDoubleValidator(0.0, 1.0, 3)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.thresh.setValidator(validator)

        self.peak_w = QLineEdit(str(settings.minimum_peak_width))
        self.peak_w.setValidator(QIntValidator(1, 1000))

        self.model = QComboBox()
        self.model.addItems(get_available_models())
        index = self.model.findText(settings.model_name)
        if index >= 0:
            self.model.setCurrentIndex(index)
        self.model.setFixedWidth(200)
        self.direction = QComboBox()
        self.direction.addItems(["negative", "positive"])
        if settings.direction == "negative":
            self.direction.setCurrentIndex(0)
        else:
            self.direction.setCurrentIndex(1)
        self.direction.setFixedWidth(200)
        self.batchsize = QLineEdit(str(settings.batch_size))

        self.filter_factor = QLineEdit(str(settings.filter_factor))
        self.filter_factor.setValidator(QDoubleValidator(1.0, 1000.0, 1))

        self.gradient_convolve_window = QLineEdit(str(settings.gradient_convolve_win))
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


class AutoSettingsWindow(QDialog):
    """
    Helper dialog for estimating analysis settings from selected events.
    """

    def __init__(
        self,
        *,
        trace,
        settings,
        detection,
        on_commit,
        on_warning,
        parent=None,
    ):
        """
        Build interactive plots and controls for auto-setting suggestions.
        """
        super().__init__(parent)
        self.trace = trace
        self.settings = settings
        self.detection = detection
        self.on_commit = on_commit
        self.on_warning = on_warning
        self.peak_window = 200

        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.tracePlot = pg.PlotWidget()
        self.gradient = np.gradient(self.trace.data)
        self.plotData = self.tracePlot.plot(
            self.trace.time_axis,
            self.trace.data,
            pen=pg.mkPen(color=self.settings.colors[3], width=1),
            clear=True,
        )
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", self.trace.y_unit)
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)

        self.tracePlot.setTitle("Double click to mark events", size="16pt", bold=True)
        if self.tracePlot.scene() is not None:
            self.tracePlot.scene().sigMouseClicked.connect(self.mouse_clicked)  # type: ignore

        top_layout.addWidget(self.tracePlot)

        button_layout = QVBoxLayout()
        self.select_button = QPushButton("Select events")
        self.select_button.clicked.connect(self.transfer_events)
        self.select_button.setMinimumWidth(150)
        self.select_button.setMaximumWidth(150)
        self.auto_button = QPushButton("Auto")
        self.auto_button.clicked.connect(self.auto_detect)
        self.auto_button.setMinimumWidth(150)
        self.auto_button.setMaximumWidth(150)
        self.reset_button = QPushButton("Delete cursors")
        self.reset_button.clicked.connect(self.clear_cursors)
        self.reset_button.setMinimumWidth(150)
        self.reset_button.setMaximumWidth(150)
        button_layout.addWidget(self.auto_button)
        button_layout.addStretch()

        self.time = QLineEdit("24.0")
        self.time.setMaximumWidth(150)
        self.time.setValidator(QDoubleValidator(0.0, 10000.0, 5))
        self.time.setStyleSheet("font-weight: bold;")
        time_label = QLabel("Time (ms)")
        time_label.setStyleSheet("font-weight: bold;")
        button_layout.addWidget(time_label)
        button_layout.addWidget(self.time)

        button_layout.addWidget(self.select_button)
        button_layout.addStretch()
        button_layout.addStretch()
        button_layout.addWidget(self.reset_button)

        top_layout.addLayout(button_layout)
        layout.addLayout(top_layout)
        layout.addStretch()
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setLineWidth(1)
        line.setStyleSheet("background-color: #000000;")
        layout.addWidget(line)
        layout.addStretch()

        bottom_layout = QHBoxLayout()

        self.eventPlot = pg.PlotWidget()
        self.eventPlot.setLabel("bottom", "Time", "s")
        self.eventPlot.setLabel("left", "Imon", self.trace.y_unit)
        self.eventPlot.showGrid(x=True, y=True, alpha=0.1)
        self.eventPlot.setTitle("Selected events", size="16pt", bold=True)

        bottom_layout.addWidget(self.eventPlot)

        controls = QFormLayout()
        win_size = self.settings.event_window
        self.window_size = QLineEdit(str(win_size))
        self.window_size.setMinimumWidth(80)
        self.window_size.setValidator(QIntValidator(1, 10000))
        self.window_size.editingFinished.connect(
            lambda: self.update_window_size(int(self.window_size.text()))
        )
        self.window_size.setStyleSheet("font-weight: bold;")
        self.window_time = QLineEdit(str(win_size * self.trace.sampling * 1000))
        self.window_time.setMinimumWidth(80)
        self.window_time.setValidator(QDoubleValidator(0.0001, 10000.0, 3))
        self.window_time.editingFinished.connect(
            lambda: self.update_window_size(float(self.window_time.text()))
        )
        self.window_time.setStyleSheet("font-weight: bold;")
        self.filter_factor = QLineEdit(str(self.settings.filter_factor))
        self.filter_factor.setMinimumWidth(80)
        self.filter_factor.setValidator(QDoubleValidator(1.0, 1000.0, 1))
        self.filter_factor.setStyleSheet("font-weight: bold;")
        self.convolve_window = QLineEdit(str(self.settings.gradient_convolve_win))
        self.convolve_window.setMinimumWidth(80)
        self.convolve_window.setValidator(QIntValidator(1, 1000))
        self.convolve_window.setStyleSheet("font-weight: bold;")

        label_1 = QLabel("Window size (samples)")
        label_1.setStyleSheet("font-weight: bold;")
        controls.addRow(label_1, self.window_size)
        label_2 = QLabel("Window size (ms)")
        label_2.setStyleSheet("font-weight: bold;")
        controls.addRow(label_2, self.window_time)

        auto_win_size = QPushButton("Auto window size")
        auto_win_size.setMinimumWidth(150)
        auto_win_size.setMaximumWidth(150)
        auto_win_size.clicked.connect(self.auto_window_size)
        controls.addRow(auto_win_size)

        controls.addRow(QLabel(""))
        label_3 = QLabel("Filter factor")
        label_3.setStyleSheet("font-weight: bold;")
        controls.addRow(label_3, self.filter_factor)
        label_4 = QLabel("Gradient filter window")
        label_4.setStyleSheet("font-weight: bold;")
        controls.addRow(label_4, self.convolve_window)

        auto_filter = QPushButton("Auto filter factor")
        auto_filter.setMinimumWidth(150)
        auto_filter.setMaximumWidth(150)
        auto_filter.clicked.connect(self.auto_filter_settings)
        controls.addRow(auto_filter)

        bottom_layout.addLayout(controls)

        self.gradientPlot = pg.PlotWidget()
        self.gradientPlot.setLabel("bottom", "Time", "s")
        self.gradientPlot.setLabel("left", "Gradient", f"{self.trace.y_unit}/s")
        self.gradientPlot.showGrid(x=True, y=True, alpha=0.1)
        self.gradientPlot.setTitle("Gradient", size="16pt", bold=True)

        bottom_layout.addWidget(self.gradientPlot)

        layout.addLayout(bottom_layout)

        def custom_accept():
            self.on_commit(
                filter_factor=float(self.filter_factor.text()),
                event_window=int(self.window_size.text()),
                gradient_convolve_win=int(self.convolve_window.text()),
            )
            self.accept()

        self.buttonBox = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttonBox.accepted.connect(custom_accept)
        self.buttonBox.rejected.connect(self.reject)

        layout.addWidget(self.buttonBox)

        self.setLayout(layout)
        self.resize(900, 600)
        self.setWindowTitle("Auto settings")

    def update_window_size(self, new_value):
        """
        Update event window fields and redraw the selection region.
        """
        if isinstance(new_value, int):
            new_value = int(new_value)
        elif isinstance(new_value, float):
            new_value = int(np.round(new_value / self.trace.sampling / 1000))
        else:
            raise TypeError("new_value must be an integer or float")
        self.window_size.setText(str(new_value))
        self.window_time.setText(
            str(np.round(new_value * self.trace.sampling * 1000, 5))
        )
        self.draw_window_region(new_value)

    def mouse_clicked(self, mouseClickEvent):
        """
        Add an event cursor when the user double-clicks the trace plot.
        """
        if mouseClickEvent.double():
            pos = self.tracePlot.plotItem.vb.mapToView(mouseClickEvent.pos())
            x = pos.x()
            y = pos.y()
            cursor = pg.TargetItem(
                pos=(x, y),
                label=f"x={x:.2f}",
                size=12,
                pen=pg.mkPen(color=(255, 202, 58, 255), width=2),
                hoverPen=pg.mkPen(color=(255, 89, 94, 255), width=2),
                brush=pg.mkBrush(color=(255, 202, 58, 100)),
                hoverBrush=pg.mkBrush(color=(255, 89, 94, 100)),
            )
            self.tracePlot.addItem(cursor)

    def transfer_events(self):
        """
        Extract selected events, compute their mean, and update helper plots.
        """
        self.eventPlot.clear()
        self.region = None
        self.ev_positions = []

        extract_window = int(float(self.time.text()) * 1e-3 / self.trace.sampling)
        before = extract_window // 5
        after = extract_window - before

        for item in self.tracePlot.items():
            if isinstance(item, pg.TargetItem):
                x = item.pos()[0]
                index = np.argmin(np.abs(self.trace.time_axis - x))
                event_location = (
                    np.argmax(np.abs(self.gradient[index - self.peak_window : index]))
                    + index
                    - self.peak_window
                )
                self.eventPlot.plot(
                    self.trace.time_axis[: before + after],
                    self.trace.data[event_location - before : event_location + after],
                    pen=pg.mkPen(color=(0, 0, 0, 90), width=2),
                )
                self.ev_positions.append(event_location)

        if len(self.eventPlot.listDataItems()) > 0:
            event_data = np.zeros((len(self.eventPlot.listDataItems()), before + after))
            for i, item in enumerate(self.eventPlot.listDataItems()):
                event_data[i] = item.getData()[1]
            self.avg_event = np.mean(event_data, axis=0)
            self.eventPlot.plot(
                self.trace.time_axis[: before + after],
                self.avg_event,
                pen=pg.mkPen(color=self.settings.colors[3], width=4),
            )
        else:
            return

        self.draw_window_region(self.settings.event_window)

        self.gradientPlot.clear()
        win = hann(self.settings.gradient_convolve_win)
        for item in self.eventPlot.listDataItems():
            if item.opts["pen"].width() == 4:
                continue
            filtered_ev = self.detection.lowpass_filter(
                data=item.getData()[1],
                cutoff=self.detection.trace.sampling_rate / self.settings.filter_factor,
                order=4,
            )
            gradient = convolve(
                np.gradient(filtered_ev, self.trace.sampling), win, mode="same"
            ) / sum(win)
            self.gradientPlot.plot(
                self.trace.time_axis[: before + after],
                gradient,
                pen=pg.mkPen(
                    color=(*hex_to_rgb(self.settings.colors[3]), 100), width=2
                ),
            )

    def draw_window_region(self, win_size):
        """
        Draw or update the average-event window selection overlay.
        """
        peak = np.argmax(np.abs(scale(self.avg_event)))
        win_start = (peak - win_size // 5) * self.trace.sampling
        win_end = (peak + win_size // 1.25) * self.trace.sampling

        def update_positions():
            x1, x2 = self.region.getRegion()
            x1 = int(x1 / self.trace.sampling)
            x2 = int(x2 / self.trace.sampling)
            self.update_window_size(x2 - x1)

        if not hasattr(self, "region") or self.region is None:
            x_max = self.eventPlot.viewRange()[0][1]
            self.region = pg.LinearRegionItem(
                brush=(138, 201, 38, 50),
                hoverBrush=(138, 201, 38, 90),
                pen=(138, 201, 38, 255),
                hoverPen=(0, 0, 0, 255),
                bounds=[0, x_max * 2],
                swapMode="block",
            )
            self.region.setZValue(-1)
            self.eventPlot.addItem(self.region)
            self.region.sigRegionChangeFinished.connect(update_positions)

        self.region.setRegion([win_start, win_end])

    def clear_cursors(self):
        """
        Remove all manually placed event cursors from the trace plot.
        """
        for item in self.tracePlot.items():
            if isinstance(item, pg.TargetItem):
                self.tracePlot.removeItem(item)

    def auto_detect(self):
        """
        Automatically place cursors at candidate high-gradient events.
        """
        normalized_trace = scale(self.detection.trace.data)
        self.filtered_trace = self.detection.lowpass_filter(
            data=normalized_trace,
            cutoff=self.detection.trace.sampling_rate / 50,
            order=4,
        )
        self.gradient = np.gradient(self.filtered_trace)
        threshold = np.std(np.abs(self.gradient)) * 20
        peaks, properties = find_peaks(
            np.abs(self.gradient), height=threshold, width=10, distance=500
        )

        if len(peaks) == 0:
            self.on_warning(message="No events detected.")
            return
        if len(peaks) > 10:
            peaks = peaks[np.argsort(properties["peak_heights"])[-10:]]

        for peak in peaks:
            peak = (
                np.argmax(np.abs(self.filtered_trace[peak : peak + self.peak_window]))
                + peak
            )
            cursor = pg.TargetItem(
                pos=(self.trace.time_axis[peak], self.trace.data[peak]),
                label=f"x={self.trace.time_axis[peak]:.2f}",
                size=12,
                pen=pg.mkPen(color=(255, 202, 58, 255), width=2),
                hoverPen=pg.mkPen(color=(255, 89, 94, 255), width=2),
                brush=pg.mkBrush(color=(255, 202, 58, 100)),
                hoverBrush=pg.mkBrush(color=(255, 89, 94, 100)),
            )
            self.tracePlot.addItem(cursor)

    def auto_window_size(self):
        """
        Suggest an event window size from the selected average waveform.
        """
        if not hasattr(self, "avg_event"):
            self.on_warning(message="Please select events first.")
            return
        t1 = np.argmax(np.abs(scale(self.avg_event)))
        event_avg_copy = np.copy(self.avg_event)

        if event_avg_copy[t1] < np.mean(event_avg_copy[0 : int(t1 // 1.5)]):
            event_avg_copy *= -1

        bsl = np.mean(event_avg_copy[0 : int(t1 // 1.5)])
        t2 = np.argmax(event_avg_copy[t1:] < bsl) + t1
        if t2 == t1:
            t2 = len(event_avg_copy) * 0.75

        suggested_window = ((t2 - t1) // 100 + 1) * 100

        model = keras.models.load_model(self.settings.model_path)
        scores, window_sizes = [], []
        for factor in [0.6, 0.8, 1, 1.2, 1.4, 1.6]:
            window_size = (int(suggested_window * factor) // 100) * 100
            window_sizes.append(window_size)

            win_start = t1 - window_size // 5
            win_start = max(win_start, 0)
            win_end = t1 + window_size // 1.25
            win_end = min(win_end, len(event_avg_copy))
            data = event_avg_copy[int(win_start) : int(win_end)]

            scores.append(
                np.squeeze(
                    model(
                        np.expand_dims(
                            minmax_scale(resample(data, 600) * -1), axis=(0, -1)
                        )
                    )
                )
            )

        best_score = np.argmax(scores)
        self.update_window_size(window_sizes[best_score])

    def auto_filter_settings(self):
        """
        Suggest filter-factor and gradient-window settings from SNR scans.
        """
        if not hasattr(self, "ev_positions"):
            self.on_warning(message="Please select events first.")
            return
        test_values = np.arange(5, 50, 5)
        before = int(int(self.window_size.text()) // 5)
        after = int(self.window_size.text()) - before

        snr_data = []
        for pos in self.ev_positions:
            ev_data = self.trace.data[pos - before : pos + after]
            snr_data.append(
                np.max(np.abs(scale(ev_data)))
                / np.std(np.abs(scale(ev_data)[0 : before // 2]))
            )
        raw_snr = np.mean(np.abs(snr_data))

        result = []
        from itertools import product

        for window, filter_factor in product(test_values, test_values):
            win = hann(window)
            snr_filtered = []
            for pos in self.ev_positions:
                ev_data = self.trace.data[pos - before : pos + after]
                filtered_ev = self.detection.lowpass_filter(
                    data=ev_data,
                    cutoff=self.trace.sampling_rate / filter_factor,
                    order=4,
                )
                gradient = convolve(
                    np.gradient(filtered_ev, self.trace.sampling),
                    win,
                    mode="same",
                ) / sum(win)
                snr_filtered.append(
                    np.max(np.abs(gradient)) / np.std(np.abs(gradient[0 : before // 2]))
                )

            result.append(np.mean(np.abs(snr_filtered)) / raw_snr)

        result = np.array(result).reshape((len(test_values), len(test_values)))
        target_value = 1.5
        closest_value = np.unravel_index(
            np.abs(result - target_value).argmin(), result.shape
        )

        self.filter_factor.setText(str(test_values[closest_value[1]]))
        self.convolve_window.setText(str(test_values[closest_value[0]]))

        self.gradientPlot.clear()
        win = hann(test_values[closest_value[0]])
        for item in self.eventPlot.listDataItems():
            if item.opts["pen"].width() == 4:
                continue
            filtered_ev = self.detection.lowpass_filter(
                data=item.getData()[1],
                cutoff=self.trace.sampling_rate / test_values[closest_value[1]],
                order=4,
            )
            gradient = convolve(
                np.gradient(filtered_ev, self.trace.sampling), win, mode="same"
            ) / sum(win)
            self.gradientPlot.plot(
                self.trace.time_axis[: filtered_ev.shape[0]],
                gradient,
                pen=pg.mkPen(
                    color=(*hex_to_rgb(self.settings.colors[3]), 100), width=2
                ),
            )
