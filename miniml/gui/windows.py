from pathlib import Path

import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtGui import QCursor, QDoubleValidator, QIcon, QIntValidator
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QStyleFactory,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from scipy.interpolate import interp1d
from scipy.signal import convolve, find_peaks, resample
from scipy.signal.windows import hann
from sklearn.preprocessing import minmax_scale, scale

from miniml.core.event import EventDetection
from miniml.gui.panels import (
    AboutPanel,
    CutPanel,
    FileInfoPanel,
    FilterPanel,
    LoadAbfPanel,
    LoadDatPanel,
    LoadHdfPanel,
    SettingsPanel,
    SummaryPanel,
)
from miniml.gui.util import (
    get_icon_file_path,
    hex_to_rgb,
    load_trace_from_file,
)
from miniml.gui.viewer import EventViewer
from miniml.settings import Settings


class AutoSettingsWindow(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.peak_window = 200  # window in samples to search for peak after steepest rise point # TO DO: replace hard-coded value

        layout = QVBoxLayout(self)

        top_layout = QHBoxLayout()
        self.tracePlot = pg.PlotWidget()
        self.gradient = np.gradient(parent.trace.data)
        self.plotData = self.tracePlot.plot(
            parent.trace.time_axis,
            parent.trace.data,
            pen=pg.mkPen(color=parent.settings.colors[3], width=1),
            clear=True,
        )
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", parent.trace.y_unit)
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
        # add a horizontal line
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
        self.eventPlot.setLabel("left", "Imon", parent.trace.y_unit)
        self.eventPlot.showGrid(x=True, y=True, alpha=0.1)
        self.eventPlot.setTitle("Selected events", size="16pt", bold=True)

        bottom_layout.addWidget(self.eventPlot)

        controls = QFormLayout()
        win_size = parent.settings.event_window
        self.window_size = QLineEdit(str(win_size))
        self.window_size.setMinimumWidth(80)
        self.window_size.setValidator(QIntValidator(1, 10000))
        self.window_size.editingFinished.connect(
            lambda: self.update_window_size(int(self.window_size.text()))
        )
        self.window_size.setStyleSheet("font-weight: bold;")
        self.window_time = QLineEdit(str(win_size * parent.trace.sampling * 1000))
        self.window_time.setMinimumWidth(80)
        self.window_time.setValidator(QDoubleValidator(0.0001, 10000.0, 3))
        self.window_time.editingFinished.connect(
            lambda: self.update_window_size(float(self.window_time.text()))
        )
        self.window_time.setStyleSheet("font-weight: bold;")
        self.filter_factor = QLineEdit(str(parent.settings.filter_factor))
        self.filter_factor.setMinimumWidth(80)
        self.filter_factor.setValidator(QDoubleValidator(1.0, 1000.0, 1))
        self.filter_factor.setStyleSheet("font-weight: bold;")
        self.convolve_window = QLineEdit(str(parent.settings.gradient_convolve_win))
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
        self.gradientPlot.setLabel("left", "Gradient", f"{parent.trace.y_unit}/s")
        self.gradientPlot.showGrid(x=True, y=True, alpha=0.1)
        self.gradientPlot.setTitle("Gradient", size="16pt", bold=True)

        bottom_layout.addWidget(self.gradientPlot)

        layout.addLayout(bottom_layout)

        def custom_accept():
            parent.settings.filter_factor = float(self.filter_factor.text())
            parent.settings.event_window = int(self.window_size.text())
            parent.settings.gradient_convolve_win = int(self.convolve_window.text())
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
        Update the window size using the provided value.
        If the value is an integer, it updates the window size in samples.
        If the value is a float, it updates the window size in milliseconds.
        """
        # check if the value is an integer or float
        if isinstance(new_value, int):
            new_value = int(new_value)
        elif isinstance(new_value, float):
            new_value = int(np.round(new_value / self.parent.trace.sampling / 1000))
        else:
            raise TypeError("new_value must be an integer or float")
        # update the window size in the parent settings
        self.window_size.setText(str(new_value))
        self.window_time.setText(
            str(np.round(new_value * self.parent.trace.sampling * 1000, 5))
        )
        self.draw_window_region(new_value)

    def mouse_clicked(self, mouseClickEvent):
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
        self.eventPlot.clear()
        self.region = None
        self.ev_positions = []

        extract_window = int(
            float(self.time.text()) * 1e-3 / self.parent.trace.sampling
        )
        before = extract_window // 5
        after = extract_window - before

        for item in self.tracePlot.items():
            if isinstance(item, pg.TargetItem):
                # get the x position of the cursor, and the index of the closest point in the trace
                x = item.pos()[0]
                index = np.argmin(np.abs(self.parent.trace.time_axis - x))
                event_location = (
                    np.argmax(np.abs(self.gradient[index - self.peak_window : index]))
                    + index
                    - self.peak_window
                )
                self.eventPlot.plot(
                    self.parent.trace.time_axis[: before + after],
                    self.parent.trace.data[
                        event_location - before : event_location + after
                    ],
                    pen=pg.mkPen(color=(0, 0, 0, 90), width=2),
                )
                self.ev_positions.append(event_location)

        # average events
        if len(self.eventPlot.listDataItems()) > 0:
            event_data = np.zeros((len(self.eventPlot.listDataItems()), before + after))
            for i, item in enumerate(self.eventPlot.listDataItems()):
                event_data[i] = item.getData()[1]
            self.avg_event = np.mean(event_data, axis=0)
            self.eventPlot.plot(
                self.parent.trace.time_axis[: before + after],
                self.avg_event,
                pen=pg.mkPen(color=self.parent.settings.colors[3], width=4),
            )
        else:
            return

        self.draw_window_region(self.parent.settings.event_window)

        # calculate the gradient
        self.gradientPlot.clear()
        win = hann(self.parent.settings.gradient_convolve_win)
        for i, item in enumerate(self.eventPlot.listDataItems()):
            if item.opts["pen"].width() == 4:
                continue
            filtered_ev = self.parent.detection.lowpass_filter(
                data=item.getData()[1],
                cutoff=self.parent.detection.trace.sampling_rate
                / self.parent.settings.filter_factor,
                order=4,
            )
            gradient = convolve(
                np.gradient(filtered_ev, self.parent.trace.sampling), win, mode="same"
            ) / sum(win)
            self.gradientPlot.plot(
                self.parent.trace.time_axis[: before + after],
                gradient,
                pen=pg.mkPen(
                    color=(*hex_to_rgb(self.parent.settings.colors[3]), 100), width=2
                ),
            )

    def draw_window_region(self, win_size):
        peak = np.argmax(np.abs(scale(self.avg_event)))
        win_start = (peak - win_size // 5) * self.parent.trace.sampling
        win_end = (peak + win_size // 1.25) * self.parent.trace.sampling

        def update_positions():
            x1, x2 = self.region.getRegion()
            x1 = int(x1 / self.parent.trace.sampling)
            x2 = int(x2 / self.parent.trace.sampling)
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
        for item in self.tracePlot.items():
            if isinstance(item, pg.TargetItem):
                self.tracePlot.removeItem(item)

    def auto_detect(self):
        """
        Automatically detect events based on the first derivative.
        """
        normalized_trace = scale(self.parent.detection.trace.data)
        self.filtered_trace = self.parent.detection.lowpass_filter(
            data=normalized_trace,
            cutoff=self.parent.detection.trace.sampling_rate / 50,
            order=4,
        )
        self.gradient = np.gradient(self.filtered_trace)
        threshold = np.std(np.abs(self.gradient)) * 20
        peaks, properties = find_peaks(
            np.abs(self.gradient), height=threshold, width=10, distance=500
        )

        if len(peaks) == 0:
            self.parent._warning_box(message="No events detected.")
            return
        elif len(peaks) > 10:
            peaks = peaks[np.argsort(properties["peak_heights"])[-10:]]

        for peak in peaks:
            peak = (
                np.argmax(np.abs(self.filtered_trace[peak : peak + self.peak_window]))
                + peak
            )

            cursor = pg.TargetItem(
                pos=(self.parent.trace.time_axis[peak], self.parent.trace.data[peak]),
                label=f"x={self.parent.trace.time_axis[peak]:.2f}",
                size=12,
                pen=pg.mkPen(color=(255, 202, 58, 255), width=2),
                hoverPen=pg.mkPen(color=(255, 89, 94, 255), width=2),
                brush=pg.mkBrush(color=(255, 202, 58, 100)),
                hoverBrush=pg.mkBrush(color=(255, 89, 94, 100)),
            )
            self.tracePlot.addItem(cursor)

    def auto_window_size(self):
        """
        Automatically detect the window size based on return to baseline.
        """
        if not hasattr(self, "avg_event"):
            self.parent._warning_box(message="Please select events first.")
            return
        t1 = np.argmax(np.abs(scale(self.avg_event)))  # peak_index
        event_avg_copy = np.copy(self.avg_event)

        if event_avg_copy[t1] < np.mean(event_avg_copy[0 : int(t1 // 1.5)]):
            event_avg_copy *= -1

        bsl = np.mean(event_avg_copy[0 : int(t1 // 1.5)])
        t2 = np.argmax(event_avg_copy[t1:] < bsl) + t1
        if t2 == t1:
            t2 = len(event_avg_copy) * 0.75

        # round to next larger multiple of 100
        suggested_window = ((t2 - t1) // 100 + 1) * 100

        model = tf.keras.models.load_model(self.parent.settings.model_path)
        scores, window_sizes = [], []
        for factor in [0.6, 0.8, 1, 1.2, 1.4, 1.6]:
            window_size = (int(suggested_window * factor) // 100) * 100
            window_sizes.append(window_size)

            win_start = t1 - window_size // 5
            win_start = max(win_start, 0)
            win_end = t1 + window_size // 1.25
            win_end = min(win_end, len(event_avg_copy))
            data = event_avg_copy[int(win_start) : int(win_end)]

            # resample to 600 points and normalize, then predict using the model
            scores.append(
                np.squeeze(
                    model(
                        np.expand_dims(
                            minmax_scale(resample(data, 600) * -1), axis=(0, -1)
                        )
                    )
                )
            )

        # find the best score
        best_score = np.argmax(scores)

        self.update_window_size(window_sizes[best_score])

    def auto_filter_settings(self):
        """
        Auto suggest filter settings based on the SNR of the first derivative.
        """
        if not hasattr(self, "ev_positions"):
            self.parent._warning_box(message="Please select events first.")
            return
        test_values = np.arange(5, 50, 5)
        before = int(int(self.window_size.text()) // 5)
        after = int(self.window_size.text()) - before

        SNR_data = []
        for pos in self.ev_positions:
            ev_data = self.parent.trace.data[pos - before : pos + after]
            SNR_data.append(
                np.max(np.abs(scale(ev_data)))
                / np.std(np.abs(scale(ev_data)[0 : before // 2]))
            )
        raw_SNR = np.mean(np.abs(SNR_data))

        result = []
        from itertools import product

        for window, filter_factor in product(test_values, test_values):
            win = hann(window)
            SNR_filtered = []
            for pos in self.ev_positions:
                ev_data = self.parent.trace.data[pos - before : pos + after]
                filtered_ev = self.parent.detection.lowpass_filter(
                    data=ev_data,
                    cutoff=self.parent.trace.sampling_rate / filter_factor,
                    order=4,
                )
                gradient = convolve(
                    np.gradient(filtered_ev, self.parent.trace.sampling),
                    win,
                    mode="same",
                ) / sum(win)
                SNR_filtered.append(
                    np.max(np.abs(gradient)) / np.std(np.abs(gradient[0 : before // 2]))
                )

            result.append(np.mean(np.abs(SNR_filtered)) / raw_SNR)

        result = np.array(result).reshape((len(test_values), len(test_values)))

        target_value = 1.5
        closest_value = np.unravel_index(
            np.abs(result - target_value).argmin(), result.shape
        )

        self.filter_factor.setText(str(test_values[closest_value[1]]))
        self.convolve_window.setText(str(test_values[closest_value[0]]))

        self.gradientPlot.clear()
        win = hann(test_values[closest_value[0]])
        for i, item in enumerate(self.eventPlot.listDataItems()):
            if item.opts["pen"].width() == 4:
                continue
            filtered_ev = self.parent.detection.lowpass_filter(
                data=item.getData()[1],
                cutoff=self.parent.trace.sampling_rate / test_values[closest_value[1]],
                order=4,
            )
            gradient = convolve(
                np.gradient(filtered_ev, self.parent.trace.sampling), win, mode="same"
            ) / sum(win)
            self.gradientPlot.plot(
                self.parent.trace.time_axis[: filtered_ev.shape[0]],
                gradient,
                pen=pg.mkPen(
                    color=(*hex_to_rgb(self.parent.settings.colors[3]), 100), width=2
                ),
            )


class AppMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self._create_toolbar()
        self._connect_actions()
        self._create_menubar()
        self.info_dialog = None
        self.settings = Settings()
        self.was_analyzed = False

    def init_ui(self):
        statusbar = QStatusBar(self)
        statusbar.setSizeGripEnabled(False)
        self.setStatusBar(statusbar)

        self.tracePlot = pg.PlotWidget()
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", "")

        self.predictionPlot = pg.PlotWidget()
        self.predictionPlot.setLabel("left", "Confidence", "")
        self.predictionPlot.setXLink(self.tracePlot)

        self.eventPlot = pg.PlotWidget()
        self.histogramPlot = pg.PlotWidget()
        self.averagePlot = pg.PlotWidget()

        splitter1 = QSplitter(Qt.Orientation.Horizontal)
        splitter1.setHandleWidth(12)
        splitter1.addWidget(self.eventPlot)
        splitter1.addWidget(self.averagePlot)
        splitter1.addWidget(self.histogramPlot)
        splitter1.setSizes([250, 250, 250])

        splitter2 = QSplitter(Qt.Orientation.Vertical)
        splitter2.setHandleWidth(12)
        splitter2.addWidget(self.predictionPlot)
        splitter2.addWidget(self.tracePlot)
        splitter2.addWidget(splitter1)
        splitter2.setSizes([130, 270, 150])

        splitter3 = QSplitter(Qt.Orientation.Horizontal)
        splitter3.setHandleWidth(12)
        splitter3.addWidget(splitter2)

        self.tableWidget = self._create_table()

        splitter3.addWidget(self.tableWidget)
        splitter3.setSizes([750, 400])

        self.setCentralWidget(splitter3)
        QApplication.setStyle(QStyleFactory.create("Cleanlooks"))

        self.setGeometry(100, 100, 1150, 750)
        self.setWindowTitle("miniML")

    def _create_menubar(self):
        menubar = self.menuBar()

        if menubar is None:
            return

        fileMenu = menubar.addMenu("File")
        if fileMenu is not None:
            fileMenu.addAction(self.openAction)
            fileMenu.addAction(self.resetAction)
            fileMenu.addAction(self.saveAction)
            fileMenu.addAction(self.closeAction)

        editMenu = menubar.addMenu("Edit")
        if editMenu is not None:
            editMenu.addAction(self.filterAction)
            editMenu.addAction(self.cutAction)
            editMenu.addAction(self.infoAction)

        viewMenu = menubar.addMenu("View")
        if viewMenu is not None:
            viewMenu.addAction(self.plotAction)
            viewMenu.addAction(self.tableAction)
            viewMenu.addAction(self.predictionAction)
            viewMenu.addAction(self.eventViewerAction)

        runMenu = menubar.addMenu("Run")
        if runMenu is not None:
            runMenu.addAction(self.settingsAction)
            runMenu.addAction(self.analyseAction)
            runMenu.addAction(self.summaryAction)

        helpMenu = menubar.addMenu("Help")
        if helpMenu is not None:
            helpMenu.addAction(self.aboutAction)

    def _get_icon(self, icon_name):
        path = get_icon_file_path(icon_name)
        return QIcon(path)

    def _create_toolbar(self):
        self.tb = self.addToolBar("Menu")
        if self.tb is None:
            return

        self.openAction = QAction(
            self._get_icon("load_file_24px_blue.svg"), "Open...", self
        )
        self.openAction.setShortcut("Ctrl+O")
        self.tb.addAction(self.openAction)
        self.filterAction = QAction(
            self._get_icon("filter_24px_blue.svg"), "Filter", self
        )
        self.filterAction.setShortcut("Ctrl+F")
        self.tb.addAction(self.filterAction)
        self.infoAction = QAction(self._get_icon("info_24px_blue.svg"), "Info", self)
        self.infoAction.setShortcut("Ctrl+I")
        self.tb.addAction(self.infoAction)
        self.cutAction = QAction(
            self._get_icon("content_cut_24px_blue.svg"), "Cut trace", self
        )
        self.cutAction.setShortcut("Ctrl+X")
        self.tb.addAction(self.cutAction)
        self.resetAction = QAction(
            self._get_icon("restore_page_24px_blue.svg"), "Reload", self
        )
        self.resetAction.setShortcut("Ctrl+R")
        self.tb.addAction(self.resetAction)
        self.analyseAction = QAction(
            self._get_icon("rocket_launch_24px_blue.svg"), "Analyse", self
        )
        self.analyseAction.setShortcut("Ctrl+A")
        self.tb.addAction(self.analyseAction)
        self.predictionAction = QAction(
            self._get_icon("ssid_chart_24px_blue.svg"), "Prediction", self
        )
        self.tb.addAction(self.predictionAction)
        self.summaryAction = QAction(
            self._get_icon("functions_24px_blue.svg"), "Summary", self
        )
        self.tb.addAction(self.summaryAction)
        self.plotAction = QAction(
            self._get_icon("insert_chart_24px_blue.svg"), "Plot", self
        )
        self.tb.addAction(self.plotAction)
        self.tableAction = QAction(self._get_icon("table_24px_blue.svg"), "Table", self)
        self.tb.addAction(self.tableAction)
        self.eventViewerAction = QAction(
            self._get_icon("event_mode_24px_blue.svg"), "Event Viewer", self
        )
        self.eventViewerAction.setShortcut("Ctrl+E")
        self.tb.addAction(self.eventViewerAction)
        self.saveAction = QAction(
            self._get_icon("save_24px_blue.svg"), "Save results", self
        )
        self.saveAction.setShortcut("Ctrl+S")
        self.tb.addAction(self.saveAction)
        self.helperAction = QAction(
            self._get_icon("settings_suggest_24px_blue.svg"), "Settings helper", self
        )
        self.helperAction.setShortcut("Ctrl+H")
        self.tb.addAction(self.helperAction)
        self.settingsAction = QAction(
            self._get_icon("settings_24px_blue.svg"), "Settings", self
        )
        self.settingsAction.setShortcut("Ctrl+P")
        self.tb.addAction(self.settingsAction)

        # qActions for MenuBar
        self.closeAction = QAction(
            self._get_icon("cancel_24px_blue.svg"), "Close Window", self
        )
        self.closeAction.setShortcut("Ctrl+W")
        self.aboutAction = QAction(self._get_icon("info_24px_blue.svg"), "About", self)
        self.aboutAction.setShortcut("Ctrl+H")

    def _connect_actions(self):
        self.openAction.triggered.connect(self.new_file)
        self.filterAction.triggered.connect(self.filter_data)
        self.infoAction.triggered.connect(self.info_window)
        self.cutAction.triggered.connect(self.cut_data)
        self.resetAction.triggered.connect(self.reload_data)
        self.analyseAction.triggered.connect(self.run_analysis)
        self.predictionAction.triggered.connect(self.toggle_prediction_win)
        self.summaryAction.triggered.connect(self.summary_window)
        self.plotAction.triggered.connect(self.toggle_plot_win)
        self.tableAction.triggered.connect(self.toggle_table_win)
        self.settingsAction.triggered.connect(self.settings_window)
        self.helperAction.triggered.connect(self.auto_settings_window)
        self.saveAction.triggered.connect(self.save_results)
        self.closeAction.triggered.connect(self.close_gui)
        self.aboutAction.triggered.connect(self.about_win)
        self.eventViewerAction.triggered.connect(self.show_event_viewer)

    def _create_table(self):
        tableWidget = QTableWidget()
        header = tableWidget.verticalHeader()
        if header is not None:
            header.setDefaultSectionSize(10)
        header = tableWidget.horizontalHeader()
        if header is not None:
            header.setDefaultSectionSize(90)
        tableWidget.setRowCount(0)
        tableWidget.setColumnCount(5)
        tableWidget.setHorizontalHeaderLabels(
            ["Position", "Amplitude", "Area", "Risetime", "Decay"]
        )
        viewport = tableWidget.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        tableWidget.setSelectionBehavior(QTableView.SelectRows)
        return tableWidget

    def _warning_box(self, message):
        """
        Display a warning box with a message.

        Parameters
        ----------
        message: str
            The message to be displayed in the warning box.

        Returns
        -------
        None
        """
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle("Message")
        msgbox.setText(message)
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()

    def eventFilter(self, source, event):  # type: ignore
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                index = self.tableWidget.indexAt(event.pos())
                if index.data():
                    selected_ev = index.row()
            elif event.button() == Qt.MouseButton.RightButton:
                index = self.tableWidget.indexAt(event.pos())
                if index.isValid():
                    pass

        return super().eventFilter(source, event)

    def contextMenuEvent(self, event) -> None:
        """
        Create a context menu for the selected event.
        """
        gp = event.globalPos()
        vp_pos = self.tableWidget.viewport().mapFromGlobal(gp)
        row = self.tableWidget.rowAt(vp_pos.y())
        if row >= 0 and self.tableWidget.indexAt(vp_pos).data():
            self.menu = QMenu(self)
            inspectAction = QAction("Inspect event", self)
            inspectAction.triggered.connect(lambda: self.inspect_event(event, row))
            self.menu.addAction(inspectAction)

            deleteAction = QAction("Delete event", self)
            deleteAction.triggered.connect(
                lambda: self.delete_detected_event(event, row)
            )
            self.menu.addAction(deleteAction)

            self.menu.popup(QCursor.pos())

    def inspect_event(self, event, row) -> None:
        """
        Zoom in onto the selected event in main plot window.
        """
        xstart = int(
            self.detection.event_locations[row] - self.detection.window_size / 2
        )
        xend = int(self.detection.event_locations[row] + self.detection.window_size)
        ymin = np.amin(self.detection.trace.data[xstart:xend]) * 1.05
        ymax = np.amax(self.detection.trace.data[xstart:xend]) * 1.05
        self.tracePlot.setXRange(
            xstart * self.detection.trace.sampling, xend * self.detection.trace.sampling
        )
        self.tracePlot.setYRange(ymin, ymax)

    def delete_detected_event(self, event, row) -> None:
        """
        Deletes an event from the detection object.

        Args:
            event (QEvent): The event that triggered the deletion.
            row (int): The index of the event to be deleted.

        Returns:
            None

        This function prompts the user with a confirmation dialog to delete an event.
        After deleting the event, the function updates the main plot, plots the detected events, and tabulates the results.
        """
        answer = QMessageBox.question(
            self,
            "",
            "Do you really want to delete this event?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer == QMessageBox.Yes:
            self.detection.delete_events(event_indices=[row], eval=False)

            self.exclude_events = np.delete(self.exclude_events, row, axis=0)
            self.use_for_avg = np.delete(self.use_for_avg, row, axis=0)
            self.detection.singular_event_indices = np.where(self.use_for_avg == 1)[0]
            self.detection._eval_events()

            self.update_main_plot()

            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()

    def delete_multiple_events(self, rows: list = []) -> None:
        """
        Deletes multiple events from the detection object after exclusion in the Event Viewer.

        Args:
            rows (list): list of the event indices to be deleted.

        Returns:
            None

        This function prompts the user with a confirmation dialog to delete the events.
        After deleting the event, the function updates the main plot, plots the detected events, and tabulates the results.
        """
        if len(rows) > 0:
            answer = QMessageBox.question(
                self,
                "",
                f"Do you really want to delete {len(rows)} event(s)? This can not be reverted",
                QMessageBox.Yes | QMessageBox.No,
            )

            if answer == QMessageBox.Yes:
                self.detection.delete_events(event_indices=rows, eval=False)

                self.exclude_events = np.delete(self.exclude_events, rows, axis=0)
                self.use_for_avg = np.delete(self.use_for_avg, rows, axis=0)

        self.detection.singular_event_indices = np.where(self.use_for_avg == 1)[0]
        if not len(self.detection.singular_event_indices):
            self._warning_box(
                message="All events excluded for average. At least one has to remain, using all detected events instead!"
            )

        if len(self.detection.event_locations) > 0:
            self.detection._eval_events()
            self.update_main_plot()
            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()
            self.num_events = self.detection.event_locations.shape[0]
        else:
            self.num_events = 0
            self._warning_box(message="All detected events were deleted.")

    def filter_data(self) -> None:
        """
        A function that filters data based on the selected filter options in the FilterPanel.
        Otherwise, it applies various filters (detrend, highpass, notch, lowpass) to the trace data based on the user-selected options in the FilterPanel.
        The function then updates the main plot with the filtered data.
        """
        if not hasattr(self, "trace"):
            return

        panel = FilterPanel(self)
        panel.exec_()
        if panel.result() == 0:
            return

        self.update_main_plot()

    def cut_data(self) -> None:
        """
        Display the CutPanel window for slicing the data trace.
        """
        if not hasattr(self, "trace"):
            return
        if self.was_analyzed:
            print("cutting data only possible before analysis")
            return
        cut_panel = CutPanel(self)
        cut_panel.exec_()
        if cut_panel.result() == 0:
            return

        def cut_ends():
            start_x = int(float(cut_panel.start.text()) / self.trace.sampling)
            end_x = int(float(cut_panel.end.text()) / self.trace.sampling)

            self.trace.data = self.trace.data[start_x:end_x]

        def cut_section():
            start_x = int(float(cut_panel.start.text()) / self.trace.sampling)
            end_x = int(float(cut_panel.end.text()) / self.trace.sampling)

            if start_x > 0 or end_x < len(self.trace.data) - 1:
                self.trace.data = np.delete(self.trace.data, np.arange(start_x, end_x))

        if cut_panel.switch.isChecked():
            cut_section()
        else:
            cut_ends()

        self.update_main_plot()

    def update_main_plot(self) -> None:
        """
        Updates the main plot with the data trace.
        """
        self.data_display, self.time_ax_display = self.resample_for_display(
            data=self.trace.data, time_axis=self.trace.time_axis
        )

        pen = pg.mkPen(color=self.settings.colors[3], width=1)
        self.plotData = self.tracePlot.plot(
            self.time_ax_display, self.data_display, pen=pen, clear=True
        )

        self.tracePlot.setLabel("bottom", "Time", "s")
        label1 = "Vmon" if self.recording_mode == "current-clamp" else "Imon"
        self.tracePlot.setLabel("left", label1, self.trace.y_unit)
        if self.was_analyzed and self.detection.event_locations.shape[0] > 0:
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[
                self.detection.event_peak_locations
            ]
            pen = pg.mkPen(None)
            self.plotDetected = self.tracePlot.plot(
                ev_positions,
                ev_peakvalues,
                pen=pen,
                symbol="o",
                symbolSize=8,
                symbolpen=self.settings.colors[0],
                symbolBrush=self.settings.colors[0],
            )

    def toggle_table_win(self) -> None:
        """
        Toggle the display of the event table window.
        """
        if 0 in self.splitter3.sizes():
            self.splitter3.setSizes(self._store_size_c)
        else:
            self._store_size_c = self.splitter3.sizes()
            self.splitter3.setSizes([np.sum(self.splitter3.sizes()), 0])

    def toggle_plot_win(self) -> None:
        """
        Toggle the display of the event plot window.
        """
        sizes = self.splitter2.sizes()
        if sizes[2] == 0:  # panel is hidden
            sizes[0] = 0 if sizes[0] == 0 else self._store_size[0]
            sizes[1] = (
                (np.sum(sizes[0:3]) - self._store_size_b)
                if sizes[0] == 0
                else (self._store_size[1] - self._store_size_b)
            )
            sizes[2] = self._store_size_b
            self._store_size = sizes
        else:  # panel is shown
            self._store_size = sizes
            self._store_size_b = sizes[2]
            sizes[0] = 0 if sizes[0] == 0 else sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[0] == 0 else np.sum(sizes[1:3])
            sizes[2] = 0
        self.splitter2.setSizes(sizes)

    def toggle_prediction_win(self) -> None:
        """
        Toggle the display of the event prediction window.
        """
        sizes = self.splitter2.sizes()
        if sizes[0] == 0:  # panel is hidden
            sizes[0] = self._store_size_a
            sizes[1] = (
                (np.sum(sizes[0:3]) - self._store_size_a)
                if sizes[2] == 0
                else (self._store_size[1] - self._store_size_a)
            )
            sizes[2] = 0 if sizes[2] == 0 else self._store_size[2]
            self._store_size = sizes
        else:  # panel is shown
            self._store_size = sizes
            self._store_size_a = sizes[0]
            sizes[1] = np.sum(sizes[0:3]) if sizes[2] == 0 else np.sum(sizes[0:2])
            sizes[2] = 0 if sizes[2] == 0 else sizes[2]
            sizes[0] = 0
        self.splitter2.setSizes(sizes)

    def reload_data(self) -> None:
        """
        Reload the data from file and reset all windows.
        """
        if not hasattr(self, "filename"):
            return

        msgbox = QMessageBox
        answer = msgbox.question(
            self, "", "Do you want to reload data?", msgbox.Yes | msgbox.No
        )

        if answer == msgbox.Yes:
            self.trace = load_trace_from_file(self.filetype, self.load_args)
            self.was_analyzed = False
            self.detection = EventDetection(self.trace)
            self.update_main_plot()
            self.reset_windows()

    def resample_for_display(self, data, time_axis):
        """
        Data > about 89000000 points crashes pyqtgraph on the hardware it was tested on.
        Resample for display only to prevent crash with large number of datapoints.

        Arguments:
            data: np.array
                the original data
            time_axis: np.array
                the original time axis

        Returns:
            data_display: np.array
                the resampled data
            time_ax_display: np.array
                the resampled time axis
        """
        if data.shape[0] > 89_000_000:
            point_ax = np.arange(0, data.shape[0])
            point_ax_interpol = np.linspace(0, data.shape[0] - 1, 80_000_000)
            f = interp1d(point_ax, data)
            data_display = f(point_ax_interpol)
            time_ax_display = np.linspace(
                time_axis[0], time_axis[-1], data_display.shape[0]
            )
        else:
            data_display = data
            time_ax_display = time_axis
        return data_display, time_ax_display

    def reset_windows(self) -> None:
        """
        Clear all plot and table windows.
        """
        self.tableWidget.setRowCount(0)
        self.eventPlot.clear()
        self.histogramPlot.clear()
        self.averagePlot.clear()
        self.predictionPlot.clear()

    def close_gui(self) -> None:
        """
        Closes the GUI application window.
        """
        self.close()

    def about_win(self) -> None:
        """
        Display the About window.
        """
        about = AboutPanel(self)
        about.exec_()

    def new_file(self) -> None:
        """
        Open a new file via OS dialog and load data from it. Plots the data and initiates a detection object.
        """
        self.filename = QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "HDF, DAT, or ABF files (*.h5 *.hdf *.hdf5 *.dat *.abf)",
        )[0]
        if self.filename == "":
            return

        # HDF file
        if self.filename.endswith("h5"):
            panel = LoadHdfPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return

            self.filetype = "HDF5"
            self.protocol = "none"
            self.load_args = {
                "filename": self.filename,
                "tracename": panel.e1.currentText(),
                "sampling": float(panel.e2.text()),
                "scaling": float(panel.e3.text()),
                "unit": panel.e4.text(),
            }

        # ABF file
        elif self.filename.endswith("abf"):
            panel = LoadAbfPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return

            self.filetype = "AXON ABF"
            self.protocol = panel.protocol.text()
            self.load_args = {
                "filename": self.filename,
                "channel": int(panel.channel.currentText()),
                "scaling": float(panel.scale.text()),
                "unit": panel.unit.text() if (panel.unit.text() != "") else None,
            }

        # DAT file
        elif self.filename.endswith("dat"):
            panel = LoadDatPanel(self)
            panel.exec_()
            if panel.result() == 0:
                return

            self.filetype = "HEKA DAT"
            series_no, rectype = panel.series.currentText().split(" - ")
            self.protocol = rectype
            group_no, _ = panel.group.currentText().split(" - ")
            try:
                series_list = [
                    int(s) - 1 for s in panel.e1.text().replace(",", ";").split(";")
                ]
            except ValueError:
                series_list = []

            load_series = [] if panel.load_option.isChecked() else [int(series_no) - 1]

            self.load_args = {
                "filename": self.filename,
                "rectype": rectype,
                "group": int(group_no) - 1,
                "load_series": load_series,
                "exclude_series": series_list,
                "scaling": float(panel.e2.text()),
                "unit": panel.e3.text() if (panel.e3.text() != "") else None,
            }

        self.trace = load_trace_from_file(self.filetype, self.load_args)
        self.recording_mode = (
            "current-clamp" if "V" in self.trace.y_unit else "voltage-clamp"
        )

        self.was_analyzed = False
        self.detection = EventDetection(self.trace)
        self.update_main_plot()
        self.reset_windows()

    def info_window(self) -> None:
        """
        Display the File Information window.
        """
        if not hasattr(self, "trace"):
            return

        info_win = FileInfoPanel(self)
        info_win.exec_()

    def summary_window(self) -> None:
        """
        Display the analysis summary window.
        """
        if not hasattr(self, "trace"):
            return

        summary_win = SummaryPanel(self)
        summary_win.exec_()

    def settings_window(self) -> None:
        """
        Display the settings window.
        """
        settings_win = SettingsPanel(self)
        settings_win.exec_()
        if settings_win.result() == 0:
            return

        self.settings.event_window = int(settings_win.ev_len.text())
        self.settings.stride = int(settings_win.stride.text())
        self.settings.model_path = str(settings_win.model.currentText())
        self.settings.model_name = str(settings_win.model.currentText())
        self.settings.event_threshold = (
            float(settings_win.thresh.text())
            if settings_win.thresh.hasAcceptableInput()
            else 0.5
        )
        self.settings.minimum_peak_width = (
            int(settings_win.peak_w.text())
            if settings_win.peak_w.hasAcceptableInput()
            else 5
        )
        self.settings.direction = str(settings_win.direction.currentText())
        self.settings.batch_size = int(settings_win.batchsize.text())
        self.settings.filter_factor = int(settings_win.filter_factor.text())
        self.settings.gradient_convolve_win = int(
            settings_win.gradient_convolve_window.text()
        )

    def auto_settings_window(self) -> None:
        """
        Display the settings helper window.
        """
        if not hasattr(self, "trace"):
            return

        settings_helper = AutoSettingsWindow(self)
        settings_helper.exec_()
        if settings_helper.result() == 0:
            return

    def run_analysis(self) -> None:
        """
        Run the miniML analysis on the loaded trace.
        """
        if not hasattr(self, "trace"):
            return

        if self.was_analyzed:
            msgbox = QMessageBox
            answer = msgbox.question(
                self, "", "Do you want to reanalyze this trace?", msgbox.Yes | msgbox.No
            )

            if answer == msgbox.No:
                return

            self.was_analyzed = False

            self.predictionPlot.clear()
            self.eventPlot.clear()
            self.averagePlot.clear()
            self.histogramPlot.clear()
            self.tableWidget.clear()
            self.update_main_plot()

        n_batches = np.ceil(
            (self.trace.data.shape[0] - self.settings.event_window)
            / (self.settings.stride * self.settings.batch_size)
        ).astype(int)
        n_batches = np.floor(n_batches / 5)
        tf.get_logger().setLevel("ERROR")

        with pg.ProgressDialog(
            labelText="Detecting events",
            minimum=0,
            maximum=int(n_batches),
            busyCursor=True,
            cancelText=None,
            wait=0,
        ) as self.dlg:

            def update_progress():
                self.dlg += 1

            class CustomCallback(tf.keras.callbacks.Callback):
                def on_predict_batch_end(self, batch, logs=None):
                    if batch % 5 == 0 and batch > 0:
                        update_progress()

            self.detection = EventDetection(
                data=self.trace,
                model_path=self.settings.model_path,
                model_threshold=self.settings.event_threshold,
                window_size=self.settings.event_window,
                batch_size=self.settings.batch_size,
                event_direction=self.settings.direction,
                verbose=0,
                callbacks=CustomCallback(),
            )

            self.detection.detect_events(
                stride=self.settings.stride,
                eval=True,
                peak_w=self.settings.minimum_peak_width,
                filter_factor=self.settings.filter_factor,
                gradient_convolve_win=self.settings.gradient_convolve_win,
            )

            self.was_analyzed = True
            pen = pg.mkPen(color=self.settings.colors[3], width=1)
            prediction_x = (
                np.arange(0, len(self.detection.prediction)) * self.trace.sampling
            )

            prediction_display, prediction_x_display = self.resample_for_display(
                data=self.detection.prediction, time_axis=prediction_x
            )
            self.predictionPlot.plot(
                prediction_x_display, prediction_display, pen=pen, clear=True
            )

            self.predictionPlot.plot(
                [0, prediction_x[-1]],
                [self.settings.event_threshold, self.settings.event_threshold],
                pen=pg.mkPen(
                    color=self.settings.colors[0], style=Qt.PenStyle.DashLine, width=1
                ),
            )

        if self.detection.event_locations.shape[0] > 0:
            ev_positions = self.detection.event_peak_times
            ev_peakvalues = self.detection.trace.data[
                self.detection.event_peak_locations
            ]
            pen = pg.mkPen(None)
            self.plotDetected = self.tracePlot.plot(
                ev_positions,
                ev_peakvalues,
                pen=pen,
                symbol="o",
                symbolSize=8,
                symbolpen=self.settings.colors[0],
                symbolBrush=self.settings.colors[0],
            )

            self.tabulate_results(tableWidget=self.tableWidget)
            self.plot_events()

            # Set variables needed for event viewer to work.
            self.num_events = self.detection.event_locations.shape[0]
            self.exclude_events = np.zeros(self.num_events)
            self.use_for_avg = np.zeros(self.num_events, dtype=int)
            self.use_for_avg[self.detection.singular_event_indices] = 1

        else:
            self._warning_box(message="No events detected.")

    def show_event_viewer(self) -> None:
        """
        Start the event viewer.
        """
        if not hasattr(self, "detection"):
            return
        if self.was_analyzed and self.num_events > 0:
            event_win = EventViewer(self)
            event_win.exec_()
        else:
            self._warning_box(message="Please load and analyze data first!")

    def save_results(self) -> None:
        """
        Opens a file dialog for saving the results of the event detection analysis.

        Depending on the selected file type, the results are saved using the appropriate
        method from the EventDetection class. Supported file types are CSV, Pickle, and HDF.
        """
        if not hasattr(self, "detection"):
            return
        default_filename = (
            Path(self.filename).with_suffix("") if self.filename else Path("")
        )
        file_types = "CSV (*.csv);;Pickle (*.pickle);;HDF (*.h5 *.hdf *.hdf5)"
        save_filename, selected_filter = QFileDialog.getSaveFileName(
            self, "Save file", str(default_filename), file_types
        )

        if not save_filename:
            return

        if selected_filter == "CSV (*.csv)":
            self.detection.save_to_csv(filename=save_filename)
        elif selected_filter == "Pickle (*.pickle)":
            self.detection.save_to_pickle(filename=save_filename)
        elif selected_filter == "HDF (*.h5 *.hdf *.hdf5)":
            self.detection.save_to_h5(filename=save_filename)

    def plot_events(self):
        """
        Plot events, histogram and average event.
        """
        self.eventPlot.clear()
        self.eventPlot.setTitle("Detected events")
        time_data = (
            np.arange(0, self.detection.events[0].shape[0])
            * self.detection.trace.sampling
        )
        for event in self.detection.events:
            self.eventPlot.plot(
                time_data, event, pen=pg.mkPen(color=self.settings.colors[3], width=1)
            )
        self.eventPlot.setLabel("bottom", "Time", "s")
        self.eventPlot.setLabel("left", "Amplitude", self.detection.trace.y_unit)

        y, x = np.histogram(self.detection.event_stats.amplitudes, bins="auto")
        curve = pg.PlotCurveItem(
            x, y, stepMode="center", fillLevel=0, brush=self.settings.colors[3]
        )
        self.histogramPlot.clear()
        self.histogramPlot.setTitle("Amplitude histogram")
        self.histogramPlot.addItem(curve)
        self.histogramPlot.setLabel("bottom", "Amplitude", self.detection.trace.y_unit)
        self.histogramPlot.setLabel("left", "Count", "")

        ev_average = (
            np.mean(
                self.detection.events[self.detection.singular_event_indices], axis=0
            )
            if len(self.detection.singular_event_indices) > 0
            else np.zeros(self.detection.events[0].shape[0])
        )
        self.averagePlot.clear()
        self.averagePlot.setTitle("Average event waveform")
        time_data = (
            np.arange(0, self.detection.events[0].shape[0])
            * self.detection.trace.sampling
        )
        self.averagePlot.plot(
            time_data, ev_average, pen=pg.mkPen(color=self.settings.colors[2], width=2)
        )
        self.averagePlot.setLabel("bottom", "Time", "s")
        self.averagePlot.setLabel("left", "Amplitude", self.detection.trace.y_unit)

    def tabulate_results(self, tableWidget):
        """
        Populate a QTableWidget with the results of the event detection analysis.
        """
        tableWidget.clear()
        n_events = len(self.detection.event_stats.amplitudes)
        tableWidget.setHorizontalHeaderLabels(
            ["Location", "Amplitude", "Area", "Risetime", "Decay"]
        )
        tableWidget.setRowCount(n_events)
        for i in range(n_events):
            tableWidget.setItem(
                i,
                0,
                QTableWidgetItem(
                    f"{self.detection.event_locations[i] * self.detection.trace.sampling:.5f}"
                ),
            )
            tableWidget.setItem(
                i,
                1,
                QTableWidgetItem(f"{self.detection.event_stats.amplitudes[i]:.5f}"),
            )
            tableWidget.setItem(
                i, 2, QTableWidgetItem(f"{self.detection.event_stats.charges[i]:.5f}")
            )
            tableWidget.setItem(
                i, 3, QTableWidgetItem(f"{self.detection.event_stats.risetimes[i]:.5f}")
            )
            tableWidget.setItem(
                i,
                4,
                QTableWidgetItem(f"{self.detection.event_stats.halfdecays[i]:.5f}"),
            )
        tableWidget.show()
