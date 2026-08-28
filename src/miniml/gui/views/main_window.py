from pathlib import Path

import keras
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5.QtCore import QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QStyleFactory,
    QTableView,
    QTableWidget,
)
from scipy.interpolate import interp1d

from miniml.core.event import EventDetection
from miniml.gui.controllers.layout_controller import SplitterLayoutController
from miniml.gui.dialogs import (
    AboutPanel,
    AutoSettingsWindow,
    CutPanel,
    FileInfoPanel,
    FilterPanel,
    LoadAbfPanel,
    LoadDatPanel,
    LoadHdfPanel,
    SettingsPanel,
    SummaryPanel,
)
from miniml.gui.presenters import AnalysisPlotPresenter
from miniml.gui.services import AppServices
from miniml.gui.state import AppState
from miniml.gui.views import EventViewer
from miniml.resources.util import get_icon_file_path


class MainWindow(QMainWindow):
    """
    Primary GUI window coordinating controls, plots, and dialogs.
    """

    openRequested = pyqtSignal(object)
    reloadRequested = pyqtSignal(bool)
    analyzeRequested = pyqtSignal(bool)
    analysisCompleted = pyqtSignal(object)
    saveRequested = pyqtSignal(object)
    filterRequested = pyqtSignal(object)
    cutRequested = pyqtSignal(object)
    settingsRequested = pyqtSignal(object)
    helperRequested = pyqtSignal(object)
    eventViewerRequested = pyqtSignal()
    deleteEventsRequested = pyqtSignal(object)

    def __init__(self, *, state: AppState, services: AppServices):
        """
        Initialize the main window with shared state and services.
        """
        super().__init__()
        self.state = state
        self.services = services
        self.init_ui()

    def init_ui(self):
        """
        Build widgets, layout, and window-level configuration.
        """
        self._create_actions()
        self._create_toolbar()
        self._create_menubar()

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

        self.splitter1 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter1.setHandleWidth(12)
        self.splitter1.addWidget(self.eventPlot)
        self.splitter1.addWidget(self.averagePlot)
        self.splitter1.addWidget(self.histogramPlot)
        self.splitter1.setSizes([250, 250, 250])

        self.splitter2 = QSplitter(Qt.Orientation.Vertical)
        self.splitter2.setHandleWidth(12)
        self.splitter2.addWidget(self.predictionPlot)
        self.splitter2.addWidget(self.tracePlot)
        self.splitter2.addWidget(self.splitter1)
        self.splitter2.setSizes([130, 270, 150])

        self.splitter3 = QSplitter(Qt.Orientation.Horizontal)
        self.splitter3.setHandleWidth(12)
        self.splitter3.addWidget(self.splitter2)

        self.tableWidget = self._create_table()

        self.splitter3.addWidget(self.tableWidget)
        self.splitter3.setSizes([750, 400])

        self.setCentralWidget(self.splitter3)
        QApplication.setStyle(QStyleFactory.create("Cleanlooks"))

        self.layout_controller = SplitterLayoutController(
            splitter2=self.splitter2,
            splitter3=self.splitter3,
        )

        self.setGeometry(100, 100, 1150, 750)
        self.setWindowTitle("miniML")

    def _create_actions(self):
        """
        Create all toolbar and menu actions.
        """

        def _action(icon, text, shortcut=None):
            """
            Create a QAction with icon, label, and optional shortcut.
            """
            path = get_icon_file_path(icon)
            action = QAction(QIcon(path), text, self)
            if shortcut:
                action.setShortcut(shortcut)
            return action

        self.openAction = _action(
            "load_file_24px_blue.svg",
            "Open...",
            "Ctrl+O",
        )
        self.filterAction = _action(
            "filter_24px_blue.svg",
            "Filter",
            "Ctrl+F",
        )
        self.infoAction = _action(
            "info_24px_blue.svg",
            "Info",
            "Ctrl+I",
        )
        self.cutAction = _action(
            "content_cut_24px_blue.svg",
            "Cut trace",
            "Ctrl+X",
        )
        self.resetAction = _action(
            "restore_page_24px_blue.svg",
            "Reload",
            "Ctrl+R",
        )
        self.analyseAction = _action(
            "rocket_launch_24px_blue.svg",
            "Analyse",
            "Ctrl+A",
        )
        self.predictionAction = _action(
            "ssid_chart_24px_blue.svg",
            "Prediction",
            "Ctrl+P",
        )
        self.summaryAction = _action(
            "functions_24px_blue.svg",
            "Summary",
        )
        self.plotAction = _action(
            "insert_chart_24px_blue.svg",
            "Plot",
        )
        self.tableAction = _action(
            "table_24px_blue.svg",
            "Table",
        )
        self.eventViewerAction = _action(
            "event_mode_24px_blue.svg",
            "Event Viewer",
            "Ctrl+E",
        )
        self.saveAction = _action(
            "save_24px_blue.svg",
            "Save results",
            "Ctrl+S",
        )
        self.helperAction = _action(
            "settings_suggest_24px_blue.svg",
            "Settings helper",
            "Ctrl+H",
        )
        self.settingsAction = _action(
            "settings_24px_blue.svg",
            "Settings",
            "Ctrl+P",
        )

        # qActions for MenuBar
        self.closeAction = _action(
            "cancel_24px_blue.svg",
            "Close Window",
            "Ctrl+W",
        )
        self.aboutAction = _action(
            "info_24px_blue.svg",
            "About",
            "Ctrl+H",
        )

    def _create_menubar(self):
        """
        Populate the main menu bar and attach actions.
        """
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

    def _create_toolbar(self):
        """
        Populate the top toolbar with primary actions.
        """
        toolbar = self.addToolBar("Menu")
        if toolbar is None:
            return

        toolbar.addAction(self.openAction)
        toolbar.addAction(self.infoAction)
        toolbar.addAction(self.resetAction)
        toolbar.addSeparator()
        toolbar.addAction(self.filterAction)
        toolbar.addAction(self.cutAction)
        toolbar.addSeparator()
        toolbar.addAction(self.predictionAction)
        toolbar.addAction(self.plotAction)
        toolbar.addAction(self.tableAction)
        toolbar.addSeparator()
        toolbar.addAction(self.analyseAction)
        toolbar.addAction(self.summaryAction)
        toolbar.addAction(self.eventViewerAction)
        toolbar.addAction(self.saveAction)
        toolbar.addSeparator()
        toolbar.addAction(self.helperAction)
        toolbar.addAction(self.settingsAction)

    def _execute_requested_analysis(self) -> None:
        """
        Run analysis and emit the completion signal with results.
        """
        detection = self._run_analysis_with_progress()
        self.analysisCompleted.emit(detection)

    def _open_event_viewer_from_payload(self, payload: object) -> None:
        """
        Open the event viewer from a presenter-provided payload.
        """
        if not isinstance(payload, dict):
            return

        num_events = payload.get("num_events")
        if not isinstance(num_events, int):
            return

        self.open_event_viewer(
            detection=payload.get("detection"),
            settings=payload.get("settings"),
            num_events=num_events,
            exclude_events=payload.get("exclude_events"),
            use_for_avg=payload.get("use_for_avg"),
            time_ax_display=self.time_ax_display,
            data_display=self.data_display,
            on_commit=self.delete_multiple_events,
        )

    def _create_table(self):
        """
        Create and configure the event results table widget.
        """
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
        Display a warning message dialog.
        """
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Warning)
        msgbox.setWindowTitle("Message")
        msgbox.setText(message)
        msgbox.setStandardButtons(QMessageBox.Ok)
        msgbox.exec_()

    # Presenter view API adapters
    def show_warning(self, message: str) -> None:
        """
        Show a warning message requested by the presenter.
        """
        self._warning_box(message)

    def ask_yes_no(self, message: str) -> bool:
        """
        Ask a Yes/No question and return True for Yes.
        """
        answer = QMessageBox.question(
            self,
            "",
            message,
            QMessageBox.Yes | QMessageBox.No,
        )
        return answer == QMessageBox.Yes

    def _clear_analysis_views(self) -> None:
        """
        Clear analysis-related plots and table contents.
        """
        self.predictionPlot.clear()
        self.eventPlot.clear()
        self.averagePlot.clear()
        self.histogramPlot.clear()
        self.tableWidget.clear()

    def _open_data_file_dialog(self) -> str:
        """
        Open a file chooser for supported recording files.
        """
        return QFileDialog.getOpenFileName(
            self,
            "Open file",
            "",
            "HDF, DAT, or ABF files (*.h5 *.hdf *.hdf5 *.dat *.abf)",
        )[0]

    def _collect_hdf_load_context(self, filename: str) -> tuple[str, str, dict] | None:
        """
        Collect HDF-specific loading options from the user.
        """
        panel = LoadHdfPanel(filename=filename, parent=self)
        panel.exec_()
        if panel.result() == 0:
            return None

        return (
            "HDF5",
            "none",
            {
                "filename": filename,
                "tracename": panel.e1.currentText(),
                "sampling": float(panel.e2.text()),
                "scaling": float(panel.e3.text()),
                "unit": panel.e4.text(),
            },
        )

    def _collect_abf_load_context(self, filename: str) -> tuple[str, str, dict] | None:
        """
        Collect ABF-specific loading options from the user.
        """
        panel = LoadAbfPanel(filename=filename, parent=self)
        panel.exec_()
        if panel.result() == 0:
            return None

        return (
            "AXON ABF",
            panel.protocol.text(),
            {
                "filename": filename,
                "channel": int(panel.channel.currentText()),
                "scaling": float(panel.scale.text()),
                "unit": panel.unit.text() if (panel.unit.text() != "") else None,
            },
        )

    def _collect_dat_load_context(self, filename: str) -> tuple[str, str, dict] | None:
        """
        Collect DAT-specific loading options from the user.
        """
        panel = LoadDatPanel(filename=filename, parent=self)
        panel.exec_()
        if panel.result() == 0:
            return None

        series_no, rectype = panel.series.currentText().split(" - ")
        group_no, _ = panel.group.currentText().split(" - ")
        series_list = self._parse_excluded_series(panel.e1.text())
        load_series = [] if panel.load_option.isChecked() else [int(series_no) - 1]

        return (
            "HEKA DAT",
            rectype,
            {
                "filename": filename,
                "rectype": rectype,
                "group": int(group_no) - 1,
                "load_series": load_series,
                "exclude_series": series_list,
                "scaling": float(panel.e2.text()),
                "unit": panel.e3.text() if (panel.e3.text() != "") else None,
            },
        )

    def _parse_excluded_series(self, series_text: str) -> list[int]:
        """
        Parse excluded DAT series indices from text input.
        """
        try:
            return [int(s) - 1 for s in series_text.replace(",", ";").split(";")]
        except ValueError:
            return []

    def _run_analysis_with_progress(self) -> EventDetection:
        """
        Run event detection while updating a progress dialog.
        """
        if self.state.trace is None:
            raise ValueError("Trace is not loaded")

        n_batches = np.ceil(
            (self.state.trace.data.shape[0] - self.state.settings.event_window)
            / (self.state.settings.stride * self.state.settings.batch_size)
        ).astype(int)
        n_batches = int(np.floor(n_batches / 5))
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
                """
                Advance the progress dialog by one step.
                """
                self.dlg += 1

            class CustomCallback(keras.callbacks.Callback):
                def on_predict_batch_end(self, batch, logs=None):
                    """
                    Update progress periodically during prediction.
                    """
                    if batch % 5 == 0 and batch > 0:
                        update_progress()

            return self.services.analysis.run_event_detection(
                trace=self.state.trace,
                settings=self.state.settings,
                callbacks=CustomCallback(),
                verbose=0,
            )

    def _plot_prediction_trace(self) -> None:
        """
        Plot model prediction probabilities and threshold line.
        """
        pen = pg.mkPen(color=self.state.settings.colors[3], width=1)
        prediction_x = (
            np.arange(0, len(self.state.detection.prediction))
            * self.state.trace.sampling
        )

        prediction_display, prediction_x_display = self.resample_for_display(
            data=self.state.detection.prediction, time_axis=prediction_x
        )
        self.predictionPlot.plot(
            prediction_x_display, prediction_display, pen=pen, clear=True
        )

        self.predictionPlot.plot(
            [0, prediction_x[-1]],
            [self.state.settings.event_threshold, self.state.settings.event_threshold],
            pen=pg.mkPen(
                color=self.state.settings.colors[0], style=Qt.PenStyle.DashLine, width=1
            ),
        )

    def _render_detection_results(self) -> None:
        """
        Render detected event markers and dependent analysis views.
        """
        if self.state.detection.event_locations.shape[0] == 0:
            self._warning_box(message="No events detected.")
            return

        ev_positions = self.state.detection.event_peak_times
        ev_peakvalues = self.state.detection.trace.data[
            self.state.detection.event_peak_locations
        ]
        pen = pg.mkPen(None)
        self.plotDetected = self.tracePlot.plot(
            ev_positions,
            ev_peakvalues,
            pen=pen,
            symbol="o",
            symbolSize=8,
            symbolpen=self.state.settings.colors[0],
            symbolBrush=self.state.settings.colors[0],
        )

        self._refresh_analysis_views()
        self.state.init_event_selection_state()

    def _refresh_analysis_views(self) -> None:
        """
        Refresh event plots and statistics table from detection data.
        """
        AnalysisPlotPresenter.populate_results_table(
            detection=self.state.detection,
            table_widget=self.tableWidget,
        )
        AnalysisPlotPresenter.render_event_views(
            detection=self.state.detection,
            settings=self.state.settings,
            event_plot=self.eventPlot,
            histogram_plot=self.histogramPlot,
            average_plot=self.averagePlot,
        )

    def eventFilter(self, source, event):  # type: ignore
        """
        Handle mouse press events on the table viewport.
        """
        if event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton:
                index = self.tableWidget.indexAt(event.pos())
                if index.data():
                    index.row()
            elif event.button() == Qt.MouseButton.RightButton:
                index = self.tableWidget.indexAt(event.pos())
                if index.isValid():
                    pass

        return super().eventFilter(source, event)

    def contextMenuEvent(self, event) -> None:
        """
        Open a context menu for the selected event row.
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
        Zoom the main trace plot to the selected event.
        """
        xstart = int(
            self.state.detection.event_locations[row]
            - self.state.detection.window_size / 2
        )
        xend = int(
            self.state.detection.event_locations[row] + self.state.detection.window_size
        )
        ymin = np.amin(self.state.detection.trace.data[xstart:xend]) * 1.05
        ymax = np.amax(self.state.detection.trace.data[xstart:xend]) * 1.05
        self.tracePlot.setXRange(
            xstart * self.state.detection.trace.sampling,
            xend * self.state.detection.trace.sampling,
        )
        self.tracePlot.setYRange(ymin, ymax)

    def delete_detected_event(self, event, row) -> None:
        """
        Confirm and request deletion of a single detected event.
        """
        answer = QMessageBox.question(
            self,
            "",
            "Do you really want to delete this event?",
            QMessageBox.Yes | QMessageBox.No,
        )

        if answer == QMessageBox.Yes:
            self.deleteEventsRequested.emit([row])

    def delete_multiple_events(self, rows: list | None = None) -> None:
        """
        Confirm and request deletion of multiple detected events.
        """
        rows = [] if rows is None else rows
        if len(rows) > 0:
            answer = QMessageBox.question(
                self,
                "",
                f"Do you really want to delete {len(rows)} event(s)? This can not be reverted",
                QMessageBox.Yes | QMessageBox.No,
            )

            if answer == QMessageBox.Yes:
                self.deleteEventsRequested.emit(rows)
            else:
                return
        else:
            self.deleteEventsRequested.emit([])

    def filter_data(self) -> None:
        """
        Open filter controls and emit filtered trace updates.
        """
        if self.state.trace is None:
            return

        filtered_trace = self.open_filter_panel(
            trace=self.state.trace,
            time_ax_display=self.time_ax_display,
            data_display=self.data_display,
            resample_for_display=self.resample_for_display,
        )
        if filtered_trace is None:
            return

        self.filterRequested.emit(filtered_trace)

    def open_filter_panel(
        self,
        *,
        trace,
        time_ax_display,
        data_display,
        resample_for_display,
    ):
        """
        Show the filter dialog and return filtered trace data.
        """
        panel = FilterPanel(
            trace=trace,
            time_ax_display=time_ax_display,
            data_display=data_display,
            resample_for_display=resample_for_display,
            parent=self,
        )
        panel.exec_()
        if panel.result() == 0:
            return None
        if hasattr(panel, "filtered_trace"):
            return panel.filtered_trace
        return None

    def cut_data(self) -> None:
        """
        Open cut controls and emit cut parameters.
        """
        if self.state.trace is None:
            return

        cut_params = self.open_cut_panel(
            time_ax_display=self.time_ax_display,
            data_display=self.data_display,
            y_unit=self.state.trace.y_unit,
            total_time=self.state.trace.total_time,
        )
        if cut_params is None:
            return

        start_seconds, end_seconds, remove_section = cut_params
        self.cutRequested.emit(
            {
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "remove_section": remove_section,
            }
        )

    def open_cut_panel(
        self,
        *,
        time_ax_display,
        data_display,
        y_unit: str,
        total_time: float,
    ) -> tuple[float, float, bool] | None:
        """
        Show the cut dialog and return selected cut values.
        """
        cut_panel = CutPanel(
            time_ax_display=time_ax_display,
            data_display=data_display,
            y_unit=y_unit,
            total_time=total_time,
            parent=self,
        )
        cut_panel.exec_()
        if cut_panel.result() == 0:
            return None

        return (
            float(cut_panel.start.text()),
            float(cut_panel.end.text()),
            cut_panel.switch.isChecked(),
        )

    def update_main_plot(self) -> None:
        """
        Render the primary trace plot and event markers.
        """
        self.data_display, self.time_ax_display = self.resample_for_display(
            data=self.state.trace.data, time_axis=self.state.trace.time_axis
        )

        pen = pg.mkPen(color=self.state.settings.colors[3], width=1)
        self.plotData = self.tracePlot.plot(
            self.time_ax_display, self.data_display, pen=pen, clear=True
        )

        self.tracePlot.setLabel("bottom", "Time", "s")
        label1 = "Vmon" if self.state.recording_mode == "current-clamp" else "Imon"
        self.tracePlot.setLabel("left", label1, self.state.trace.y_unit)
        if (
            self.state.was_analyzed
            and self.state.detection.event_locations.shape[0] > 0
        ):
            ev_positions = self.state.detection.event_peak_times
            ev_peakvalues = self.state.detection.trace.data[
                self.state.detection.event_peak_locations
            ]
            pen = pg.mkPen(None)
            self.plotDetected = self.tracePlot.plot(
                ev_positions,
                ev_peakvalues,
                pen=pen,
                symbol="o",
                symbolSize=8,
                symbolpen=self.state.settings.colors[0],
                symbolBrush=self.state.settings.colors[0],
            )

    def toggle_table_win(self) -> None:
        """
        Toggle visibility of the event table pane.
        """
        self.layout_controller.toggle_table()

    def toggle_plot_win(self) -> None:
        """
        Toggle visibility of the analysis plot pane.
        """
        self.layout_controller.toggle_plot()

    def toggle_prediction_win(self) -> None:
        """
        Toggle visibility of the prediction pane.
        """
        self.layout_controller.toggle_prediction()

    def reload_data(self) -> None:
        """
        Ask for confirmation and emit a reload request.
        """
        if self.state.filename is None:
            return
        confirmed = self.ask_yes_no("Do you want to reload data?")
        self.reloadRequested.emit(confirmed)

    def resample_for_display(self, data, time_axis):
        """
        Downsample very large traces for safe plotting.
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
        Clear secondary plots and reset table content.
        """
        self.tableWidget.setRowCount(0)
        self.eventPlot.clear()
        self.histogramPlot.clear()
        self.averagePlot.clear()
        self.predictionPlot.clear()

    def close_gui(self) -> None:
        """
        Close the main application window.
        """
        self.close()

    def about_win(self) -> None:
        """
        Open the About dialog.
        """
        about = AboutPanel(self)
        about.exec_()

    def new_file(self) -> None:
        """
        Gather file-loading context and emit an open request.
        """
        filename = self._open_data_file_dialog()
        if not filename:
            return

        lower_name = filename.lower()
        if lower_name.endswith((".h5", ".hdf", ".hdf5")):
            load_context = self._collect_hdf_load_context(filename)
        elif lower_name.endswith(".abf"):
            load_context = self._collect_abf_load_context(filename)
        elif lower_name.endswith(".dat"):
            load_context = self._collect_dat_load_context(filename)
        else:
            self.show_warning("Unsupported file format")
            return

        if load_context is None:
            return

        filetype, protocol, load_args = load_context
        self.openRequested.emit(
            {
                "filename": filename,
                "filetype": filetype,
                "protocol": protocol,
                "load_args": load_args,
            }
        )

    def info_window(self) -> None:
        """
        Open the file information dialog for the current trace.
        """
        trace = self.state.trace
        filetype = self.state.filetype
        recording_mode = self.state.recording_mode
        protocol = self.state.protocol
        if (
            trace is None
            or filetype is None
            or recording_mode is None
            or protocol is None
        ):
            return

        self.open_info_window(
            trace_filename=trace.filename,
            filetype=filetype,
            total_time=trace.total_time,
            y_unit=trace.y_unit,
            recording_mode=recording_mode,
            sampling_rate=trace.sampling_rate,
            protocol=protocol,
        )

    def open_info_window(
        self,
        *,
        trace_filename: str,
        filetype: str,
        total_time: float,
        y_unit: str,
        recording_mode: str,
        sampling_rate: float,
        protocol: str,
    ) -> None:
        """
        Construct and display the file information dialog.
        """
        info_win = FileInfoPanel(
            trace_filename=trace_filename,
            filetype=filetype,
            total_time=total_time,
            y_unit=y_unit,
            recording_mode=recording_mode,
            sampling_rate=sampling_rate,
            protocol=protocol,
            parent=self,
        )
        info_win.exec_()

    def open_summary_window(self, *, trace_filename: str, detection) -> None:
        """
        Construct and display the summary statistics dialog.
        """
        summary_win = SummaryPanel(
            trace_filename=trace_filename,
            detection=detection,
            parent=self,
        )
        summary_win.exec_()

    def summary_window(self) -> None:
        """
        Open the summary dialog for the current detection.
        """
        trace = self.state.trace
        if trace is None:
            return

        self.open_summary_window(
            trace_filename=trace.filename,
            detection=self.state.detection,
        )

    def settings_window(self) -> None:
        """
        Open settings dialog and emit updated settings.
        """
        updated = self.open_settings_panel(settings=self.state.settings)
        if updated is None:
            return

        self.settingsRequested.emit(updated)

    def open_settings_panel(self, *, settings) -> dict | None:
        """
        Show the settings dialog and return selected values.
        """
        settings_win = SettingsPanel(settings=settings, parent=self)
        settings_win.exec_()
        if settings_win.result() == 0:
            return None

        return {
            "event_window": int(settings_win.ev_len.text()),
            "stride": int(settings_win.stride.text()),
            "model_path": str(settings_win.model.currentText()),
            "model_name": str(settings_win.model.currentText()),
            "event_threshold": (
                float(settings_win.thresh.text())
                if settings_win.thresh.hasAcceptableInput()
                else 0.5
            ),
            "minimum_peak_width": (
                int(settings_win.peak_w.text())
                if settings_win.peak_w.hasAcceptableInput()
                else 5
            ),
            "direction": str(settings_win.direction.currentText()),
            "batch_size": int(settings_win.batchsize.text() or 128.0),
            "filter_factor": float(settings_win.filter_factor.text() or 1.0),
            "gradient_convolve_win": int(
                settings_win.gradient_convolve_window.text() or 0.0
            ),
        }

    def _apply_auto_settings(
        self, *, filter_factor: float, event_window: int, gradient_convolve_win: int
    ) -> None:
        """
        Emit helper-derived settings updates.
        """
        self.helperRequested.emit(
            {
                "filter_factor": filter_factor,
                "event_window": event_window,
                "gradient_convolve_win": gradient_convolve_win,
            }
        )

    def auto_settings_window(self) -> None:
        """
        Open the automatic settings helper dialog.
        """
        if self.state.trace is None:
            return

        self.open_auto_settings_window(
            trace=self.state.trace,
            settings=self.state.settings,
            detection=self.state.detection,
            on_commit=self._apply_auto_settings,
            on_warning=self.show_warning,
        )

    def open_auto_settings_window(
        self,
        *,
        trace,
        settings,
        detection,
        on_commit,
        on_warning,
    ) -> None:
        """
        Construct and display the automatic settings dialog.
        """
        settings_helper = AutoSettingsWindow(
            trace=trace,
            settings=settings,
            detection=detection,
            on_commit=on_commit,
            on_warning=on_warning,
            parent=self,
        )
        settings_helper.exec_()

    def run_analysis(self) -> None:
        """
        Ask for reanalysis confirmation and emit analyze request.
        """
        confirmed = True
        if self.state.was_analyzed:
            confirmed = self.ask_yes_no("Do you want to reanalyze this trace?")
        self.analyzeRequested.emit(confirmed)

    def open_event_viewer(
        self,
        *,
        detection,
        settings,
        num_events: int,
        exclude_events,
        use_for_avg,
        time_ax_display,
        data_display,
        on_commit,
    ) -> None:
        """
        Construct and display the event viewer dialog.
        """
        event_win = EventViewer(
            detection=detection,
            settings=settings,
            num_events=num_events,
            exclude_events=exclude_events,
            use_for_avg=use_for_avg,
            time_ax_display=time_ax_display,
            data_display=data_display,
            on_commit=on_commit,
            parent=self,
        )
        event_win.exec_()

    def save_results(self) -> None:
        """
        Open save dialog and emit selected output options.
        """
        if self.state.detection is None:
            return

        default_filename = (
            Path(self.state.filename).with_suffix("")
            if self.state.filename
            else Path("")
        )
        file_types = "CSV (*.csv);;HDF (*.h5 *.hdf *.hdf5)"
        save_filename, selected_filter = self.open_save_results_dialog(
            default_filename=str(default_filename),
            file_types=file_types,
        )
        self.saveRequested.emit(
            {
                "save_filename": save_filename,
                "selected_filter": selected_filter,
            }
        )

    def open_save_results_dialog(
        self, *, default_filename: str, file_types: str
    ) -> tuple[str, str]:
        """
        Show save dialog and return filename and selected filter.
        """
        return QFileDialog.getSaveFileName(
            self,
            "Save file",
            default_filename,
            file_types,
        )
