from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from miniml.core.event import EventDetection
from miniml.core.trace import MiniTrace
from miniml.gui.services import AppServices
from miniml.gui.state import AppState


class MainWindowPresenter(QObject):
    """
    Coordinates main-window use-cases between state/services and Qt signals.
    """

    showWarning = pyqtSignal(str)

    requestMainPlotUpdate = pyqtSignal()
    requestResetWindows = pyqtSignal()
    requestClearAnalysisViews = pyqtSignal()
    requestPredictionPlot = pyqtSignal()
    requestRenderDetectionResults = pyqtSignal()
    requestRefreshAnalysisViews = pyqtSignal()
    requestRunAnalysisWithProgress = pyqtSignal()

    requestOpenEventViewer = pyqtSignal(object)

    def __init__(
        self, *, state: AppState, services: AppServices, parent: QObject | None = None
    ):
        """
        Initialize presenter dependencies and internal view reference.
        """
        super().__init__(parent)
        self.state = state
        self.services = services
        self.view: object | None = None

    def bind_view(self, view) -> None:
        """
        Wire user-intent signals and presenter-effect signals for the given view.
        """
        self.view = view

        # Toolbar/menu actions -> view intent preparation
        view.openAction.triggered.connect(view.new_file)
        view.filterAction.triggered.connect(view.filter_data)
        view.infoAction.triggered.connect(view.info_window)
        view.cutAction.triggered.connect(view.cut_data)
        view.resetAction.triggered.connect(view.reload_data)
        view.analyseAction.triggered.connect(view.run_analysis)
        view.predictionAction.triggered.connect(view.toggle_prediction_win)
        view.summaryAction.triggered.connect(view.summary_window)
        view.plotAction.triggered.connect(view.toggle_plot_win)
        view.tableAction.triggered.connect(view.toggle_table_win)
        view.settingsAction.triggered.connect(view.settings_window)
        view.helperAction.triggered.connect(view.auto_settings_window)
        view.saveAction.triggered.connect(view.save_results)
        view.closeAction.triggered.connect(view.close_gui)
        view.aboutAction.triggered.connect(view.about_win)
        view.eventViewerAction.triggered.connect(
            lambda _: view.eventViewerRequested.emit()
        )

        # View intents -> presenter slots
        view.openRequested.connect(self.on_open_requested)
        view.reloadRequested.connect(self.on_reload_requested)
        view.analyzeRequested.connect(self.on_analyze_requested)
        view.analysisCompleted.connect(self.on_analysis_completed)
        view.saveRequested.connect(self.on_save_requested)
        view.filterRequested.connect(self.on_filter_requested)
        view.cutRequested.connect(self.on_cut_requested)
        view.settingsRequested.connect(self.on_settings_requested)
        view.helperRequested.connect(self.on_helper_requested)
        view.eventViewerRequested.connect(self.on_event_viewer_requested)
        view.deleteEventsRequested.connect(self.on_delete_events_requested)

        # Presenter effect signals -> view adapters
        self.showWarning.connect(view.show_warning)
        self.requestMainPlotUpdate.connect(view.update_main_plot)
        self.requestResetWindows.connect(view.reset_windows)
        self.requestClearAnalysisViews.connect(view._clear_analysis_views)
        self.requestRunAnalysisWithProgress.connect(view._execute_requested_analysis)
        self.requestPredictionPlot.connect(view._plot_prediction_trace)
        self.requestRenderDetectionResults.connect(view._render_detection_results)
        self.requestRefreshAnalysisViews.connect(view._refresh_analysis_views)
        self.requestOpenEventViewer.connect(view._open_event_viewer_from_payload)

    def has_trace(self) -> bool:
        """
        Return whether a trace is currently loaded.
        """
        return self.state.trace is not None

    def has_detection(self) -> bool:
        """
        Return whether a detection object is currently available.
        """
        return self.state.detection is not None

    def apply_settings(self, **values) -> None:
        """
        Apply user-provided settings values to application state.
        """
        self.state.update_settings(**values)

    def apply_auto_settings(
        self, *, filter_factor: float, event_window: int, gradient_convolve_win: int
    ) -> None:
        """
        Apply helper-derived settings values to application state.
        """
        self.state.update_settings(
            filter_factor=filter_factor,
            event_window=event_window,
            gradient_convolve_win=gradient_convolve_win,
        )

    def apply_filtered_trace(self, *, filtered_trace: MiniTrace) -> None:
        """
        Replace current trace data with a filtered trace.
        """
        self.state.replace_trace(trace=filtered_trace)

    def apply_cut(self, *, start_x: int, end_x: int, remove_section: bool) -> None:
        """
        Apply a cut operation to the current trace samples.
        """
        if self.state.trace is None:
            raise ValueError("Trace is not loaded")

        if remove_section:
            if start_x > 0 or end_x < len(self.state.trace.data) - 1:
                self.state.trace.data = np.delete(
                    self.state.trace.data, np.arange(start_x, end_x)
                )
            return

        self.state.trace.data = self.state.trace.data[start_x:end_x]

    def save_results(self, *, save_filename: str, selected_filter: str) -> bool:
        """
        Save detection results if output parameters are valid.
        """
        if self.state.detection is None or not save_filename:
            return False

        self.services.results.save_detection(
            detection=self.state.detection,
            filename=save_filename,
            selected_filter=selected_filter,
        )
        return True

    @pyqtSlot(object)
    def on_open_requested(self, load_request: object) -> None:
        """
        Handle file-open requests and initialize trace context.
        """
        if not isinstance(load_request, dict):
            return

        filename = load_request.get("filename")
        filetype = load_request.get("filetype")
        protocol = load_request.get("protocol")
        load_args = load_request.get("load_args")

        if (
            not isinstance(filename, str)
            or not isinstance(filetype, str)
            or not isinstance(protocol, str)
            or not isinstance(load_args, dict)
        ):
            return

        loaded_trace = self.services.trace.load_trace(filetype, load_args)
        recording_mode = self.services.trace.infer_recording_mode(loaded_trace)
        self.state.set_trace_context(
            trace=loaded_trace,
            filename=filename,
            filetype=filetype,
            protocol=protocol,
            load_args=load_args,
            recording_mode=recording_mode,
        )

        self.requestMainPlotUpdate.emit()
        self.requestResetWindows.emit()

    @pyqtSlot(bool)
    def on_reload_requested(self, confirmed: bool) -> None:
        """
        Reload the current file and reset analysis state if confirmed.
        """
        if not confirmed:
            return

        if (
            self.state.filename is None
            or self.state.filetype is None
            or self.state.load_args is None
        ):
            return

        trace = self.services.trace.load_trace(
            self.state.filetype, self.state.load_args
        )
        detection = self.services.analysis.create_detection(trace)
        self.state.replace_trace(trace=trace, detection=detection)

        self.requestMainPlotUpdate.emit()
        self.requestResetWindows.emit()

    @pyqtSlot(bool)
    def on_analyze_requested(self, confirmed_reanalysis: bool) -> None:
        """
        Start analysis flow, handling optional reanalysis confirmation.
        """
        if not self.has_trace():
            return

        if self.state.was_analyzed and not confirmed_reanalysis:
            return

        if self.state.was_analyzed:
            self.state.mark_analyzed(False)
            self.requestClearAnalysisViews.emit()
            self.requestMainPlotUpdate.emit()

        self.requestRunAnalysisWithProgress.emit()

    @pyqtSlot(object)
    def on_analysis_completed(self, detection_obj: object) -> None:
        """
        Store completed detection results and trigger view rendering.
        """
        if not isinstance(detection_obj, EventDetection):
            return

        self.state.detection = detection_obj
        self.state.mark_analyzed(True)
        self.requestPredictionPlot.emit()
        self.requestRenderDetectionResults.emit()

    @pyqtSlot(object)
    def on_save_requested(self, save_request: object) -> None:
        """
        Validate and process save-result requests from the view.
        """
        if not isinstance(save_request, dict):
            return

        save_filename = save_request.get("save_filename")
        selected_filter = save_request.get("selected_filter")
        if not isinstance(save_filename, str) or not isinstance(selected_filter, str):
            return

        self.save_results(
            save_filename=save_filename,
            selected_filter=selected_filter,
        )

    @pyqtSlot(object)
    def on_filter_requested(self, filtered_trace_obj: object) -> None:
        """
        Apply filtered trace data and refresh the main plot.
        """
        if not isinstance(filtered_trace_obj, MiniTrace):
            return

        self.apply_filtered_trace(filtered_trace=filtered_trace_obj)
        self.requestMainPlotUpdate.emit()

    @pyqtSlot(object)
    def on_cut_requested(self, cut_request: object) -> None:
        """
        Validate and apply trace cut parameters from the view.
        """
        if self.state.trace is None:
            return

        if self.state.was_analyzed:
            self.showWarning.emit("Cutting data is only possible before analysis")
            return

        if not isinstance(cut_request, dict):
            return

        start_raw = cut_request.get("start_seconds")
        end_raw = cut_request.get("end_seconds")
        remove_raw = cut_request.get("remove_section")
        if (
            not isinstance(start_raw, (int, float))
            or not isinstance(end_raw, (int, float))
            or not isinstance(remove_raw, bool)
        ):
            return

        start_seconds = float(start_raw)
        end_seconds = float(end_raw)
        remove_section = remove_raw

        start_x = int(start_seconds / self.state.trace.sampling)
        end_x = int(end_seconds / self.state.trace.sampling)
        self.apply_cut(
            start_x=start_x,
            end_x=end_x,
            remove_section=remove_section,
        )
        self.requestMainPlotUpdate.emit()

    @pyqtSlot(object)
    def on_settings_requested(self, settings_values: object) -> None:
        """
        Normalize and apply settings updates from the settings dialog.
        """
        if not isinstance(settings_values, dict):
            return

        normalized: dict[str, Any] = {}
        for key, value in settings_values.items():
            if isinstance(key, str):
                normalized[key] = value
        self.apply_settings(**normalized)

    @pyqtSlot(object)
    def on_helper_requested(self, auto_settings_values: object) -> None:
        """
        Validate and apply settings from the helper dialog.
        """
        if not isinstance(auto_settings_values, dict):
            return

        filter_raw = auto_settings_values.get("filter_factor")
        event_raw = auto_settings_values.get("event_window")
        gradient_raw = auto_settings_values.get("gradient_convolve_win")
        if (
            not isinstance(filter_raw, (int, float))
            or not isinstance(event_raw, int)
            or not isinstance(gradient_raw, int)
        ):
            return

        self.apply_auto_settings(
            filter_factor=float(filter_raw),
            event_window=event_raw,
            gradient_convolve_win=gradient_raw,
        )

    @pyqtSlot()
    def on_event_viewer_requested(self) -> None:
        """
        Open the event viewer when analyzed event state is available.
        """
        if self.state.detection is None:
            self.showWarning.emit("Please load and analyze data first!")
            return

        if not (self.state.was_analyzed and self.state.num_events > 0):
            self.showWarning.emit("Please load and analyze data first!")
            return

        if self.state.exclude_events is None or self.state.use_for_avg is None:
            self.showWarning.emit("Please load and analyze data first!")
            return

        self.requestOpenEventViewer.emit(
            {
                "detection": self.state.detection,
                "settings": self.state.settings,
                "num_events": self.state.num_events,
                "exclude_events": np.copy(self.state.exclude_events),
                "use_for_avg": np.copy(self.state.use_for_avg),
            }
        )

    @pyqtSlot(object)
    def on_delete_events_requested(self, rows_obj: Sequence[int] | object) -> None:
        """
        Apply event deletions and update dependent analysis views.
        """
        rows: list[int] = []
        if isinstance(rows_obj, np.ndarray) and rows_obj.ndim == 1:
            rows = rows_obj.tolist()
        elif isinstance(rows_obj, Sequence) and not isinstance(rows_obj, (str, bytes)):
            rows = [int(row) for row in rows_obj]

        if not rows:
            return

        if (
            self.state.detection is None
            or self.state.exclude_events is None
            or self.state.use_for_avg is None
        ):
            return

        result = self.services.event_selection.apply_event_deletions(
            detection=self.state.detection,
            exclude_events=self.state.exclude_events,
            use_for_avg=self.state.use_for_avg,
            rows=rows,
        )
        self.state.apply_event_selection(
            exclude_events=result.exclude_events,
            use_for_avg=result.use_for_avg,
        )

        if not result.has_average_events:
            self.showWarning.emit(
                "All events excluded for average. At least one has to remain, using all detected events instead!"
            )

        if result.has_remaining_events:
            self.requestMainPlotUpdate.emit()
            self.requestRefreshAnalysisViews.emit()
            self.state.apply_event_selection(
                exclude_events=self.state.exclude_events,
                use_for_avg=self.state.use_for_avg,
                num_events=result.event_count,
            )
            return

        self.state.apply_event_selection(
            exclude_events=self.state.exclude_events,
            use_for_avg=self.state.use_for_avg,
            num_events=0,
        )
        self.showWarning.emit("All detected events were deleted.")
