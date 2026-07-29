import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QAction,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
)

from miniml.resources.util import get_icon_file_path


class EventViewer(QDialog):
    """
    Interactive dialog for reviewing and curating detected events.
    """

    def __init__(
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
        parent=None,
    ):
        """
        Initialize viewer state and render event-inspection widgets.
        """
        super().__init__(parent)
        self.detection = detection
        self.settings = settings
        self.num_events = num_events
        self.exclude_events = exclude_events
        self.use_for_avg = use_for_avg
        self.on_commit = on_commit

        self.resize(750, 610)

        self.ind = 0
        self.left_buffer = int(self.detection.window_size / 2)
        self.right_buffer = int(self.detection.window_size * 1.5)

        self.filtered_data = self.detection.lowpass_filter(
            data=self.detection.trace.data,
            cutoff=self.detection.trace.sampling_rate / self.detection.filter_factor,
            order=4,
        )

        self.trace_x = time_ax_display[::10]
        self.trace_y = data_display[::10]

        self.init_ui()
        self.init_trace_plot()
        self.init_avg_plot()
        self.init_histogram_plots()
        self.update_event_plot()
        self.update_table()

    def init_ui(self):
        """
        Build the main layout and child UI components.
        """
        self._layout = QGridLayout(self)
        self._layout.setColumnMinimumWidth(0, 200)
        self._layout.setColumnMinimumWidth(1, 200)
        self._layout.setColumnMinimumWidth(2, 225)
        self._layout.setRowMinimumHeight(1, 120)
        self._layout.setRowMinimumHeight(2, 160)
        self._layout.setRowMinimumHeight(3, 180)
        self._layout.setRowMinimumHeight(4, 140)

        self._layout.setColumnStretch(0, 1)
        self._layout.setColumnStretch(1, 1)
        self._layout.setColumnStretch(2, 1)
        self._layout.setRowStretch(4, 1)

        self._create_actions()
        self._create_toolbar()
        self._create_plots()
        self._create_table()
        self._create_button_box()

        self.setWindowTitle("Event Viewer")
        self.setWindowModality(pg.QtCore.Qt.WindowModality.ApplicationModal)
        self.setLayout(self._layout)

    def _create_actions(self):
        """
        Create toolbar actions and connect navigation handlers.
        """

        def _action(icon, text, shortcut=None):
            path = get_icon_file_path(icon)
            action = QAction(QIcon(path), text, self)
            if shortcut:
                action.setShortcut(shortcut)
            return action

        self.firstAction = _action("first_page_24px_blue.svg", "First event")
        self.beforeAction = _action("navigate_before_24px_blue.svg", "Previous")
        self.nextAction = _action("navigate_next_24px_blue.svg", "Next")
        self.deleteAction = _action("clear_24px_blue.svg", "Delete event")
        self.excludeAction = _action("hide_image_24px_blue.svg", "Exclude from average")

        self.firstAction.triggered.connect(self.first_event)
        self.beforeAction.triggered.connect(self.previous)
        self.nextAction.triggered.connect(self.next)
        self.deleteAction.triggered.connect(self.delete_event)
        self.excludeAction.triggered.connect(self.exclude_event)

    def _create_toolbar(self):
        """
        Build the top toolbar for event navigation and curation.
        """
        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)
        self.toolbar.addAction(self.firstAction)
        self.toolbar.addAction(self.beforeAction)
        self.toolbar.addAction(self.nextAction)
        self.toolbar.addSeparator()
        self.toolbar.addAction(self.deleteAction)
        self.toolbar.addAction(self.excludeAction)
        self._layout.addWidget(self.toolbar, 0, 0, 1, 3)

    def _create_plots(self):
        """
        Create and place trace, event, average, and histogram plots.
        """
        self.tracePlot = pg.PlotWidget()
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", "")
        self._layout.addWidget(self.tracePlot, 1, 0, 1, 3)

        self.eventPlot = pg.PlotWidget()
        self.eventPlot.showGrid(x=True, y=True, alpha=0.1)
        self.eventPlot.setLabel("bottom", "Time", "s")
        self.eventPlot.setLabel("left", "Imon", "")
        self._layout.addWidget(self.eventPlot, 2, 0, 2, 2)

        self.averagePlot = pg.PlotWidget()
        self._layout.addWidget(self.averagePlot, 4, 0, 1, 1)

        self.ampHistPlot = pg.PlotWidget()
        self._layout.addWidget(self.ampHistPlot, 4, 1, 1, 1)

        self.decayHistPlot = pg.PlotWidget()
        self._layout.addWidget(self.decayHistPlot, 4, 2, 1, 1)

    def _create_table(self):
        """
        Create and configure the per-event metrics table.
        """
        self.table = QTableWidget()
        header = self.table.verticalHeader()
        if header is not None:
            header.setDefaultSectionSize(10)
        self.table.setRowCount(12)
        self.table.setColumnCount(2)
        self.table.setColumnWidth(0, 85)
        self.table.setColumnWidth(1, 60)
        self.table.setHorizontalHeaderLabels(["Value", "Unit"])
        self.table.setVerticalHeaderLabels(
            [
                "Event",
                "Position",
                "Score",
                "Baseline",
                "Amplitude",
                "Area",
                "Risetime",
                "Slope",
                "Decay",
                "Halfwidth  ",
                "SNR",
                "Interval",
            ]
        )
        viewport = self.table.viewport()
        if viewport is not None:
            viewport.installEventFilter(self)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self._layout.addWidget(self.table, 2, 2, 2, 1)

    def _create_button_box(self):
        """
        Create commit/cancel buttons for the event-review session.
        """
        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.close_event_viewer)
        self.buttonBox.rejected.connect(self.cancel_event_viewer)
        self._layout.addWidget(self.buttonBox, 5, 2, 1, 1)

    def update_table(self):
        """
        Updates the table with current event data

        Sets the values of the table according to the current event index
        """

        bsl_sd = np.std(
            self.detection.trace.data[
                self.detection.bsl_starts[self.ind]
                - self.detection.event_locations[self.ind]
                - self.left_buffer : self.detection.bsl_ends[self.ind]
                - self.detection.event_locations[self.ind]
                - self.left_buffer
            ]
        )

        table_content = [
            (
                self.ind + 1,
                "",
            ),
            (
                f"{self.detection.event_locations[self.ind] * self.detection.trace.sampling:.5f}",
                "s",
            ),
            (
                f"{self.detection.event_scores[self.ind]:.5f}",
                "",
            ),
            (
                f"{self.detection.event_bsls[self.ind]:.5f}",
                self.detection.trace.y_unit,
            ),
            (
                f"{self.detection.event_stats.amplitudes[self.ind]:.5f}",
                self.detection.trace.y_unit,
            ),
            (
                f"{self.detection.event_stats.charges[self.ind]:.5f}",
                self.detection.trace.y_unit + "*s",
            ),
            (
                f"{self.detection.event_stats.risetimes[self.ind] * 1e3:.5f}",
                "ms",
            ),
            (
                f"{self.detection.event_stats.slopes[self.ind] * 1e-3:.5f}",
                self.detection.trace.y_unit + "/ms",
            ),
            (
                f"{self.detection.event_stats.halfdecays[self.ind] * 1e3:.5f}",
                "ms",
            ),
            (
                f"{self.detection.event_stats.halfwidths[self.ind] * 1e3:.5f}",
                "ms",
            ),
            (
                f"{np.abs(self.detection.event_stats.amplitudes[self.ind] / bsl_sd):.5f}",
                "",
            ),
            (
                f"{self.detection.interevent_intervals[self.ind]:.5f}",
                "s",
            ),
        ]

        for i, (value, unit) in enumerate(table_content):
            self.table.setItem(i, 0, QTableWidgetItem(str(value)))
            self.table.setItem(i, 1, QTableWidgetItem(unit))

    def cancel_event_viewer(self):
        """
        Close the viewer without committing selection changes.
        """
        self.close()

    def close_event_viewer(self):
        """
        Commit excluded events and close the viewer.
        """
        rows = np.where(self.exclude_events == 1)[0]
        self.on_commit(rows)
        self.close()

    def init_trace_plot(self):
        """
        Initialize overview trace plot and active-event marker.
        """
        self.tracePlot.clear()
        trace = pg.PlotDataItem(
            self.trace_x,
            self.trace_y,
            pen=pg.mkPen(color=self.settings.colors[3], width=1),
        )
        self.tracePlot.addItem(trace)
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Amplitude", self.detection.trace.y_unit)

        self.update_trace_plot()

    def init_avg_plot(self):
        """
        Initialize the average waveform plot from selected events.
        """
        self.averagePlot.clear()
        self.avg_time_ax = (
            np.arange(0, self.detection.events[0].shape[0])
            * self.detection.trace.sampling
        )
        self.avg = pg.PlotDataItem(
            self.avg_time_ax,
            np.mean(
                self.detection.events[self.detection.singular_event_indices], axis=0
            ),
            pen=pg.mkPen(color=self.settings.colors[2], width=2),
        )
        self.averagePlot.addItem(self.avg)
        self.averagePlot.setLabel("bottom", "Time", "s")
        self.averagePlot.setLabel("left", "Amplitude", self.detection.trace.y_unit)

    def init_histogram_plots(self):
        """
        Initialize amplitude and decay histograms.
        """
        self.ampHistPlot.clear()
        self.decayHistPlot.clear()

        y, x = np.histogram(self.detection.event_stats.amplitudes, bins="auto")
        self.amp_curve = pg.PlotCurveItem(
            x, y, stepMode="center", fillLevel=0, brush=self.settings.colors[3]
        )
        self.ampHistPlot.addItem(self.amp_curve)
        self.ampHistPlot.setLabel("bottom", "Amplitude", self.detection.trace.y_unit)
        self.ampHistPlot.setLabel("left", "Count", "")

        y, x = np.histogram(
            self.detection.event_stats.halfdecays[
                ~np.isnan(self.detection.event_stats.halfdecays)
            ]
            * 1e3,
            bins="auto",
        )
        self.decay_curve = pg.PlotCurveItem(
            x, y, stepMode="center", fillLevel=0, brush=self.settings.colors[3]
        )
        self.decayHistPlot.addItem(self.decay_curve)
        self.decayHistPlot.setLabel("bottom", "Decay time (ms)", "")
        self.decayHistPlot.setLabel("left", "Count", "")

    def update_avg_plot(self):
        """
        Refresh the average waveform after selection changes.
        """
        if np.sum(self.use_for_avg) == 0:
            self.avg.setData(self.avg_time_ax, np.zeros(self.avg_time_ax.shape))
        else:
            self.avg.setData(
                self.avg_time_ax,
                np.mean(self.detection.events[self.use_for_avg == 1], axis=0),
            )

    def update_histogram_plots(self):
        """
        Refresh histogram curves using currently included events.
        """
        if np.sum(self.exclude_events) == self.num_events:
            self.amp_curve.setData([0, 0], [0])
            self.decay_curve.setData([0, 0], [0])
        else:
            y, x = np.histogram(
                self.detection.event_stats.amplitudes[self.exclude_events == 0],
                bins="auto",
            )
            self.amp_curve.setData(x, y)

            values_for_plot = self.detection.event_stats.halfdecays[
                self.exclude_events == 0
            ]
            values_for_plot = values_for_plot[~np.isnan(values_for_plot)]
            y, x = np.histogram(values_for_plot * 1e3, bins="auto")
            self.decay_curve.setData(x, y)

    def update_trace_plot(self):
        """
        Move or create the vertical marker for the active event.
        """
        peak_loc = self.detection.event_peak_locations[self.ind]

        if hasattr(self, "eventitem"):
            self.eventitem.setData(
                [
                    peak_loc * self.detection.trace.sampling,
                    peak_loc * self.detection.trace.sampling,
                ],
                [np.min(self.detection.trace.data), np.max(self.detection.trace.data)],
            )
        else:
            self.eventitem = pg.PlotDataItem(
                [
                    peak_loc * self.detection.trace.sampling,
                    peak_loc * self.detection.trace.sampling,
                ],
                [np.min(self.detection.trace.data), np.max(self.detection.trace.data)],
                pen=pg.mkPen(
                    color="orange", width=2, style=pg.QtCore.Qt.PenStyle.DotLine
                ),
            )
            self.tracePlot.addItem(self.eventitem)

    def update_event_plot(self):
        """
        Updates the event plot.
        """
        event_loc = self.detection.event_locations[self.ind]
        peak_loc = self.detection.event_peak_locations[self.ind]
        peak_val = self.detection.event_peak_values[self.ind]
        bsl = self.detection.event_bsls[self.ind]
        min_value_rise = self.detection.min_values_rise[self.ind]
        max_value_rise = self.detection.max_values_rise[self.ind]

        zero_point = event_loc - self.left_buffer
        sampling_ms = self.detection.trace.sampling * 1e3

        peaks_in_win = self.detection.event_peak_locations[
            np.logical_and(
                self.detection.event_peak_locations > peak_loc,
                self.detection.event_peak_locations < event_loc + self.right_buffer,
            )
        ]

        rel_peak_loc = (peak_loc - zero_point) * sampling_ms
        rel_peak_loc_left = (
            peak_loc - self.detection.peak_spacer - zero_point
        ) * sampling_ms
        rel_peak_loc_right = (
            peak_loc + self.detection.peak_spacer - zero_point
        ) * sampling_ms
        rel_bsl_start = (self.detection.bsl_starts[self.ind] - zero_point) * sampling_ms
        rel_bsl_end = (self.detection.bsl_ends[self.ind] - zero_point) * sampling_ms
        rel_min_rise = (
            self.detection.min_positions_rise[self.ind]
            - (zero_point * self.detection.trace.sampling)
        ) * 1e3
        rel_max_rise = (
            self.detection.max_positions_rise[self.ind]
            - (zero_point * self.detection.trace.sampling)
        ) * 1e3

        if not np.isnan(self.detection.half_decay[self.ind]):
            decay_loc = int(self.detection.half_decay[self.ind])
            rel_decay_loc = (decay_loc - zero_point) * sampling_ms

        if len(peaks_in_win):
            rel_peaks_in_win = (peaks_in_win - zero_point) * sampling_ms

        data = self.detection.trace.data[zero_point : event_loc + self.right_buffer]
        filtered_data = self.filtered_data[zero_point : event_loc + self.right_buffer]
        time_ax = np.arange(0, data.shape[0]) * sampling_ms

        data_plot = self.eventPlot.plot(
            time_ax, data, pen=pg.mkPen(color="gray", width=2.5), clear=True
        )
        data_plot.setAlpha(0.5, False)
        if self.exclude_events[self.ind]:
            event_color = self.settings.colors[0]
        elif self.use_for_avg[self.ind] == 0:
            event_color = self.settings.colors[1]
        else:
            event_color = self.settings.colors[3]
        self.eventPlot.plot(
            time_ax, filtered_data, pen=pg.mkPen(color=event_color, width=2.5)
        )

        if not self.exclude_events[self.ind]:
            bsl_times = [rel_bsl_start, rel_bsl_end]
            bsl_vals = [bsl, bsl]

            def plot_symbols(trace_plot, x, y, color, symbol, size):
                pen = pg.mkPen(None)
                trace_plot.plot(
                    x,
                    y,
                    pen=pen,
                    symbol=symbol,
                    symbolSize=size,
                    symbolpen=color,
                    symbolBrush=color,
                )

            def plot_line(trace_plot, x, y, color, width, style):
                pen = pg.mkPen(color=color, width=width, style=style)
                trace_plot.plot(x, y, pen=pen)

            plot_symbols(self.eventPlot, bsl_times, bsl_vals, "r", "o", 10)
            plot_line(
                self.eventPlot,
                bsl_times,
                bsl_vals,
                "r",
                2.5,
                pg.QtCore.Qt.PenStyle.DotLine,
            )
            plot_line(
                self.eventPlot,
                [rel_bsl_end, rel_peak_loc],
                bsl_vals,
                "k",
                2.5,
                pg.QtCore.Qt.PenStyle.DotLine,
            )

            plot_symbols(
                self.eventPlot,
                [rel_min_rise, rel_max_rise],
                [min_value_rise, max_value_rise],
                "magenta",
                "o",
                10,
            )
            plot_line(
                self.eventPlot,
                [rel_min_rise, rel_max_rise],
                [min_value_rise, min_value_rise],
                "magenta",
                2.5,
                pg.QtCore.Qt.PenStyle.DotLine,
            )
            plot_line(
                self.eventPlot,
                [rel_max_rise, rel_max_rise],
                [min_value_rise, max_value_rise],
                "magenta",
                2.5,
                pg.QtCore.Qt.PenStyle.DotLine,
            )

            plot_symbols(
                self.eventPlot,
                [rel_peak_loc_left, rel_peak_loc_right, rel_peak_loc],
                [peak_val] * 3,
                "orange",
                ["x", "x", "o"],
                [12, 12, 10],
            )
            if len(peaks_in_win):
                plot_symbols(
                    self.eventPlot,
                    rel_peaks_in_win,
                    self.filtered_data[peaks_in_win],
                    "orange",
                    "o",
                    10,
                )
            plot_line(
                self.eventPlot,
                [rel_peak_loc, rel_peak_loc],
                [peak_val, peak_val - self.detection.event_stats.amplitudes[self.ind]],
                "orange",
                2.5,
                pg.QtCore.Qt.PenStyle.DotLine,
            )

            if not np.isnan(self.detection.half_decay[self.ind]):
                plot_symbols(
                    self.eventPlot,
                    [rel_decay_loc],
                    [self.filtered_data[decay_loc]],
                    "green",
                    "o",
                    10,
                )
                plot_line(
                    self.eventPlot,
                    [rel_peak_loc, rel_decay_loc],
                    [self.filtered_data[decay_loc], self.filtered_data[decay_loc]],
                    "green",
                    2.5,
                    pg.QtCore.Qt.PenStyle.DotLine,
                )

        pen = pg.mkPen(color="k", width=1.5)

        color = "green" if self.use_for_avg[self.ind] else "red"
        text_str = f"event #{self.ind + 1}/{self.num_events}: {'used for' if self.use_for_avg[self.ind] else 'excluded from'} average waveform"
        self.text = pg.TextItem(text_str, color=color, border=pen)
        self.eventPlot.addItem(self.text)
        self.text.setPos(0, np.max(data) + (np.max(data) - np.min(data)) / 10)

        self.eventPlot.setLabel("bottom", "Time", "ms")
        self.eventPlot.setLabel("left", "Amplitude", self.detection.trace.y_unit)

    def first_event(self):
        """
        Jump to the first detected event.
        """
        self.ind = 0
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def previous(self):
        """
        Navigate to the previous event, wrapping around.
        """
        self.ind = (self.ind - 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def delete_event(self):
        """
        Toggle deletion state for the active event.
        """
        self.exclude_events[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.use_for_avg[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plots()

    def exclude_event(self):
        """
        Toggle average-inclusion state for the active event.
        """
        self.use_for_avg[self.ind] = (self.use_for_avg[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plots()

    def next(self):
        """
        Navigate to the next event, wrapping around.
        """
        self.ind = (self.ind + 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def keyPressEvent(self, event):  # type: ignore
        """
        Handle keyboard shortcuts for event navigation and curation.
        """
        key = event.key()

        if key == Qt.Key.Key_Right:
            self.next()
        elif key == Qt.Key.Key_Left:
            self.previous()
        elif key == Qt.Key.Key_M:
            self.delete_event()
        elif key == Qt.Key.Key_N:
            self.exclude_event()
