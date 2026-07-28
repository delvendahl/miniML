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

from miniml.gui.util import get_icon_file_path


class EventViewer(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.resize(750, 610)

        self.detection = parent.detection
        self.settings = parent.settings
        self.num_events = parent.num_events
        self.exclude_events = parent.exclude_events
        self.use_for_avg = parent.use_for_avg

        self.layout = QGridLayout(self)
        self.layout.setColumnMinimumWidth(0, 200)
        self.layout.setColumnMinimumWidth(1, 200)
        self.layout.setColumnMinimumWidth(2, 225)
        self.layout.setRowMinimumHeight(1, 120)
        self.layout.setRowMinimumHeight(2, 160)
        self.layout.setRowMinimumHeight(3, 180)
        self.layout.setRowMinimumHeight(4, 140)

        self.layout.setColumnStretch(0, 1)
        self.layout.setColumnStretch(1, 1)
        self.layout.setColumnStretch(2, 1)
        self.layout.setRowStretch(4, 1)

        self.toolbar = QToolBar()
        self.toolbar.setMovable(False)

        self.layout.addWidget(self.toolbar, 0, 0, 1, 3)

        self.firstAction = QAction(
            QIcon(get_icon_file_path("first_page_24px_blue.svg")),
            "First event",
            self.toolbar,
        )
        self.toolbar.addAction(self.firstAction)
        self.firstAction.triggered.connect(self.first_event)
        self.beforeAction = QAction(
            QIcon(get_icon_file_path("navigate_before_24px_blue.svg")),
            "Previous",
            self.toolbar,
        )
        self.toolbar.addAction(self.beforeAction)
        self.beforeAction.triggered.connect(self.previous)
        self.nextAction = QAction(
            QIcon(get_icon_file_path("navigate_next_24px_blue.svg")),
            "Next",
            self.toolbar,
        )
        self.toolbar.addAction(self.nextAction)
        self.nextAction.triggered.connect(self.next)
        self.toolbar.addSeparator()
        self.deleteAction = QAction(
            QIcon(get_icon_file_path("clear_24px_blue.svg")),
            "Delete event",
            self.toolbar,
        )
        self.toolbar.addAction(self.deleteAction)
        self.deleteAction.triggered.connect(self.delete_event)
        self.excludeAction = QAction(
            QIcon(get_icon_file_path("hide_image_24px_blue.svg")),
            "Exclude from average",
            self.toolbar,
        )
        self.toolbar.addAction(self.excludeAction)
        self.excludeAction.triggered.connect(self.exclude_event)

        self.tracePlot = pg.PlotWidget()
        self.tracePlot.showGrid(x=True, y=True, alpha=0.1)
        self.tracePlot.setLabel("bottom", "Time", "s")
        self.tracePlot.setLabel("left", "Imon", "")
        self.layout.addWidget(self.tracePlot, 1, 0, 1, 3)

        self.eventPlot = pg.PlotWidget()
        self.eventPlot.showGrid(x=True, y=True, alpha=0.1)
        self.eventPlot.setLabel("bottom", "Time", "s")
        self.eventPlot.setLabel("left", "Imon", "")
        self.layout.addWidget(self.eventPlot, 2, 0, 2, 2)

        self.averagePlot = pg.PlotWidget()
        self.layout.addWidget(self.averagePlot, 4, 0, 1, 1)

        self.ampHistPlot = pg.PlotWidget()
        self.layout.addWidget(self.ampHistPlot, 4, 1, 1, 1)

        self.decayHistPlot = pg.PlotWidget()
        self.layout.addWidget(self.decayHistPlot, 4, 2, 1, 1)

        self.ind = 0
        self.left_buffer = int(self.detection.window_size / 2)
        self.right_buffer = int(self.detection.window_size * 1.5)

        self.filtered_data = self.detection.lowpass_filter(
            data=self.detection.trace.data,
            cutoff=self.detection.trace.sampling_rate / self.detection.filter_factor,
            order=4,
        )

        self.trace_x = parent.time_ax_display[::10]
        self.trace_y = parent.data_display[::10]

        self.init_trace_plot()
        self.init_avg_plot()
        self.init_histogram_plots()
        self.update_event_plot()

        self.table = QTableWidget()
        self.table.verticalHeader().setDefaultSectionSize(10)
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
        self.table.viewport().installEventFilter(self)
        self.table.setSelectionBehavior(QTableView.SelectRows)
        self.update_table()

        self.layout.addWidget(self.table, 2, 2, 2, 1)

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.close_event_viewer)
        self.buttonBox.rejected.connect(self.cancel_event_viewer)

        self.layout.addWidget(self.buttonBox, 5, 2, 1, 1)
        self.setWindowTitle("Event Viewer")
        self.setWindowModality(pg.QtCore.Qt.ApplicationModal)

    def update_table(self):
        """
        Updates the table with current event data

        Sets the values of the table according to the current event index
        """
        self.table.setItem(0, 0, QTableWidgetItem(f"{self.ind + 1}"))
        self.table.setItem(
            1,
            0,
            QTableWidgetItem(
                f"{self.detection.event_locations[self.ind] * self.detection.trace.sampling:.5f}"
            ),
        )
        self.table.setItem(
            2, 0, QTableWidgetItem(f"{self.detection.event_scores[self.ind]:.5f}")
        )
        self.table.setItem(
            3, 0, QTableWidgetItem(f"{self.detection.event_bsls[self.ind]:.5f}")
        )
        self.table.setItem(
            4,
            0,
            QTableWidgetItem(f"{self.detection.event_stats.amplitudes[self.ind]:.5f}"),
        )
        self.table.setItem(
            5,
            0,
            QTableWidgetItem(f"{self.detection.event_stats.charges[self.ind]:.5f}"),
        )
        self.table.setItem(
            6,
            0,
            QTableWidgetItem(
                f"{self.detection.event_stats.risetimes[self.ind] * 1e3:.5f}"
            ),
        )
        self.table.setItem(
            7,
            0,
            QTableWidgetItem(
                f"{self.detection.event_stats.slopes[self.ind] * 1e-3:.5f}"
            ),
        )
        self.table.setItem(
            8,
            0,
            QTableWidgetItem(
                f"{self.detection.event_stats.halfdecays[self.ind] * 1e3:.5f}"
            ),
        )
        self.table.setItem(
            9,
            0,
            QTableWidgetItem(
                f"{self.detection.event_stats.halfwidths[self.ind] * 1e3:.5f}"
            ),
        )
        bsl_sd = np.std(
            self.detection.trace.data[
                self.detection.bsl_starts[self.ind]
                - self.detection.event_locations[self.ind]
                - self.left_buffer : self.detection.bsl_ends[self.ind]
                - self.detection.event_locations[self.ind]
                - self.left_buffer
            ]
        )
        self.table.setItem(
            10,
            0,
            QTableWidgetItem(
                f"{np.abs(self.detection.event_stats.amplitudes[self.ind] / bsl_sd):.5f}"
            ),
        )
        self.table.setItem(
            11,
            0,
            QTableWidgetItem(f"{self.detection.interevent_intervals[self.ind]:.5f}"),
        )

        self.table.setItem(0, 1, QTableWidgetItem(""))
        self.table.setItem(1, 1, QTableWidgetItem("s"))
        self.table.setItem(2, 1, QTableWidgetItem(""))
        self.table.setItem(3, 1, QTableWidgetItem(self.detection.trace.y_unit))
        self.table.setItem(4, 1, QTableWidgetItem(self.detection.trace.y_unit))
        self.table.setItem(5, 1, QTableWidgetItem(f"{self.detection.trace.y_unit}*s"))
        self.table.setItem(6, 1, QTableWidgetItem("ms"))
        self.table.setItem(7, 1, QTableWidgetItem(self.detection.trace.y_unit + "/ms"))
        self.table.setItem(8, 1, QTableWidgetItem("ms"))
        self.table.setItem(9, 1, QTableWidgetItem("ms"))
        self.table.setItem(10, 1, QTableWidgetItem(""))
        self.table.setItem(11, 1, QTableWidgetItem("s"))

    def cancel_event_viewer(self):
        self.exclude_events = 0
        self.use_for_avg = 1
        self.close()

    def close_event_viewer(self):
        rows = np.where(self.exclude_events == 1)[0]
        self.parent().delete_multiple_events(rows)
        self.close()

    def init_trace_plot(self):
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
        if np.sum(self.use_for_avg) == 0:
            self.avg.setData(self.avg_time_ax, np.zeros(self.avg_time_ax.shape))
        else:
            self.avg.setData(
                self.avg_time_ax,
                np.mean(self.detection.events[self.use_for_avg == 1], axis=0),
            )

    def update_histogram_plots(self):
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
                pen=pg.mkPen(color="orange", width=2, style=pg.QtCore.Qt.DotLine),
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
                pen = pg.mkPen(style=pg.QtCore.Qt.NoPen)
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
                self.eventPlot, bsl_times, bsl_vals, "r", 2.5, pg.QtCore.Qt.DotLine
            )
            plot_line(
                self.eventPlot,
                [rel_bsl_end, rel_peak_loc],
                bsl_vals,
                "k",
                2.5,
                pg.QtCore.Qt.DotLine,
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
                pg.QtCore.Qt.DotLine,
            )
            plot_line(
                self.eventPlot,
                [rel_max_rise, rel_max_rise],
                [min_value_rise, max_value_rise],
                "magenta",
                2.5,
                pg.QtCore.Qt.DotLine,
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
                pg.QtCore.Qt.DotLine,
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
                    pg.QtCore.Qt.DotLine,
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
        self.ind = 0
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def previous(self):
        self.ind = (self.ind - 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def delete_event(self):
        self.exclude_events[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.use_for_avg[self.ind] = (self.exclude_events[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plots()

    def exclude_event(self):
        self.use_for_avg[self.ind] = (self.use_for_avg[self.ind] + 1) % 2
        self.update_event_plot()
        self.update_avg_plot()
        self.update_histogram_plots()

    def next(self):
        self.ind = (self.ind + 1) % self.num_events
        self.update_event_plot()
        self.update_trace_plot()
        self.update_table()

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_Right:
            self.next()
        elif key == Qt.Key_Left:
            self.previous()
        elif key == Qt.Key_M:
            self.delete_event()
        elif key == Qt.Key_N:
            self.exclude_event()
