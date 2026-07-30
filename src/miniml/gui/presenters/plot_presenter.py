import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem


class AnalysisPlotPresenter:
    """
    Static helpers for rendering analysis plots and tables.
    """

    @staticmethod
    def render_event_views(
        *,
        detection,
        settings,
        event_plot,
        histogram_plot,
        average_plot,
    ) -> None:
        """Render detected events, amplitude histogram, and average waveform."""
        event_plot.clear()
        event_plot.setTitle("Detected events")
        time_data = (
            np.arange(0, detection.events[0].shape[0]) * detection.trace.sampling
        )
        for event in detection.events:
            event_plot.plot(
                time_data, event, pen=pg.mkPen(color=settings.colors[3], width=1)
            )
        event_plot.setLabel("bottom", "Time", "s")
        event_plot.setLabel("left", "Amplitude", detection.trace.y_unit)

        y, x = np.histogram(detection.event_stats.amplitudes, bins="auto")
        curve = pg.PlotCurveItem(
            x, y, stepMode="center", fillLevel=0, brush=settings.colors[3]
        )
        histogram_plot.clear()
        histogram_plot.setTitle("Amplitude histogram")
        histogram_plot.addItem(curve)
        histogram_plot.setLabel("bottom", "Amplitude", detection.trace.y_unit)
        histogram_plot.setLabel("left", "Count", "")

        ev_average = (
            np.mean(detection.events[detection.singular_event_indices], axis=0)
            if len(detection.singular_event_indices) > 0
            else np.zeros(detection.events[0].shape[0])
        )
        average_plot.clear()
        average_plot.setTitle("Average event waveform")
        average_plot.plot(
            time_data, ev_average, pen=pg.mkPen(color=settings.colors[2], width=2)
        )
        average_plot.setLabel("bottom", "Time", "s")
        average_plot.setLabel("left", "Amplitude", detection.trace.y_unit)

    @staticmethod
    def populate_results_table(*, detection, table_widget: QTableWidget) -> None:
        """Populate the event statistics table from a detection object."""
        table_widget.clear()
        n_events = len(detection.event_stats.amplitudes)
        table_widget.setHorizontalHeaderLabels(
            ["Location", "Amplitude", "Area", "Risetime", "Decay"]
        )
        table_widget.setRowCount(n_events)
        for i in range(n_events):
            table_widget.setItem(
                i,
                0,
                QTableWidgetItem(
                    f"{detection.event_locations[i] * detection.trace.sampling:.5f}"
                ),
            )
            table_widget.setItem(
                i,
                1,
                QTableWidgetItem(f"{detection.event_stats.amplitudes[i]:.5f}"),
            )
            table_widget.setItem(
                i, 2, QTableWidgetItem(f"{detection.event_stats.charges[i]:.5f}")
            )
            table_widget.setItem(
                i, 3, QTableWidgetItem(f"{detection.event_stats.risetimes[i]:.5f}")
            )
            table_widget.setItem(
                i,
                4,
                QTableWidgetItem(f"{detection.event_stats.halfdecays[i]:.5f}"),
            )
        table_widget.show()
