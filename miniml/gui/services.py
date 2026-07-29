from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from miniml.core.event import EventDetection
from miniml.core.trace import MiniTrace
from miniml.gui.util import load_trace_from_file
from miniml.settings import Settings


class TraceService:
    """Trace loading and trace-related utility operations."""

    def load_trace(self, filetype: str, load_args: dict) -> MiniTrace:
        """
        Load trace data from disk using the selected file adapter.
        """
        return load_trace_from_file(filetype, load_args)

    def infer_recording_mode(self, trace: MiniTrace) -> str:
        """
        Infer recording mode from the trace unit label.
        """
        return "current-clamp" if "V" in trace.y_unit else "voltage-clamp"


class AnalysisService:
    """Event detection use-cases."""

    def create_detection(self, trace: MiniTrace) -> EventDetection:
        """
        Create a fresh detection object for the given trace.
        """
        return EventDetection(trace)

    def run_event_detection(
        self,
        *,
        trace: MiniTrace,
        settings: Settings,
        callbacks=None,
        verbose: int = 0,
    ) -> EventDetection:
        """
        Run event detection using the current analysis settings.
        """
        detection = EventDetection(
            data=trace,
            model_path=settings.model_path,
            model_threshold=settings.event_threshold,
            window_size=int(settings.event_window),
            batch_size=settings.batch_size,
            event_direction=settings.direction,
            verbose=verbose,
            callbacks=callbacks,
        )

        detection.detect_events(
            stride=settings.stride,
            eval=True,
            peak_w=settings.minimum_peak_width,
            filter_factor=settings.filter_factor,
            gradient_convolve_win=settings.gradient_convolve_win,
        )

        return detection


class ResultsService:
    """Save analysis output in the selected format."""

    def save_detection(
        self, detection: EventDetection, filename: str, selected_filter: str
    ) -> None:
        """
        Persist detection results in the user-selected file format.
        """
        if selected_filter == "CSV (*.csv)":
            detection.save_to_csv(filename=filename)
        elif selected_filter == "Pickle (*.pickle)":
            detection.save_to_pickle(filename=filename)
        elif selected_filter == "HDF (*.h5 *.hdf *.hdf5)":
            detection.save_to_h5(filename=filename)


@dataclass
class EventSelectionResult:
    """
    Result values produced after applying event-selection edits.
    """

    exclude_events: np.ndarray
    use_for_avg: np.ndarray
    has_remaining_events: bool
    has_average_events: bool
    event_count: int


class EventSelectionService:
    """Mutations for deleted/excluded event selections."""

    def apply_event_deletions(
        self,
        *,
        detection: EventDetection,
        exclude_events: np.ndarray,
        use_for_avg: np.ndarray,
        rows: list[int],
    ) -> EventSelectionResult:
        """
        Apply deletions and recompute selection and evaluation state.
        """
        if rows:
            detection.delete_events(event_indices=rows, eval=False)
            exclude_events = np.delete(exclude_events, rows, axis=0)
            use_for_avg = np.delete(use_for_avg, rows, axis=0)

        detection.singular_event_indices = np.where(use_for_avg == 1)[0]
        has_average_events = len(detection.singular_event_indices) > 0
        has_remaining_events = len(detection.event_locations) > 0

        if has_remaining_events:
            detection._eval_events()

        return EventSelectionResult(
            exclude_events=exclude_events,
            use_for_avg=use_for_avg,
            has_remaining_events=has_remaining_events,
            has_average_events=has_average_events,
            event_count=len(detection.event_locations),
        )


@dataclass
class AppServices:
    """
    Aggregated service container used by the GUI presenter.
    """

    trace: TraceService = field(default_factory=TraceService)
    analysis: AnalysisService = field(default_factory=AnalysisService)
    results: ResultsService = field(default_factory=ResultsService)
    event_selection: EventSelectionService = field(
        default_factory=EventSelectionService
    )
