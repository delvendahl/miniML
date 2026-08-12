from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from miniml.core.detection import EventDetection
from miniml.core.trace import MiniTrace
from miniml.settings import Settings


@dataclass
class AppState:
    """
    Container for mutable GUI application state.

    Attributes
    ----------
    settings : Settings
        Active analysis settings.
    trace : MiniTrace | None
        Currently loaded trace.
    filtered_trace : MiniTrace | None
        Filtered version of the loaded trace, when available.
    detection : EventDetection | None
        Detection state associated with the current trace.
    filename : str | None
        Path or display name of the loaded file.
    filetype : str | None
        File format identifier for the loaded trace.
    protocol : str | None
        Recording protocol metadata.
    load_args : dict | None
        Loader arguments used to open the trace.
    recording_mode : str | None
        Inferred or selected recording mode.
    was_analyzed : bool
        Whether the current trace has been analyzed.
    num_events : int
        Number of currently tracked events.
    exclude_events : np.ndarray | None
        Event exclusion mask used by the GUI.
    use_for_avg : np.ndarray | None
        Event-selection mask used for averaging.
    """

    settings: Settings = field(default_factory=Settings)
    trace: MiniTrace | None = None
    filtered_trace: MiniTrace | None = None
    detection: EventDetection | None = None

    filename: str | None = None
    filetype: str | None = None
    protocol: str | None = None
    load_args: dict | None = None
    recording_mode: str | None = None

    was_analyzed: bool = False
    num_events: int = 0
    exclude_events: np.ndarray | None = None
    use_for_avg: np.ndarray | None = None

    def clear_analysis(self) -> None:
        """
        Reset analysis results while preserving the currently loaded trace.
        """
        self.was_analyzed = False
        self.detection = EventDetection(self.trace) if self.trace is not None else None
        self.num_events = 0
        self.exclude_events = None
        self.use_for_avg = None

    def set_trace_context(
        self,
        *,
        trace: MiniTrace,
        filename: str,
        filetype: str,
        protocol: str,
        load_args: dict,
        recording_mode: str,
    ) -> None:
        """
        Set all metadata tied to a loaded trace and clear previous analysis.

        Parameters
        ----------
        trace : MiniTrace
            Loaded trace object.
        filename : str
            Path or display name of the loaded file.
        filetype : str
            File format identifier for the loaded trace.
        protocol : str
            Recording protocol metadata.
        load_args : dict
            Loader arguments used to open the trace.
        recording_mode : str
            Inferred or selected recording mode.
        """
        self.trace = trace
        self.filtered_trace = None
        self.filename = filename
        self.filetype = filetype
        self.protocol = protocol
        self.load_args = load_args
        self.recording_mode = recording_mode
        self.clear_analysis()

    def init_event_selection_state(self) -> None:
        """
        Initialize event-selection arrays used by the event viewer.

        Raises
        ------
        ValueError
            If no detection state is available.
        """
        if self.detection is None:
            raise ValueError("Detection state is not available")

        self.num_events = int(self.detection.event_locations.shape[0])
        self.exclude_events = np.zeros(self.num_events)
        self.use_for_avg = np.zeros(self.num_events, dtype=int)
        self.use_for_avg[self.detection.singular_event_indices] = 1

    def update_settings(self, **values) -> None:
        """
        Apply multiple settings mutations in one operation.

        Parameters
        ----------
        **values
            Mapping of setting names to replacement values.
        """
        for key, value in values.items():
            setattr(self.settings, key, value)

    def mark_analyzed(self, analyzed: bool) -> None:
        """
        Set whether the current trace has completed analysis.

        Parameters
        ----------
        analyzed : bool
            Analysis completion state for the current trace.
        """
        self.was_analyzed = analyzed

    def apply_event_selection(
        self,
        *,
        exclude_events: np.ndarray,
        use_for_avg: np.ndarray,
        num_events: int | None = None,
    ) -> None:
        """
        Store the current event-selection arrays and optional event count.

        Parameters
        ----------
        exclude_events : np.ndarray
            Updated event exclusion mask.
        use_for_avg : np.ndarray
            Updated averaging-selection mask.
        num_events : int | None, optional
            Replacement event count.
        """
        self.exclude_events = exclude_events
        self.use_for_avg = use_for_avg
        if num_events is not None:
            self.num_events = num_events

    def replace_trace(
        self, *, trace: MiniTrace, detection: EventDetection | None = None
    ) -> None:
        """
        Replace trace data while resetting analysis status and optional detection.

        Parameters
        ----------
        trace : MiniTrace
            Replacement trace object.
        detection : EventDetection | None, optional
            Detection state to associate with the replacement trace.
        """
        self.trace = trace
        self.filtered_trace = None
        self.was_analyzed = False
        self.detection = detection
        self.num_events = 0
        self.exclude_events = None
        self.use_for_avg = None
