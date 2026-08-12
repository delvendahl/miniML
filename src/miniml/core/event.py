import numpy as np


class Event:
    """
    Represent an individual synaptic event with all its properties.

    Attributes
    ----------
    location : int
        Detected event location (steepest rise/onset position) in samples.
    score : float
        Prediction score or confidence associated with the event.
    waveform : np.ndarray | None
        Extracted waveform snippet around the event.
    peak_location : int
        Peak position in sample coordinates of the original trace.
    peak_value : float
        Event peak value (amplitude in y-units).
    bsl_start : int
        Start index of the baseline window in original trace coordinates.
    bsl_end : int
        End index of the baseline window in original trace coordinates.
    bsl_value : float
        Calculated baseline value (in y-units).
    bsl_duration : float
        Baseline window duration in samples.
    onset_location : int
        Calculated onset position in sample coordinates of the original trace.
    decaytime : float
        Half-decay time (in seconds).
    charge : float
        Event charge transfer (in pC).
    risetime : float
        10-90 percent rise time (in seconds).
    half_decay : float
        Half-decay position in sample coordinates of the original trace.
    halfwidth : float
        Event half-width (in seconds).
    rise_half_amp_time : float
        Time at rise half-amplitude (in seconds).
    decay_half_amp_time : float
        Time at decay half-amplitude (in seconds).
    min_position_rise : float
        Minimum rise position (in seconds).
    max_position_rise : float
        Maximum rise position (in seconds).
    min_value_rise : float
        Minimum rise value (in y-units).
    max_value_rise : float
        Maximum rise value (in y-units).
    slope : float
        Slope/steepest gradient value.
    excluded : bool
        Flag for potential exclusion of the event (primarily used in the GUI).
    """

    def __init__(
        self,
        location: int,
        score: float,
        waveform: np.ndarray | None = None,
        peak_location: int = -1,
        peak_value: float = np.nan,
        bsl_start: int = -1,
        bsl_end: int = -1,
        bsl_value: float = np.nan,
        bsl_duration: float = np.nan,
        onset_location: int = -1,
        decaytime: float = np.nan,
        charge: float = np.nan,
        risetime: float = np.nan,
        half_decay: float = np.nan,
        halfwidth: float = np.nan,
        rise_half_amp_time: float = np.nan,
        decay_half_amp_time: float = np.nan,
        min_position_rise: float = np.nan,
        max_position_rise: float = np.nan,
        min_value_rise: float = np.nan,
        max_value_rise: float = np.nan,
        slope: float = np.nan,
        excluded: bool = False,
    ) -> None:
        self.location = location
        self.score = score
        self.waveform = waveform
        self.peak_location = peak_location
        self.peak_value = peak_value
        self.bsl_start = bsl_start
        self.bsl_end = bsl_end
        self.bsl_value = bsl_value
        self.bsl_duration = bsl_duration
        self.onset_location = onset_location
        self.decaytime = decaytime
        self.charge = charge
        self.risetime = risetime
        self.half_decay = half_decay
        self.halfwidth = halfwidth
        self.rise_half_amp_time = rise_half_amp_time
        self.decay_half_amp_time = decay_half_amp_time
        self.min_position_rise = min_position_rise
        self.max_position_rise = max_position_rise
        self.min_value_rise = min_value_rise
        self.max_value_rise = max_value_rise
        self.slope = slope
        self.excluded = excluded

    def __repr__(self) -> str:
        return (
            f"Event(location={self.location}, score={self.score:.4f}, "
            f"amplitude={self.peak_value - self.bsl_value:.4f}, excluded={self.excluded})"
        )
