import os

from miniml.resources.util import get_resource_file_path


class Settings:
    """
    Store analysis settings used by miniML event detection.

    Attributes
    ----------
    stride : int
        Stride of the sliding window used during feature extraction.
    event_window : int
        Length of the event window in samples.
    model_path : str
        Resolved filesystem path to the selected model file.
    model_name : str
        Model filename as provided to the constructor.
    event_threshold : float
        Minimum prediction peak height used for event detection.
    minimum_peak_width : int
        Minimum prediction peak width to classify a detection as an event.
    direction : str
        Event polarity to detect, typically ``"positive"`` or ``"negative"``.
    batch_size : int
        Batch size used during model inference.
    filter_factor : float
        Low-pass filter factor used during peak finding.
    gradient_convolve_win : int
        Hann window size used to smooth the first derivative for event timing.
    relative_prominence : float
        Relative prominence threshold used during post-processing.
    colors : list[str]
        Default color palette used for plotting and UI elements.
    """

    def __init__(
        self,
        stride: int = 20,
        event_length: int = 600,
        model: str = "GC_lstm_model.h5",
        event_threshold: float = 0.5,
        minimum_peak_width: int = 5,
        direction: str = "negative",
        batch_size: int = 512,
        filter_factor: float = 20,
        gradient_convolve_win: int = 25,
        relative_prominence: float = 0.25,
    ) -> None:
        self.stride = stride
        self.event_window = event_length
        self.model_path = model
        self.model_name = model
        self.event_threshold = event_threshold
        self.minimum_peak_width = minimum_peak_width
        self.direction = direction
        self.batch_size = batch_size
        self.filter_factor = filter_factor
        self.gradient_convolve_win = gradient_convolve_win
        self.relative_prominence = relative_prominence
        self.colors = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]

    @property
    def stride(self) -> int:
        return self._stride

    @stride.setter
    def stride(self, value) -> None:
        if value < 1:
            raise ValueError("Stride must be larger than 0")
        self._stride = value

    @property
    def event_window(self) -> float:
        return self._event_window

    @event_window.setter
    def event_window(self, value) -> None:
        if value < 1:
            raise ValueError("Event window must be larger than 0")

        self._event_window = value

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, value: str) -> None:
        if value.strip() == "":
            self._model_path = ""
            return

        self._model_path = get_resource_file_path(f"models/{value}")
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(f"Model file not found: {self._model_path}")

    @property
    def event_threshold(self) -> float:
        return self._event_threshold

    @event_threshold.setter
    def event_threshold(self, value) -> None:
        if value < 0 or value > 1:
            raise ValueError("Event threshold must be within (0,1)")

        self._event_threshold = value

    @property
    def minimum_peak_width(self) -> int:
        return self._minimum_peak_width

    @minimum_peak_width.setter
    def minimum_peak_width(self, value) -> None:
        if value < 1:
            raise ValueError("Minimum peak width must be larger than 0")
        if type(value) is not int:
            raise ValueError("Minimum peak width must be an integer")

        self._minimum_peak_width = value

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value) -> None:
        if value <= 0 or type(value) is not int:
            raise ValueError("Batch size must be a positive integer")

        self._batch_size = value

    @property
    def filter_factor(self) -> int:
        return self._filter_factor

    @filter_factor.setter
    def filter_factor(self, value) -> None:
        if value < 0:
            raise ValueError("filter_factor must be larger than 0")

        self._filter_factor = value

    @property
    def gradient_convolve_win(self) -> int:
        return self._gradient_convolve_win

    @gradient_convolve_win.setter
    def gradient_convolve_win(self, value) -> None:
        if value < 0:
            raise ValueError("Convolution window must be positive")
        if type(value) is not int:
            raise ValueError("Convolution window must be an integer")

        self._gradient_convolve_win = value

    @property
    def relative_prominence(self) -> float:
        return self._relative_prominence

    @relative_prominence.setter
    def relative_prominence(self, value) -> None:
        if value < 0 or value > 1:
            raise ValueError("Relative prominence must be within (0,1)")

        self._relative_prominence = value
