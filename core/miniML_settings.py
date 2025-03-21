import os


class MinimlSettings():
    def __init__(self, 
                 stride: int=20, 
                 event_length: int=600,
                 model: str='GC_lstm_model.h5',
                 event_threshold: float=0.5,
                 minimum_peak_width: int=5,
                 direction: str='negative',
                 batch_size: int=512,
                 convolve_win: int=20,
                 gradient_convolve_win: int=0) -> None:

        self.stride = stride
        self.event_window = event_length
        self.model_path = model
        self.model_name = model
        self.event_threshold = event_threshold
        self.minimum_peak_width = minimum_peak_width
        self.direction = direction
        self.batch_size = batch_size
        self.convolve_win = convolve_win
        self.gradient_convolve_win = gradient_convolve_win
        self.colors = ["#ff595e", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]


    @property
    def stride(self) -> int:
        return self._stride
    
    @stride.setter
    def stride(self, value) -> None:
        if value < 1:
            raise ValueError('Stride must be larger than 0')
        self._stride = value

    
    @property
    def event_window(self) -> float:
        return self._event_window
    
    @event_window.setter
    def event_window(self, value) -> None:
        if value < 1:
            raise ValueError('Event window must be larger than 0')

        self._event_window = value

    @property
    def model_path(self) -> str:
        return self._model_path

    @model_path.setter
    def model_path(self, value) -> None:
        model_path = f'../models/{value}'
        if not os.path.exists(model_path):
            raise FileNotFoundError('Model file not found')

        self._model_path = model_path

    @property
    def event_threshold(self) -> float:
        return self._event_threshold
    
    @event_threshold.setter
    def event_threshold(self, value) -> None:
        if value < 0 or value > 1:
            raise ValueError('Event threshold must be within (0,1)')

        self._event_threshold = value

    @property
    def minimum_peak_width(self) -> int:
        return self._minimum_peak_width
    
    @minimum_peak_width.setter
    def minimum_peak_width(self, value) -> None:
        if value < 1:
            raise ValueError('Minimum peak width must be larger than 0')
        if type(value) is not int:
            raise ValueError('Minimum peak width must be an integer')

        self._minimum_peak_width = value

    @property
    def batch_size(self) -> int:
        return self._batch_size
    
    @batch_size.setter
    def batch_size(self, value) -> None:
        if value <= 0 or type(value) is not int:
            raise ValueError('Batch size must be a positive integer')

        self._batch_size = value


    @property
    def convolve_win(self) -> int:
        return self._convolve_win
    
    @convolve_win.setter
    def convolve_win(self, value) -> None:
        if value < 1:
            raise ValueError('Convolution window must be larger than 0')
        if type(value) is not int:
            raise ValueError('Convolution window must be an integer')

        self._convolve_win = value


    @property
    def gradient_convolve_win(self) -> int:
        return self._gradient_convolve_win
    
    @gradient_convolve_win.setter
    def gradient_convolve_win(self, value) -> None:
        if value < 0:
            raise ValueError('Convolution window must be positive')
        if type(value) is not int:
            raise ValueError('Convolution window must be an integer')

        self._gradient_convolve_win = value
