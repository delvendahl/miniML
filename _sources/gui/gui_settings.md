# miniML Settings

The Settings menu allows to change the default settings of miniML.

## Stride length

Stride length defines the step size for the sliding window in samples. The default value is 20 samples. This parameter determiens the resolution of the event detection.

```{tip}
Larger values will speed inference, but may cause poorer detection performance. We recommend to adjust this parameter only if necessary
```

## Event length

This parameter defines the length of the events in samples. The default value is 600 samples. Event length can be adjusted if event kinetics differ from the training data. Use a higher number for events with slower kinetics, and a lower number for events with faster kinetics. 

```{tip}
We recommend to adjust this parameter with steps of 300.
```

## Minimum peak height

Defines the minimum peak height in the prediction trace (range, 0-1). The default value is 0.5. This parameter can be adjusted to increase or decrease the sensitivity of the detection algorithm. 

## Minimum peak width

Defines the minimum peak width in strides. The default value is 5. This parameter can be adjusted to increase or decrease the sensitivity of the detection algorithm.

## Model

Allows selection of the miniML model to be used for event detection. The default model is the GC model. The dropdown menu lists all model files in the `models` folder.

## Event direction

Allows selection of the direction of the events to be detected. The default is 'negative'.

## Batch size

Sets the batch size for model inference. The default value is 512. This parameter can be adjusted to increase or decrease the speed of the detection algorithm. 

```{important}
Large values of the batch size can sometimes lead to errors when running miniML on a GPU. Check the prediction trace for errors if you encounter problems.
```

## Filter factor

Defines the filter factor for the lowpass Butterworth filter. The data are fitlered using a forward-backward Butterworth filter with the specified filter factor (cutoff frequency = sampling rate / filter factor). The default value is 25. The raw data is filtered for determining event location and event analysis. This parameter can be adjusted to optimize the event location and peak location determination.

## Gradient filter window

Defines the Hann window size for the gradient convolution filter. The default value is 0. The gradient of the raw data is filtered for determining event location (steepest rise). This parameter can be adjusted to improve the detection of event location.

## Relative prominence

The relative prominence cutoff determines the minimum peak in the gradient trace for acceptin additional events in the vicinity of an already detected event. The default value is 0.25. Setting this parameter to 1 will disable the relative prominence filter and prevent the detection of neraby or overlapping events.

```{important}
Very low values will typically lead to detection of false positives.
```