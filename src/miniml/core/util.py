import json
import os
from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

FloatArray: TypeAlias = npt.NDArray[np.float64]


def exp_fit(x: FloatArray, amp: float, tau: float, offset: float) -> FloatArray:
    """
    Evaluate a single-exponential decay curve.

    Parameters
    ----------
    x : numpy.ndarray
        Sample positions at which to evaluate the curve.
    amp : float
        Exponential amplitude.
    tau : float
        Exponential time constant.
    offset : float
        Constant offset added to the curve.

    Returns
    -------
    numpy.ndarray
        Exponential decay evaluated relative to the first sample in ``x``.
    """

    return amp * np.exp(-(x - x[0]) / tau) + offset


@tf.function
def minmax_scaling(x: tf.Tensor) -> tf.Tensor:
    """
    Apply min-max normalization to a tensor.

    Parameters
    ----------
    x : tf.Tensor
        Input tensor to normalize.

    Returns
    -------
    tf.Tensor
        Tensor rescaled to the ``[0, 1]`` range using its minimum and maximum.
    """
    x_min = tf.expand_dims(tf.math.reduce_min(x), axis=-1)
    x_max = tf.expand_dims(tf.math.reduce_max(x), axis=-1)

    if tf.math.equal(x_max, x_min):
        return tf.zeros_like(x)

    return tf.math.divide(tf.math.subtract(x, x_min), tf.math.subtract(x_max, x_min))


def mEPSC_template(
    x: FloatArray, amplitude: float, t_rise: float, t_decay: float, x0: float
) -> FloatArray:
    """
    Generate a miniature excitatory postsynaptic current template.

    Parameters
    ----------
    x : numpy.ndarray
        Sample positions for the template waveform.
    amplitude : float
        Peak scaling factor of the template current.
    t_rise : float
        Rise time constant.
    t_decay : float
        Decay time constant.
    x0 : float
        Onset position of the template.

    Returns
    -------
    numpy.ndarray
        Template waveform with all samples before ``x0`` set to zero.

    Notes
    -----
    The template is computed as

    ``amplitude * (1 - exp(-(x - x0) / t_rise)) * exp(-(x - x0) / t_decay)``.
    """
    y = amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay)
    y[x < x0] = 0

    return y


def robust_noise_mad(
    gradient: FloatArray, multiplier: float = 4.0
) -> tuple[float, float]:
    """
    Calculates a robust noise threshold using the Median Absolute Deviation.

    Parameters
    ----------
    gradient : np.ndarray
        The gradient trace from which to calculate the noise threshold.
    multiplier : float, optional
        The multiplier for the robust standard deviation to set the threshold.
        Default is 4.0.

    Returns
    -------
    tuple
        A tuple containing the noise threshold and the robust standard deviation.
    """
    median_grad = np.median(gradient)
    abs_dev = np.abs(gradient - median_grad)
    mad = np.median(abs_dev)

    # Convert MAD to an equivalent Standard Deviation (sigma)
    robust_sigma = 1.4826 * mad

    # Set threshold (e.g., 4 * sigma above/below median)
    threshold = median_grad + (multiplier * robust_sigma)

    return threshold, robust_sigma


def parse_model_info(model: tf.keras.Model) -> dict:
    """
    Parse the model information from a Keras model.

    Parameters
    ----------
    model : tf.keras.Model
        The Keras model to parse.

    Returns
    -------
    dict
        A dictionary containing the model information.

    """
    model_info = json.loads(model.to_json())
    model_name = model_info.get("config", {}).get("name", "unknown")
    model_backend = model_info.get("backend", "unknown")
    version = model_info.get("keras_version", "unknown")

    return {"name": model_name, "backend": model_backend, "version": version}
