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
