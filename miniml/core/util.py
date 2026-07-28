import os

import numpy as np
import tensorflow as tf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def exp_fit(x: np.ndarray, amp: float, tau: float, offset: float) -> np.ndarray:
    """
    Fit an exponential curve to the given data.

    Parameters:
        x (np.ndarray): The input data.
        amp (float): The amplitude of the exponential curve.
        tau (float): The time constant of the exponential curve.
        offset (float): The offset of the exponential curve.

    Returns:
        np.ndarray: The fitted exponential curve.
    """

    return amp * np.exp(-(x - x[0]) / tau) + offset


@tf.function
def minmax_scaling(x: tf.Tensor) -> tf.Tensor:
    """
    Apply min-max scaling to the input tensor.

    Args:
        x (tf.Tensor): The input tensor to be scaled.

    Returns:
        tf.Tensor: The scaled tensor.
    """
    x_min = tf.expand_dims(tf.math.reduce_min(x), axis=-1)
    x_max = tf.expand_dims(tf.math.reduce_max(x), axis=-1)

    return tf.math.divide(tf.math.subtract(x, x_min), tf.math.subtract(x_max, x_min))


def mEPSC_template(
    x: np.ndarray, amplitude: float, t_rise: float, t_decay: float, x0: float
) -> np.ndarray:
    """
    Generate a template miniature excitatory postsynaptic current
    (mEPSC) based on the given parameters.

    Parameters:
        x (np.ndarray): An array of x values.
        amplitude (float): The amplitude of the mEPSCs.
        t_rise (float): The rise time constant of the mEPSCs.
        t_decay (float): The decay time constant of the mEPSCs.
        x0 (float): The onset time point for the mEPSCs.

    Returns:
        np.ndarray: An array of y values representing an mEPSC template.

    Note:
        - The formula used to calculate the template is
          y = amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay).
        - Any values of x that are less than x0 will be set to 0 in the resulting array.
    """
    y = amplitude * (1 - np.exp(-(x - x0) / t_rise)) * np.exp(-(x - x0) / t_decay)
    y[x < x0] = 0

    return y
