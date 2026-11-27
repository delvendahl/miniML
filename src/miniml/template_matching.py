#
# template matching for mEPSC detection according to Clements & Bekkers (1997)
# https://doi.org/10.1016/S0006-3495(97)78062-7
# implementation by Martin Rückl, Friedrich Johenning, Stephen Lenzi
# https://github.com/samuroi/SamuROI
# with speed improvements by Igor Delvendahl
#

from typing import NamedTuple

import numpy as np
from scipy.signal import fftconvolve


class TemplateMatchResult(NamedTuple):
    """Collection of results of least squares optimization for template matching."""

    indices: np.ndarray
    """the indices of detected events, since zero padding is used for convolution, all indices need to get shifted by N/2 where N is the length of the kernel"""
    detection_trace: np.ndarray
    """the criterion vector used for comparison with the threshold"""
    s: np.ndarray
    """the vector of optimal scaling parameters"""
    c: np.ndarray
    """the vector of optimal offset parameters"""
    threshold: float
    """the threshold used for detection"""
    kernel: np.ndarray
    """the kernel that was used for matching"""

    def __repr__(self):
        return f"TemplateMatchResult(indices={self.indices}, detection_trace={self.detection_trace}, s={self.s}, c={self.c}, threshold={self.threshold}, kernel={self.kernel})"


def make_template(
    t_rise: float = 0.0005,
    t_decay: float = 0.001,
    baseline: float = 0.001,
    duration: float = 0.004,
    sampling: float = 2e-5,
) -> np.ndarray:
    """
    Creates a normalized template for template matching.
    :param t_rise: rise time of the template in milliseconds
    :param t_decay: decay time of the template in milliseconds
    :param baseline: baseline duration of the template in milliseconds
    :param duration: duration of the template in milliseconds
    :param sampling: sampling interval of the template in seconds
    """
    x = np.arange(0, duration + baseline, sampling)
    a = 1.0
    y = a * (1 - np.exp(-(x - baseline) / t_rise)) * np.exp(-(x - baseline) / t_decay)
    y[x < baseline] = 0

    return y / np.max(y)


def template_matching(
    data: np.ndarray, kernel: np.ndarray, threshold: float
) -> TemplateMatchResult:
    """Detect events in a time series by matching a template via least squares.

    Implements the detection algorithm of Clements & Bekkers [1]_: at every
    position ``n`` of the time series :math:`\\mathbf{y}` (``data``), the
    normalized template :math:`\\mathbf{e}` (``kernel``) of length :math:`K`
    is fitted to the data window :math:`y_{n}, \\ldots, y_{n+K-1}` by
    minimizing the least squares distance

    .. math::

        \\chi_n^2(S, C) = \\sum_{k} \\left[ y_{n+k} - (S \\, e_k + C) \\right]^2

    with respect to the scaling factor :math:`S` and the offset :math:`C`,
    where

    .. math::

        \\overline{e} = \\frac{1}{K} \\sum_k e_k, \\qquad
        \\overline{y_n} = \\frac{1}{K} \\sum_k y_{n+k}

    The optimal parameters are (Clements & Bekkers, Appendix I [1]_):

    .. math::

        S_n = \\frac{\\sum_k e_k y_{n+k} - K \\, \\overline{e} \\, \\overline{y_n}}
                   {\\sum_k e_k^2 - K \\, \\overline{e}^2},
        \\qquad
        C_n = \\overline{y_n} - S_n \\, \\overline{e}

    Events are detected wherever the detection criterion

    .. math::

        \\mathrm{crit}_n = \\frac{S_n}{\\sqrt{\\chi_n^2 / (K - 1)}}

    crosses ``threshold``. Consecutive positions above the threshold are
    collapsed into a single event, reported at the first crossing.

    Parameters
    ----------
    data : (M,) ndarray
        1D time series to analyze, denoted :math:`\\mathbf{y}` above.
    kernel : (K,) ndarray
        1D template to match against the data, denoted :math:`\\mathbf{e}`
        above. Typically a normalized (zero-mean, unit-norm) event shape.
    threshold : float
        Detection threshold applied to the criterion
        :math:`\\mathrm{crit}_n`. Positive values detect upward crossings
        (events with :math:`S > 0`); negative values detect downward
        crossings. Values in the range 1 to 5 usually give reasonable
        results; 4 to 5 is a common choice for low false-positive rates.

    Returns
    -------
    TemplateMatchResult
        Named tuple with the following fields:

        * ``indices`` : ndarray of int -- start indices of the detected
          events in ``data``.
        * ``detection_trace`` : (M,) ndarray of float -- the criterion
          :math:`\\mathrm{crit}_n` at every position of ``data``.
        * ``s`` : (M,) ndarray of float -- optimal scaling parameter
          :math:`S_n` at every position.
        * ``c`` : (M,) ndarray of float -- optimal offset parameter
          :math:`C_n` at every position.
        * ``threshold`` : float -- the threshold used for detection.
        * ``kernel`` : ndarray -- the template that was used for matching.

    Raises
    ------
    ValueError
        If ``data`` is not longer than ``kernel``.

    Notes
    -----
    Zero padding is used at the boundaries, so the returned ``detection``,
    ``s``, and ``c`` traces have the same length as ``data``. Event indices
    are already shifted by :math:`K/2` to compensate for this, i.e. they
    point at the first sample of the matched window.

    The traces ``s``, ``c``, and ``detection_trace`` are computed at every
    position of the data, not only at the detected events, which allows
    post-hoc inspection of the fit quality.

    Examples
    --------
    >>> kernel = make_template(duration=0.005, fs=10000)  # doctest: +SKIP
    >>> result = template_matching(data, kernel, threshold=4)
    >>> result.indices
    array([ 120,  405, 1120])

    References
    ----------
    .. [1] Clements JD, Bekker JM (1997). "A numerical method for detection
           and estimation of spontaneous miniature synaptic events."
           Biophysical Journal 73(1):394-402.
           http://dx.doi.org/10.1016/S0006-3495(97)78062-7
    """

    if len(data) <= len(kernel):
        raise ValueError("Data length needs to exceed kernel length.")

    data = np.asarray(data, dtype=np.float64)
    e = kernel[::-1]  # reversed kernel, since we convolve
    N = e.size
    M = data.size

    # scalars, computed once
    sum_e = e.sum()
    sum_ee = np.dot(e, e)  # avoids the e**2 temporary
    denom = sum_ee - sum_e * sum_e / N  # constant across all positions

    # --- sliding-window sums via cumulative sums: O(M) instead of O(M*N) ---
    # For output position t, np.convolve(..., mode="same") with a ones kernel
    # sums data[t - N//2 : t + (N-1)//2 + 1] (zero-padded at the edges).
    t = np.arange(M)
    lo = np.maximum(t - N // 2, 0)
    hi = np.minimum(t + (N - 1) // 2 + 1, M)

    c = np.concatenate(([0.0], np.cumsum(data)))
    c2 = np.concatenate(([0.0], np.cumsum(data * data)))
    sum_y = c[hi] - c[lo]  # sum of y over each window
    sum_yy = c2[hi] - c2[lo]  # sum of y**2 over each window

    # --- data x kernel via FFT: O(M log M) instead of O(M*N) ---
    sum_ey = fftconvolve(data, e, mode="same")

    # optimal scaling and offset
    s_n = (sum_ey - sum_e * sum_y / N) / denom
    c_n = (sum_y - s_n * sum_e) / N

    # SSE at the optimum, via the normal equations:
    #   sum(y^2) - S*sum(e*y) - C*sum(y)   (algebraically identical to the
    #   original 6-term expression, but with far fewer vector operations)
    sse_n = sum_yy - s_n * sum_ey - c_n * sum_y

    # detection criterion
    with np.errstate(divide="ignore", invalid="ignore"):
        crit = s_n / np.sqrt(sse_n / (N - 1))

    # threshold crossings (same logic as before)
    pos = np.flatnonzero(crit < threshold if threshold < 0 else crit > threshold)
    indices = pos[np.flatnonzero(np.diff(pos, prepend=0) > 1)] - N // 2
    indices = indices[indices > 0]  # handle negative indices

    return TemplateMatchResult(
        indices=indices,
        detection_trace=crit,
        s=s_n,
        c=c_n,
        threshold=threshold,
        kernel=kernel,
    )
