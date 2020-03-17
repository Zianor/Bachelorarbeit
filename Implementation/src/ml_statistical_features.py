from src.data_preparation import BcgData
import numpy as np
from scipy.stats import median_absolute_deviation, kurtosis, skew
from scipy.signal import find_peaks, hilbert


class Segment:
    """A segment of bcg data with its statistical features based on the paper 'Sensor data quality processing for
    vital signs with opportunistic ambient sensing' (https://ieeexplore.ieee.org/document/7591234)"""

    def __init__(self, filtered_data):
        self.bcg = filtered_data
        self.minimum = np.min(self.bcg)
        self.maximum = np.max(self.bcg)
        self.mean = np.mean(self.bcg)
        self.standard_deviation = np.std(self.bcg)
        self.range = self.maximum - self.minimum
        self.iqr = np.subtract(*np.percentile(self.bcg, [75, 25]))
        self.mad = median_absolute_deviation(self.bcg)
        self.number_zero_crossings = (np.diff(np.sign(self.bcg)) != 0).sum()
        self.kurtosis = kurtosis(self.bcg)
        self.skewness = skew(self.bcg)
        self.variance_local_maxima = np.var(self.bcg[find_peaks(self.bcg)])
        self.variance_local_minima = np.var(self.bcg[find_peaks(-self.bcg)])
        self.mean_signal_envelope = self._calc_mean_signal_envelope(self.bcg)

    @staticmethod
    def _calc_mean_signal_envelope(signal):
        """:return mean of upper envelope of rectified signal"""
        analytic_signal = hilbert(np.abs(signal))
        amplitude_envelope = np.abs(analytic_signal)
        return np.mean(amplitude_envelope)
