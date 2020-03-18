from src.data_preparation import BcgData
import numpy as np
from scipy.stats import median_absolute_deviation, kurtosis, skew
from scipy.signal import find_peaks, hilbert, butter, lfilter


class DataSet:

    def __init__(self):
        data = BcgData()
        self.segment_length = 1000  # in samples
        self._create_segments(data, self.segment_length)
        # TODO: split in trainings and test set

    def _create_segments(self, data, segment_length):
        self.segments = []
        for series in data.data_series:
            for i in range(len(series.raw_data))[::segment_length]:
                segment_data = series.raw_data[i:i+segment_length]
                informative = DataSet.is_informative()  # label as informative or non informative
                self.segments.append(segment_data, data.samplerate, informative)

    @staticmethod
    def is_informative():
        # TODO: implement, incl. paramaters
        return False


class Segment:
    """A segment of bcg data with its statistical features based on the paper 'Sensor data quality processing for
    vital signs with opportunistic ambient sensing' (https://ieeexplore.ieee.org/document/7591234)"""

    def __init__(self, raw_data, samplerate, informative):
        """
        Creates a segment and computes several statistical features
        :param filtered_data: preprocessed BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        """
        self.bcg = Segment._butter_bandpass_filter(raw_data, 1000, 12000, samplerate)
        self.informative = informative
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
        self.mean_signal_envelope = Segment._calc_mean_signal_envelope(self.bcg)

    @staticmethod
    def _calc_mean_signal_envelope(signal):
        """:return mean of upper envelope of rectified signal"""
        analytic_signal = hilbert(np.abs(signal))
        amplitude_envelope = np.abs(analytic_signal)
        return np.mean(amplitude_envelope)

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        """Butterworth Bandpass
        :param data: data to be filtered
        :param lowcut: lowcut frequency in Hz
        :param highcut: highcut frequency in Hz
        :param fs: sample rate in Hz"""
        b, a = Segment._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        """Used internally for Butterworth Bandpass"""
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

