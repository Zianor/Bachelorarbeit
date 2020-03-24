import csv
import os

import numpy as np
from scipy.signal import find_peaks, hilbert, butter, lfilter
from scipy.stats import median_absolute_deviation, kurtosis, skew

from src.data_preparation import BcgData
from src.utils import get_project_root


class DataSet:
    segment_length = 1000  # in samples

    def __init__(self, coverage_threshold, mean_error_threshold):
        self.path = os.path.join(get_project_root(), 'data/data.csv')
        self.coverage_threshold = coverage_threshold
        self.mean_error_threshold = mean_error_threshold
        self.data = BcgData()
        print("Create segments and calculate features")
        self._create_segments(self.data, self.segment_length)
        print("Save segments")
        self.save_csv()

    def _create_segments(self, data, segment_length):
        """Creates segments with a given length out of given BCG Data
                :param data: BCG Data
                :type data: BcgData
                :param segment_length: segment length in samples
                :type segment_length: int
                """
        self.segments = []
        for series in data.data_series:
            for i in range(0, len(series.raw_data), segment_length):
                if i + segment_length < len(series.raw_data):  # to prevent shorter segments, last shorter one ignored
                    segment_data = np.array(series.raw_data[i:i + segment_length])
                    informative = self.is_informative(series, i, i + segment_length)  # label
                    self.segments.append(Segment(segment_data, data.samplerate, informative))

    def is_informative(self, series, start, end):
        """
        Decides based on the coverage of detected intervals and the absolute mean error of these intervals to the ecg
        reference if the segment is informative
        :param series: the series the segment is part of
        :type series: DataSeries
        :param start: start index of the segment
        :type start: int
        :param end: end index of the segment
        :type end: int
        :return: if segment is informative
        :rtype: boolean
        """
        indices = np.where(np.logical_and(start < series.indices, series.indices < end))[0]
        coverage = 100 / DataSet.segment_length * sum(
            DataSet._seconds_to_frames(bbi, series.samplerate) for bbi in series.bbi_bcg[indices])
        if coverage < self.coverage_threshold:
            return False
        mean_error = sum(abs(series.bbi_bcg[i] - series.bbi_ecg[i]) for i in indices) / len(indices)
        if mean_error > self.mean_error_threshold:
            return False
        return True

    def save_csv(self):
        """
        Saves all segments as csv
        """
        with open(self.path, 'w') as f:
            writer = csv.writer(f)
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())

    @staticmethod
    def _seconds_to_frames(duration_seconds, frequency):
        """
        Converts a given duration in seconds to the number of frames in a given frequency
        :param duration_seconds: given duration in seconds
        :type duration_seconds: float
        :param frequency: frequency in Hz
        :type frequency: int
        :return: duration in number of frames
        :rtype: int
        """
        duration_frames = duration_seconds * frequency
        return int(duration_frames)


class Segment:
    """
    A segment of bcg data with its statistical features based on the paper 'Sensor data quality processing for
    vital signs with opportunistic ambient sensing' (https://ieeexplore.ieee.org/document/7591234)
    """

    def __init__(self, raw_data, samplerate, informative):
        """
        Creates a segment and computes several statistical features
        :param raw_data: raw BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        """
        bcg = Segment._butter_bandpass_filter(raw_data, 1, 12, samplerate)
        self.minimum = np.min(bcg)
        self.maximum = np.max(bcg)
        self.mean = np.mean(bcg)
        self.standard_deviation = np.std(bcg)
        self.range = self.maximum - self.minimum
        self.iqr = np.subtract(*np.percentile(bcg, [75, 25]))
        self.mad = median_absolute_deviation(bcg)
        self.number_zero_crossings = (np.diff(np.sign(bcg)) != 0).sum()
        self.kurtosis = kurtosis(bcg)
        self.skewness = skew(bcg)
        maxima, _ = find_peaks(bcg)
        self.variance_local_maxima = np.var(bcg[maxima])
        if np.isnan(self.variance_local_maxima):
            self.variance_local_maxima = 0
        minima, _ = find_peaks(-bcg)
        self.variance_local_minima = np.var(bcg[minima])
        if np.isnan(self.variance_local_minima):
            self.variance_local_minima = 0
        self.mean_signal_envelope = Segment._calc_mean_signal_envelope(bcg)
        self.informative = informative

    @staticmethod
    def _calc_mean_signal_envelope(signal):
        """
        :return: mean of upper envelope of rectified signal
        """
        analytic_signal = hilbert(np.abs(signal))
        amplitude_envelope = np.abs(analytic_signal)
        return np.mean(amplitude_envelope)

    @staticmethod
    def _butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
        """
        Butterworth Bandpass
        :param data: data to be filtered
        :param lowcut: lowcut frequency in Hz
        :param highcut: highcut frequency in Hz
        :param fs: sample rate in Hz
        """
        b, a = Segment._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=5):
        """
        Used internally for Butterworth Bandpass
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        return np.array([self.minimum,
                         self.maximum,
                         self.mean,
                         self.standard_deviation,
                         self.range,
                         self.iqr,
                         self.mad,
                         self.number_zero_crossings,
                         self.kurtosis,
                         self.skewness,
                         self.variance_local_maxima,
                         self.variance_local_minima,
                         self.mean_signal_envelope,
                         self.informative])


if __name__ == "__main__":
    data_set = DataSet(85, 0.005)
    count_informative = sum(segment.informative for segment in data_set.segments)
    print("Informative: " + str(count_informative))
    print("Not-informative: " + str(len(data_set.segments)-count_informative))
