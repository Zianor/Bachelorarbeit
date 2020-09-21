import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, hilbert, butter, lfilter
from scipy.stats import median_absolute_deviation, kurtosis, skew

import src.utils as utils
from src.data_preparation import Data, BCGSeries, DataSeries
from src.data_processing import get_ecg_segment_hr, get_brueser_segment_hr


class DataSet:
    """
    A data set contains BCG Data in the form of 10 seconds segments. Furthermore a DataSet writes a .csv file with the
    statistical feature representation of all segments.
    """

    def __init__(self, segment_length=10, overlap_amount=0.9, coverage_threshold=90, mean_error_threshold=0.015, hr_threshold=10):
        self.path_csv = utils.get_statistical_features_csv_path(segment_length, overlap_amount, hr_threshold)
        self.path_images = os.path.join(utils.get_data_path(), 'images')
        self.coverage_threshold = coverage_threshold
        self.mean_error_threshold = mean_error_threshold
        self.hr_threshold = hr_threshold
        data = Data()
        self.segment_length = DataSet._seconds_to_frames(segment_length, data.sample_rate)  # in samples
        self.segment_distance = DataSet._seconds_to_frames(segment_length - segment_length * overlap_amount,
                                                           data.sample_rate)
        self._create_segments(data)
        self.save_csv()

    def _create_segments(self, data):
        """
        Creates segments with a given length out of given BCG Data
        """
        self.segments = []
        for data_series in data.data_series.values():
            for bcg_data in data_series.bcg_series.values():
                # TODO: wie passiert der Übergang/sync bei der gesplitteten Aufnahme? friemel ich die händisch aneinander?
                for i in range(0, len(bcg_data.raw_data), self.segment_distance):
                    if i + self.segment_length < len(
                            bcg_data.raw_data):  # prevent shorter segments, last shorter one ignored
                        if not data_series.reference_exists(i, i + self.segment_length):  # ignore if no reference ecg
                            continue
                        segment_data = np.array(bcg_data.raw_data[i:i + self.segment_length])
                        informative_ce, coverage, mean_error = self.is_informative_ce(bcg_data,
                                                                                      i,
                                                                                      i + self.segment_length)  # label
                        informative_hr, ecg_hr, bcg_hr = self.is_informative_hr(data_series, bcg_data, i, i + self.segment_length)
                        if ecg_hr == 0:
                            continue
                        self.segments.append(Segment(data_series.patient_id, segment_data, bcg_data.sample_rate,
                                                     informative_ce, informative_hr, ecg_hr, bcg_hr, coverage, mean_error))

    def is_informative_hr(self, series: DataSeries, bcg_series: BCGSeries, bcg_start, bcg_end):
        ecg_start, ecg_end = series.get_ecg_area(bcg_start, bcg_end)
        ecg_hr = get_ecg_segment_hr(ecg_start, ecg_end, series.ecg.r_peaks, series.ecg.sample_rate)
        bcg_hr = get_brueser_segment_hr(bcg_start, bcg_end, bcg_series.unique_peaks, bcg_series.medians,
                                        bcg_series.sample_rate)
        abs_diff = np.abs(ecg_hr-bcg_hr)
        if ecg_hr == 0 or 100/ecg_hr * abs_diff > self.hr_threshold:
            return False, ecg_hr, bcg_hr
        return True, ecg_hr, bcg_hr

    def is_informative_ce(self, series: BCGSeries, start, end):
        """
        Decides based on the coverage of detected intervals and the absolute mean error of these intervals to the ecg
        reference if the segment is informative
        :param series: the series the segment is part of
        :type series: DataSeries
        :param start: start index of the segment
        :type start: int
        :param end: end index of the segment
        :type end: int
        :return: if segment is informative, coverage, mean_error
        :rtype: boolean, float, float
        """
        indices = np.where(np.logical_and(start < series.indices, series.indices < end))[0]
        coverage = 100 / self.segment_length * sum(
            DataSet._seconds_to_frames(bbi, series.sample_rate) for bbi in series.bbi_bcg[indices])
        if len(indices) > 0:
            mean_error = sum(abs(series.bbi_bcg[i] - series.bbi_ecg[i]) for i in indices) / len(indices)
        else:
            mean_error = float("inf")
        if coverage < self.coverage_threshold or mean_error > self.mean_error_threshold:
            return False, coverage, mean_error
        return True, coverage, mean_error

    def save_csv(self):
        """
        Saves all segments as csv
        """
        with open(self.path_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(Segment.get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())

    def save_images(self, count=1000):
        """
        Saves segments as images
        :param count: number of images saved
        """
        if not os.path.exists(self.path_images):
            os.makedirs(self.path_images)
        count_informative = 0
        count_non_informative = 0
        for segment in self.segments:
            if count_non_informative + count_informative > count:
                break
            if segment.informative_ce:
                count_informative += 1
                plt.plot(segment.bcg)
                plt.title("Coverage " + str(segment.coverage) + ", Mean Error " + str(segment.mean_error))
                plt.savefig(os.path.join(self.path_images, 'informative' + str(count_informative)))
                plt.clf()
            else:
                count_non_informative += 1
                plt.plot(segment.bcg)
                plt.title("Coverage " + str(segment.coverage) + ", Mean Error " + str(segment.mean_error))
                plt.savefig(os.path.join(self.path_images, 'non-informative' + str(count_non_informative)))
                plt.clf()

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

    def __init__(self, patient_id, raw_data, sample_rate, informative_ce, informative_hr, ecg_hr, bcg_hr, coverage, mean_error):
        """
        Creates a segment and computes several statistical features
        :param raw_data: raw BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        :param coverage:
        :param mean_error: mean BBI error to reference
        """
        self.bcg = Segment._butter_bandpass_filter(raw_data, 1, 12, sample_rate)
        self.patient_id = patient_id
        self.coverage = coverage
        self.mean_error = mean_error
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
        maxima, _ = find_peaks(self.bcg)
        if len(maxima) == 0:  # TODO: decide how to deal with, , drop the segments?
            self.variance_local_maxima = 0
        else:
            self.variance_local_maxima = np.var(self.bcg[maxima])
        minima, _ = find_peaks(-self.bcg)
        if len(minima) == 0:  # TODO: decide how to deal with, drop the segments?
            self.variance_local_minima = 0
        else:
            self.variance_local_minima = np.var(self.bcg[minima])
        self.mean_signal_envelope = Segment._calc_mean_signal_envelope(self.bcg)
        self.informative_ce = informative_ce
        self.informative_hr = informative_hr
        self.ecg_hr = ecg_hr
        self.bcg_hr = bcg_hr

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

    @staticmethod
    def get_feature_name_array():
        return np.array(['minimum',
                         'maximum',
                         'mean',
                         'standard deviation',
                         'range',
                         'iqr',
                         'mad',
                         'number zero crossings',
                         'kurtosis',
                         'skewness',
                         'variance local maxima',
                         'variance local minima',
                         'mean signal envelope',
                         'informative_ce',
                         'informative_hr',
                         'ecg_hr',
                         'bcg_hr',
                         'mean error',
                         'coverage',
                         'patient_id'])

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
                         self.informative_ce,
                         self.informative_hr,
                         self.ecg_hr,
                         self.bcg_hr,
                         self.mean_error,
                         self.coverage,
                         self.patient_id])
