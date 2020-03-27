import csv
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, butter, lfilter
from scipy.stats import median_absolute_deviation, kurtosis, skew
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data_preparation import BcgData
from src.utils import get_project_root


class DataSet:
    """
    A data set contains BCG Data in the form of 10 seconds segments. Furthermore a DataSet writes a .csv file with the
    statistical feature representation of all segments.
    """

    def __init__(self, coverage_threshold=90, mean_error_threshold=0.015):
        self.path_csv = os.path.join(get_project_root(), 'data/data.csv')
        self.path_images = os.path.join(get_project_root(), 'data/images')
        self.coverage_threshold = coverage_threshold
        self.mean_error_threshold = mean_error_threshold
        self.data = BcgData()
        self.segment_length = DataSet._seconds_to_frames(10, self.data.samplerate)  # in samples
        self._create_segments()
        self.save_csv()

    def _create_segments(self):
        """
        Creates segments with a given length out of given BCG Data
        """
        self.segments = []
        for series in self.data.data_series:
            for i in range(0, len(series.raw_data), self.segment_length):
                if i + self.segment_length < len(series.raw_data):  # prevent shorter segments, last shorter one ignored
                    segment_data = np.array(series.raw_data[i:i + self.segment_length])
                    informative, coverage, mean_error = self.is_informative(series, i, i + self.segment_length)  # label
                    self.segments.append(Segment(segment_data, self.data.samplerate, informative, coverage, mean_error))

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
        :return: if segment is informative, coverage, mean_error
        :rtype: boolean, float, float
        """
        indices = np.where(np.logical_and(start < series.indices, series.indices < end))[0]
        coverage = 100 / self.segment_length * sum(
            DataSet._seconds_to_frames(bbi, series.samplerate) for bbi in series.bbi_bcg[indices])
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
            if segment.informative:
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

    def __init__(self, raw_data, samplerate, informative, coverage, mean_error):
        """
        Creates a segment and computes several statistical features
        :param raw_data: raw BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        :param coverage:
        :param mean_error: mean BBI error to reference
        """
        self.bcg = Segment._butter_bandpass_filter(raw_data, 1, 12, samplerate)
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
                         'informative'])

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


def load_data():
    """
    Loads BCG data features with its target labels
    :return: BCG data features, target labels
    """
    path = os.path.join(get_project_root(), 'data/data.csv')
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet()
    df = pd.read_csv(path)
    features = df.iloc[:, 0:12]
    target = df.iloc[:, 13]
    return features, target


def data_preparation(features, target):
    """
    Splits data in training and test data and standardizes features
    :param features: feature matrix
    :param target: target vector
    :return: x_train_std, x_test_std, y_train, y_test
    """
    # Split dataset in 2/3 training and 1/3 test data
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.333, random_state=1, stratify=target)

    print('Labels counts in y:', np.bincount(target))
    print('Labels counts in y_train:', np.bincount(y_train))
    print('Labels counts in y_test:', np.bincount(y_test))

    # Standardizing features
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    return x_train_std, x_test_std, y_train, y_test


def evaluate(y_actual, y_pred):
    """
    Evaluates predicted labels
    :param y_actual: actual labels
    :param y_pred: predicted labels
    """
    print('Misclassified examples: %d' % (y_actual != y_pred).sum())
    print('Accuracy: %.3f' % accuracy_score(y_actual, y_pred))

    confusion = confusion_matrix(y_actual, y_pred)
    print('Confusion matrix')
    print(confusion)


def support_vector_machine(features, target):
    """
    Support vector machine
    :param features: feature matrix
    :param target: target vector
    """
    x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=1)
    svm.fit(x_train_std, y_train)

    y_pred = svm.predict(x_test_std)

    evaluate(y_test, y_pred)


if __name__ == "__main__":
    X, y = load_data()

    support_vector_machine(X, y)
