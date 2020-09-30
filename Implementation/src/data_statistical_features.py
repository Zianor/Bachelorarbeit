import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert
from scipy.stats import median_absolute_deviation, kurtosis, skew

import utils
from data_preparation import DataSeries, Data


class DataSet:
    """A data set contains BCG Data in the form of segments of a certain length and overlap. Furthermore a DataSet
    writes a .csv file with the feature representation of all segments.
    """

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        self.hr_threshold = hr_threshold
        self.segment_length = segment_length
        self.overlap_amount = overlap_amount
        self.segments = []
        self._create_segments()
        self.save_csv()

    def _create_segments(self):
        """
        Creates segments with a given length out of given BCG Data
        """
        if os.path.isfile(utils.get_data_object_path()):
            with open(utils.get_data_object_path(), 'rb') as file:
                data = pickle.load(file)
        else:
            data = Data()
            with open(utils.get_data_object_path(), 'wb') as file:
                pickle.dump(data, file)
                file.flush()
        segment_length_frames = utils.seconds_to_frames(self.segment_length, data.sample_rate)  # in samples
        segment_distance = utils.seconds_to_frames(self.segment_length - self.segment_length * self.overlap_amount,
                                                   data.sample_rate)
        for series in data.data_series.values():
            for i in range(0, len(series.bcg.raw_data), segment_distance):
                if i + self.segment_length < len(
                        series.bcg.raw_data):  # prevent shorter segments, last shorter one ignored
                    start = i
                    end = i + segment_length_frames
                    if not series.reference_exists(start, end):  # ignore if no reference ecg
                        continue
                    ecg_hr = series.get_ecg_hr(start, end)
                    brueser_sqi = series.get_brueser_sqi(start, end)
                    bcg_hr = series.get_bcg_hr(start, end)
                    informative = self.is_informative(ecg_hr, bcg_hr)
                    self.segments.append(
                        self._get_segment(series, start, end, informative, ecg_hr, brueser_sqi, bcg_hr))

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        return Segment(
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            informative=informative
        )

    def is_informative(self, ecg_hr, bcg_hr):
        abs_err = np.abs(ecg_hr - bcg_hr)
        rel_err = 100 / ecg_hr * abs_err
        if np.isnan(bcg_hr):
            return False
        if ecg_hr / 100 * self.hr_threshold > self.hr_threshold/2:
            if rel_err > self.hr_threshold:
                return False
            else:
                return True
        else:
            if abs_err > self.hr_threshold/2:
                return False
        return True

    def save_csv(self):
        """
        Saves all segments as csv
        """
        if not os.path.isdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount))
        path = utils.get_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(Segment.get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())
            f.flush()
        data = pd.read_csv(path, index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(utils.get_features_csv_path(self.segment_length, self.overlap_amount), index=False)

    def save_images(self, count=50):
        """
        Saves segments as images
        :param count: number of images saved
        """
        path_images = utils.get_image_folder(self.segment_length_seconds, self.overlap_amount, self.hr_threshold)
        if not os.path.exists(path_images):
            os.makedirs(path_images)
        count_informative = 0
        count_non_informative = 0
        for segment in self.segments:
            if count_non_informative > count and count_informative > count:
                break
            if segment.informative and count_informative < count:
                count_informative += 1
                plt.plot(segment.bcg)
                plt.title("Abs Error " + str(segment.abs_err) + ", Rel Error " + str(segment.rel_err))
                plt.savefig(os.path.join(self.path_images, 'informative' + str(count_informative)))
                plt.clf()
            else:
                if count_non_informative < count:
                    count_non_informative += 1
                    plt.plot(segment.bcg)
                    plt.title("Abs Error " + str(segment.abs_err) + ", Rel Error " + str(segment.rel_err))
                    plt.savefig(os.path.join(path_images, 'non-informative' + str(count_non_informative)))
                    plt.clf()


class DataSetStatistical(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(DataSetStatistical, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, ecg_hr_std, brueser_sqi, bcg_hr):
        return SegmentStatistical(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative
        )

    def save_csv(self):
        """Saves all segments as csv
        """
        if not os.path.isdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount))
        path = utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)
        with open(path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(SegmentStatistical.get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())
            f.flush()
        data = pd.read_csv(path, index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount), index=False)


class DataSetBrueser(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4):
        self.sqi_threshold = sqi_threshold
        super(DataSetBrueser, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, ecg_hr_std, brueser_sqi, bcg_hr):
        indices = np.where(np.logical_and(start < series.bcg.unique_peaks, series.bcg.unique_peaks < end))
        return SegmentBrueserSQI(
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative,
            threshold=self.sqi_threshold,
            medians=series.bcg.medians[indices],
            qualities=series.bcg.brueser_sqi[indices],
            length_samples=utils.seconds_to_frames(self.segment_length, series.bcg_sample_rate)
        )

    def save_csv(self):
        """Saves all segments as csv
        """
        if not os.path.isdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount))
        path = utils.get_brueser_features_csv_path(self.segment_length, self.overlap_amount, self.sqi_threshold,
                                                   self.hr_threshold)
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(SegmentBrueserSQI.get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())
            file.flush()
        data = pd.read_csv(path, index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(utils.get_brueser_features_csv_path(self.segment_length, self.overlap_amount, self.sqi_threshold),
                    index=False)


class DataSetPino(DataSet):

    def __init(self, segment_length=5, overlap_amount=0.8, hr_threshold=10):
        super(DataSetPino, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, ecg_hr_std, brueser_sqi, bcg_hr):
        return SegmentPino(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative,
        )

    def save_csv(self):
        if not os.path.isdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount))
        path = utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)
        with open(path, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(SegmentPino.get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())
            file.flush()
        data = pd.read_csv(path, index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount),
                    index=False)


class Segment:
    """A segment of bcg data without any features yet
    """

    def __init__(self, patient_id, ecg_hr, bcg_hr, brueser_sqi, informative):
        self.brueser_sqi = brueser_sqi
        self.patient_id = patient_id
        self.informative = informative
        self.ecg_hr = ecg_hr
        self.bcg_hr = bcg_hr
        self.abs_err = np.abs(ecg_hr - bcg_hr)
        self.rel_err = 100 / ecg_hr * self.abs_err

    @staticmethod
    def get_feature_name_array():
        return np.array([
            'brueser_sqi',
            'patient_id',
            'informative',
            'ecg_hr',
            'bcg_hr',
            'abs_err',
            'rel_err'
        ])

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        return np.array([
            self.brueser_sqi,
            self.patient_id,
            self.informative,
            self.ecg_hr,
            self.bcg_hr,
            self.abs_err,
            self.rel_err,
        ])


class SegmentPino(Segment):

    def __init__(self, raw_data, patient_id, ecg_hr, bcg_hr, brueser_sqi, sample_rate, informative):
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative)
        self.bcg = raw_data
        minimum = np.min(self.bcg)
        maximum = np.max(self.bcg)
        self.t1 = (maximum + minimum) / 2
        mean = np.mean(self.bcg)
        std = np.std(self.bcg)
        self.t2 = mean + 1.1 * std

    @staticmethod
    def get_feature_name_array():
        segment_array = Segment.get_feature_name_array()
        own_array = np.array([
            'T1',
            'T2'
        ])
        return np.concatenate((segment_array, own_array), axis=0)

    def get_feature_array(self):
        segment_array = super().get_feature_array()
        own_array = np.array([
            self.t1,
            self.t2
        ])
        return np.concatenate((segment_array, own_array), axis=0)


class SegmentStatistical(Segment):
    """
    A segment of bcg data with its statistical features based on the paper 'Sensor data quality processing for
    vital signs with opportunistic ambient sensing' (https://ieeexplore.ieee.org/document/7591234)
    """

    def __init__(self, raw_data, patient_id, ecg_hr, bcg_hr, brueser_sqi, sample_rate, informative):
        """
        Creates a segment and computes several statistical features
        :param raw_data: raw BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        :param coverage:
        :param mean_error: mean BBI error to reference
        """
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative)
        self.bcg = utils.butter_bandpass_filter(raw_data, 1, 12, sample_rate)
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
        self.mean_signal_envelope = SegmentStatistical._calc_mean_signal_envelope(self.bcg)

    @staticmethod
    def _calc_mean_signal_envelope(signal):
        """
        :return: mean of upper envelope of rectified signal
        """
        analytic_signal = hilbert(np.abs(signal))
        amplitude_envelope = np.abs(analytic_signal)
        return np.mean(amplitude_envelope)


    @staticmethod
    def get_feature_name_array():
        segment_array = Segment.get_feature_name_array()
        own_array = np.array([
            'minimum',
            'maximum',
            'mean',
            'std',
            'range',
            'iqr',
            'mad',
            'number_zero_crossings',
            'kurtosis',
            'skewness',
            'variance_local_maxima',
            'variance_local_minima',
            'mean_signal_envelope',
        ])
        return np.concatenate((segment_array, own_array), axis=0)

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        segment_array = super().get_feature_array()
        own_array = np.array([
            self.minimum,
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
        ])
        return np.concatenate((segment_array, own_array), axis=0)


class SegmentBrueserSQI(Segment):

    def __init__(self, patient_id, ecg_hr, bcg_hr, brueser_sqi, informative, threshold, medians, qualities,
                 sample_rate, length_samples):
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative)
        indices = np.argwhere(qualities >= threshold)
        self.sqi_hr = np.nan
        self.sqi_coverage = np.nan
        if len(indices) > 0:
            mean_interval_length = np.mean(medians[indices])
            self.sqi_hr = 60 / (mean_interval_length / sample_rate)
            self.sqi_coverage = 100 / length_samples * mean_interval_length * len(indices)
            if self.sqi_coverage > 100:
                self.sqi_coverage = 100

    @staticmethod
    def get_feature_name_array():
        segment_array = Segment.get_feature_name_array()
        own_array = np.array([
            'sqi_hr',
            'sqi_coverage'
        ])
        return np.concatenate((segment_array, own_array), axis=0)

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        segment_array = super().get_feature_array()
        own_array = np.array([
            self.sqi_hr,
            self.sqi_coverage
        ])
        return np.concatenate((segment_array, own_array), axis=0)


if __name__ == "__main__":
    pass
