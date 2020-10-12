import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, hilbert, welch, correlate
from scipy.stats import median_abs_deviation, kurtosis, skew
import statsmodels.api as sm

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

    def _get_path(self):
        return utils.get_features_csv_path(self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)

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
                    brueser_sqi = series.get_mean_brueser_sqi(start, end)
                    bcg_hr = series.get_bcg_hr(start, end)
                    informative = series.is_informative(start, end, self.hr_threshold)
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

    def save_csv(self):
        """
        Saves all segments as csv
        """
        if not os.path.isdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.segment_length, self.overlap_amount))
        with open(self._get_path_hr(), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(self.segments[0].get_feature_name_array())
            for segment in self.segments:
                writer.writerow(segment.get_feature_array())
            f.flush()
        data = pd.read_csv(self._get_path_hr(), index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(self._get_path(), index=False)

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

    def _get_path(self):
        return utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        return SegmentStatistical(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative
        )


class DataSetBrueser(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4):
        self.sqi_threshold = sqi_threshold
        super(DataSetBrueser, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_brueser_features_csv_path(self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_brueser_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
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


class DataSetPino(DataSet):

    def __init(self, segment_length=5, overlap_amount=0.8, hr_threshold=10):
        super(DataSetPino, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        return SegmentPino(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative,
        )


class DataSetOwn(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(DataSetOwn, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_own_features_csv_path(self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_own_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        return SegmentOwn(
            series=series,
            start=start,
            end=end,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            informative=informative
        )


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
        if not np.isfinite(self.abs_err):
            self.abs_err = np.finfo(np.float32).max
            self.rel_err = np.finfo(np.float32).max
        else:
            self.rel_err = 100 / ecg_hr * self.abs_err
        self.quality_class = self.get_quality_class()

    def get_quality_class(self):
        if self.rel_err < 5 or self.abs_err < 2.5:
            return 5
        elif self.rel_err < 10 or self.abs_err < 5:
            return 4
        elif self.rel_err < 15 or self.abs_err < 7.5:
            return 3
        elif self.rel_err < 20 or self.abs_err < 10:
            return 2
        elif self.rel_err == np.finfo(np.float32).max:
            return 0
        else:
            return 1



    @staticmethod
    def get_feature_name_array():
        return np.array([
            'brueser_sqi',
            'patient_id',
            'informative',
            'ecg_hr',
            'bcg_hr',
            'abs_err',
            'rel_err',
            'quality_class'
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
            self.quality_class
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
        self.mad = median_abs_deviation(self.bcg)
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
            self.sqi_coverage = 100 / length_samples * np.sum(medians[indices])
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


class SegmentOwn(SegmentStatistical):

    def __init__(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        self.filtered_data = series.bcg.filtered_data[start: end]
        super(SegmentOwn, self).__init__(series.bcg.raw_data[start: end], series.patient_id, ecg_hr, bcg_hr, brueser_sqi,
                                         series.bcg_sample_rate, informative)
        self.brueser_coverage = series.bcg.get_coverage(start, end)
        self.interval_lengths = series.bcg.get_interval_lengths(start, end)  # in samples
        self.sqi_array = series.bcg.get_sqi_array(start, end)
        self.peak_values = series.bcg.get_unique_peak_values(start, end)
        self.acf = sm.tsa.acf(self.filtered_data, nlags=(end-start)//2, fft=True)
        f, den = welch(self.filtered_data, fs=series.bcg_sample_rate)
        self.peak_frequency_acf = f[np.nanargmax(den)]
        self.hf_ratio_acf = self.bcg_hr / self.peak_frequency_acf
        if self.hf_ratio_acf == np.inf:
            self.hf_ratio_acf = 0
        self.abs_energy = np.sum(self.filtered_data * self.filtered_data)
        if len(self.interval_lengths) > 0:
            self.interval_lengths_std = np.std(self.interval_lengths)
            self.interval_lengths_range = np.max(self.interval_lengths) - np.min(self.interval_lengths)
            self.interval_lengths_mean = np.mean(self.interval_lengths)
            self.sqi_std = np.std(self.sqi_array)
            self.sqi_max = np.max(self.sqi_array)
            self.sqi_min = np.min(self.sqi_array)
            self.peak_max = np.max(self.peak_values)
            self.peak_min = np.max(self.peak_values)
            self.peak_std = np.std(self.peak_values)
            self.peak_mean = np.mean(self.peak_values)
            self.template_correlations = self.get_template_correlations(series.get_best_est_int(start, end),
                                                                        series.bcg.get_unique_peak_locations(start,end),
                                                                        series)
        else:
            self.sqi_std = np.finfo(np.float32).max
            self.sqi_max = np.finfo(np.float32).max
            self.sqi_min = np.finfo(np.float32).max
            self.peak_max = np.finfo(np.float32).max
            self.peak_min = np.finfo(np.float32).max
            self.peak_std = np.finfo(np.float32).max
            self.peak_mean = np.finfo(np.float32).max
            self.template_correlations = None
        if self.template_correlations is not None:
            self.template_correlation_mean = np.mean(self.template_correlations)
            self.template_correlation_std = np.std(self.template_correlations)
        else:
            self.template_correlation_mean = 0
            self.template_correlation_std = np.finfo(np.float32).max

    def get_template_correlations(self, template, peak_loactions, series):
        if template is None:
            return None
        correlations = np.zeros(len(self.interval_lengths))
        for i, peak_loaction in enumerate(peak_loactions):
            interval_length = self.interval_lengths[i]
            if interval_length == 0:
                correlations[i] = 0
            else:
                heartbeat = series.bcg.filtered_data[peak_loaction: int(peak_loaction + interval_length)]
                try:
                    curr_corr = correlate(template, heartbeat, method='auto')
                except:
                    import traceback
                    traceback.print_stack()
                    print(peak_loaction)
                    return None
                correlations[i] = np.sum(curr_corr)/len(curr_corr)
        return correlations




    @staticmethod
    def get_feature_name_array():
        segment_array = SegmentStatistical.get_feature_name_array()
        own_array = np.array([
            'hf_ratio_acf',
            'peak_frequency_acf',
            'brueser_coverage',
            'abs_energy',
            'interval_lengths_std',
            'interval_lengths_range',
            'interval_lengths_mean',
            'sqi_std',
            'sqi_min',
            'sqi_max',
            'peak_max',
            'peak_min',
            'peak_mean',
            'peak_std',
            'template_corr_mean',
            'template_corr_std'
        ])
        return np.concatenate((segment_array, own_array), axis=0)

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        segment_array = super().get_feature_array()
        own_array = np.array([
            self.hf_ratio_acf,
            self.peak_frequency_acf,
            self.brueser_coverage,
            self.abs_energy,
            self.interval_lengths_std,
            self.interval_lengths_range,
            self.interval_lengths_mean,
            self.sqi_std,
            self.sqi_min,
            self.sqi_max,
            self.peak_max,
            self.peak_min,
            self.peak_mean,
            self.peak_std,
            self.template_correlation_mean,
            self.template_correlation_std
        ])
        return np.concatenate((segment_array, own_array), axis=0)


if __name__ == "__main__":
    DataSetOwn()
    DataSetBrueser()
    DataSetStatistical()
    DataSetPino()
    DataSetPino(4, 0.75)
    pass
