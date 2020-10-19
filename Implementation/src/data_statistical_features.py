import csv
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.signal import find_peaks, hilbert, correlate, periodogram
from scipy.stats import median_abs_deviation, kurtosis, skew

import utils
from data_preparation import DataSeries, Data


class DataSet:
    """A data set contains BCG Data in the form of segments of a certain length and overlap. Furthermore a DataSet
    writes a .csv file with the feature representation of all segments.
    """

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients',
                 images=False):
        self.data_folder = data_folder
        self.hr_threshold = hr_threshold
        self.segment_length = segment_length
        self.overlap_amount = overlap_amount
        self.images = images
        self.segments = []
        self._create_segments()
        if self.images:
            self.save_images()
        self.save_csv()

    def _get_path(self):
        return utils.get_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                           self.hr_threshold)

    def _create_segments(self):
        """
        Creates segments with a given length out of given BCG Data and writes them to csv
        """
        if os.path.isfile(utils.get_data_object_path(self.data_folder)):
            with open(utils.get_data_object_path(self.data_folder), 'rb') as file:
                data = pickle.load(file)
        else:
            data = Data(self.data_folder)
            with open(utils.get_data_object_path(self.data_folder), 'wb') as file:
                pickle.dump(data, file)
                file.flush()
        segment_length_frames = utils.seconds_to_frames(self.segment_length, data.sample_rate)  # in samples
        segment_distance = utils.seconds_to_frames(self.segment_length - self.segment_length * self.overlap_amount,
                                                   data.sample_rate)
        if not os.path.isdir(utils.get_data_set_folder(self.data_folder, self.segment_length, self.overlap_amount)):
            os.mkdir(utils.get_data_set_folder(self.data_folder, self.segment_length, self.overlap_amount))
        with open(self._get_path_hr(), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self._get_feature_name_array())
            f.flush()

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
                    segment = self._get_segment(series, start, end, informative, ecg_hr, brueser_sqi, bcg_hr)
                    if self.images:
                        self.segments.append(segment)
                    with open(self._get_path_hr(), 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(segment.get_feature_array())
                        f.flush()

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        if self.images:
            filtered_data = series.bcg.filtered_data[start:end]
        else:
            filtered_data = None
        brueser_coverage = series.bcg.get_coverage(start, end)
        return Segment(
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            informative=informative,
            filtered_data=filtered_data,
            coverage=brueser_coverage
        )

    @staticmethod
    def _get_feature_name_array():
        return Segment.get_feature_name_array()

    def save_csv(self):
        """
        Saves all segments as csv without informative label
        """
        data = pd.read_csv(self._get_path_hr(), index_col=False)
        data = data.drop(labels='informative', axis='columns')
        data.to_csv(self._get_path(), index=False)

    def save_images(self, count=50):
        """
        Saves segments as images
        :param count: number of images saved
        """
        path_images = utils.get_image_folder(self.data_folder, self.segment_length, self.overlap_amount,
                                             self.hr_threshold)
        if not os.path.exists(path_images):
            os.makedirs(path_images)
        count_informative = 0
        count_non_informative = 0
        plt.rcParams.update(utils.get_plt_settings())
        for segment in self.segments:
            if count_non_informative > count and count_informative > count:
                break
            if segment.informative and count_informative < count:
                count_informative += 1
                plt.figure(figsize=utils.get_plt_normal_size())
                plt.plot(segment.bcg)
                plt.title(
                    f"E\\textsubscript{{HR}}={segment.error:.2f}, HR\\textsubscript{{EKG}} = {segment.ecg_hr:.2f}")
                plt.savefig(os.path.join(path_images, 'informative' + str(count_informative) + ".pdf"),
                            transparent=True, bbox_inches='tight', dpi=300)
                plt.clf()
            elif count_non_informative < count and not segment.informative:
                count_non_informative += 1
                plt.figure(figsize=utils.get_plt_normal_size())
                plt.plot(segment.bcg)
                plt.title(
                    f"E\\textsubscript{{HR}}={segment.error:.2f}, HR\\textsubscript{{EKG}} = {segment.ecg_hr:.2f}")
                plt.savefig(os.path.join(path_images, 'non-informative' + str(count_non_informative) + ".pdf"),
                            transparent=True, bbox_inches='tight', dpi=300)
                plt.clf()


class DataSetStatistical(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(DataSetStatistical, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_statistical_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_statistical_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                       self.hr_threshold)

    @staticmethod
    def _get_feature_name_array():
        return SegmentStatistical.get_feature_name_array()

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        brueser_coverage = series.bcg.get_coverage(start, end)
        return SegmentStatistical(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative,
            coverage=brueser_coverage
        )


class DataSetBrueser(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4):
        self.sqi_threshold = sqi_threshold
        super(DataSetBrueser, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_brueser_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                   self.sqi_threshold)

    def _get_path_hr(self):
        return utils.get_brueser_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                   self.sqi_threshold,
                                                   self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        indices = np.where(np.logical_and(start < series.bcg.unique_peaks, series.bcg.unique_peaks < end))
        brueser_coverage = series.bcg.get_coverage(start, end)
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
            length_samples=utils.seconds_to_frames(self.segment_length, series.bcg_sample_rate),
            coverage=brueser_coverage
        )

    @staticmethod
    def _get_feature_name_array():
        return SegmentBrueserSQI.get_feature_name_array()


class DataSetPino(DataSet):

    def __init(self, segment_length=5, overlap_amount=0.8, hr_threshold=10):
        super(DataSetPino, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_pino_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_pino_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                self.hr_threshold)

    def _get_segment(self, series: DataSeries, start, end, informative, ecg_hr, brueser_sqi, bcg_hr):
        brueser_coverage = series.bcg.get_coverage(start, end)
        return SegmentPino(
            raw_data=series.bcg.raw_data[start: end],
            patient_id=series.patient_id,
            ecg_hr=ecg_hr,
            bcg_hr=bcg_hr,
            brueser_sqi=brueser_sqi,
            sample_rate=series.bcg_sample_rate,
            informative=informative,
            coverage=brueser_coverage
        )

    @staticmethod
    def _get_feature_name_array():
        return SegmentPino.get_feature_name_array()


class DataSetOwn(DataSet):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(DataSetOwn, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _get_path(self):
        return utils.get_own_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)

    def _get_path_hr(self):
        return utils.get_own_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                               self.hr_threshold)

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

    @staticmethod
    def _get_feature_name_array():
        return SegmentOwn.get_feature_name_array()


class Segment:
    """A segment of bcg data without any features yet
    """

    def __init__(self, patient_id, ecg_hr, bcg_hr, brueser_sqi, informative, coverage, filtered_data=None):
        self.coverage = coverage
        self.bcg = filtered_data
        self.brueser_sqi = brueser_sqi
        self.patient_id = patient_id
        self.informative = informative
        self.ecg_hr = ecg_hr
        self.bcg_hr = bcg_hr
        self.abs_err = np.abs(ecg_hr - bcg_hr)
        if not np.isfinite(self.abs_err):
            self.abs_err = 170  # max possible error
            self.rel_err = 667  # max possible error
            self.error = 667
        else:
            self.rel_err = 100 / ecg_hr * self.abs_err
            if self.ecg_hr > 50:
                self.error = self.rel_err
            else:
                self.error = self.abs_err / 0.5
        self.quality_class = self.get_quality_class()

    def get_quality_class(self):
        if self.error < 5:
            return 5
        elif self.error < 10:
            return 4
        elif self.error < 15:
            return 3
        elif self.error < 20:
            return 2
        elif np.isclose(self.error, 667):
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
            'quality_class',
            'error',
            'brueser_coverage'
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
            self.quality_class,
            self.error,
            self.coverage
        ])


class SegmentPino(Segment):

    def __init__(self, raw_data, patient_id, ecg_hr, bcg_hr, brueser_sqi, sample_rate, informative, coverage):
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative, coverage=coverage)
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

    def __init__(self, raw_data, patient_id, ecg_hr, bcg_hr, brueser_sqi, sample_rate, informative, coverage):
        """
        Creates a segment and computes several statistical features
        :param raw_data: raw BCG data
        :param informative: boolean indicating if segment is labeled as informative or not
        :param coverage: coverage of brueser intervals
        """
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative, coverage=coverage)
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
                 sample_rate, length_samples, coverage):
        super().__init__(patient_id, ecg_hr, bcg_hr, brueser_sqi, informative, coverage=coverage)
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
        super(SegmentOwn, self).__init__(series.bcg.raw_data[start: end], series.patient_id, ecg_hr, bcg_hr,
                                         brueser_sqi, series.bcg_sample_rate, informative,
                                         coverage=series.bcg.get_coverage(start, end))
        self.interval_lengths = series.bcg.get_interval_lengths(start, end)  # in samples
        self.sqi_array = series.bcg.get_sqi_array(start, end)
        self.peak_values = series.bcg.get_unique_peak_values(start, end)
        self.acf = sm.tsa.acf(self.filtered_data, nlags=(end - start) // 2, fft=True)
        f_acf, den_acf = periodogram(self.acf, fs=series.bcg_sample_rate)
        f_data, den_data = periodogram(self.filtered_data, fs=series.bcg_sample_rate)
        if len(den_data[np.isfinite(den_data)]) > 0:
            self.peak_frequency_data = f_data[np.nanargmax(den_data)]
        else:
            self.peak_frequency_data = np.nan
        if len(den_acf[np.isfinite(den_acf)]) > 0:
            self.peak_frequency_acf = f_data[np.nanargmax(den_acf)]
        else:
            self.peak_frequency_acf = np.nan
        self.hf_ratio_data = self.bcg_hr / self.peak_frequency_data
        self.hf_diff_data = self.bcg_hr - self.peak_frequency_data
        if not np.isfinite(self.hf_ratio_data):
            self.hf_ratio_data = 0
        if not np.isfinite(self.hf_diff_data):
            self.hf_diff_data = 0
        self.hf_ratio_acf = self.bcg_hr / self.peak_frequency_acf
        self.hf_diff_acf = self.bcg_hr - self.peak_frequency_acf
        if not np.isfinite(self.hf_ratio_acf):
            self.hf_ratio_acf = 0
        if not np.isfinite(self.hf_diff_acf):
            self.hf_diff_acf = 0
        self.abs_energy = np.sum(self.filtered_data * self.filtered_data)
        if len(self.interval_lengths) > 0:
            self.sqi_coverage_03 = series.bcg.get_coverage(start, end, sqi_threshold=0.3)
            self.sqi_coverage_04 = series.bcg.get_coverage(start, end, sqi_threshold=0.4)
            self.sqi_coverage_05 = series.bcg.get_coverage(start, end, sqi_threshold=0.5)

            self.interval_means = np.zeros(self.interval_lengths.shape)
            self.interval_stds = np.zeros(self.interval_lengths.shape)
            self.interval_ranges = np.zeros(self.interval_lengths.shape)
            peak_locations = series.bcg.get_unique_peak_locations(start, end)
            for i, peak_location in enumerate(peak_locations):
                interval_length = self.interval_lengths[i]
                curr_end = int(peak_location - start + interval_length)
                if curr_end > end - start:
                    curr_end = end - start
                curr_interval_data = self.filtered_data[
                                     int(peak_location - start): ]
                if len(curr_interval_data) > 0:
                    self.interval_means[i] = np.mean(curr_interval_data)
                    self.interval_stds[i] = np.std(curr_interval_data)
                    self.interval_ranges[i] = np.max(curr_interval_data) - np.min(curr_interval_data)
                else:
                    self.interval_means[i] = np.nan
                    self.interval_stds[i] = np.nan
                    self.interval_ranges[i] = np.nan

            self.interval_means_std = np.std(self.interval_means)
            self.interval_stds_std = np.std(self.interval_stds)
            self.interval_ranges_std = np.std(self.interval_ranges)

            self.interval_lengths_std = np.std(self.interval_lengths)
            self.interval_lengths_range = np.max(self.interval_lengths) - np.min(self.interval_lengths)
            self.interval_lengths_mean = np.mean(self.interval_lengths)
            self.sqi_std = np.std(self.sqi_array)
            self.sqi_max = np.max(self.sqi_array)
            self.sqi_min = np.min(self.sqi_array)
            self.sqi_median = np.median(self.sqi_array)
            self.sqi_mean = np.mean(self.sqi_array)
            self.peak_max = np.max(self.peak_values)
            self.peak_min = np.max(self.peak_values)
            self.peak_std = np.std(self.peak_values)
            self.peak_mean = np.mean(self.peak_values)
            self.template_corrs_highest_sqi = self.get_template_correlations(series.get_best_est_int(start, end),
                                                                             series.bcg.get_unique_peak_locations(start,
                                                                                                                  end),
                                                                             series)
            self.template_corrs_median_sqi = self.get_template_correlations(series.get_median_est_int(start, end),
                                                                            series.bcg.get_unique_peak_locations(start,
                                                                                                                 end),
                                                                            series)
        else:
            self.interval_means_std = np.nan
            self.interval_stds_std = np.nan
            self.interval_ranges_std = np.nan
            self.sqi_coverage_03 = 0
            self.sqi_coverage_04 = 0
            self.sqi_coverage_05 = 0
            self.interval_lengths_std = np.nan
            self.interval_lengths_range = np.nan
            self.interval_lengths_mean = np.nan
            self.sqi_std = np.nan
            self.sqi_max = np.nan
            self.sqi_min = np.nan
            self.sqi_median = np.nan
            self.sqi_mean = np.nan
            self.peak_max = np.nan
            self.peak_min = np.nan
            self.peak_std = np.nan
            self.peak_mean = np.nan
            self.template_corrs_highest_sqi = None
            self.template_corrs_median_sqi = None
        if self.template_corrs_highest_sqi is not None:
            self.template_correlation_highest_sqi_mean = np.mean(self.template_corrs_highest_sqi)
            self.template_correlation_highest_sqi_std = np.std(self.template_corrs_highest_sqi)
            self.template_correlation_median_sqi_mean = np.mean(self.template_corrs_median_sqi)
            self.template_correlation_median_sqi_std = np.std(self.template_corrs_median_sqi)
        else:
            self.template_correlation_highest_sqi_mean = 0
            self.template_correlation_highest_sqi_std = np.nan
            self.template_correlation_median_sqi_mean = 0
            self.template_correlation_median_sqi_std = np.nan

    def get_template_correlations(self, template, peak_locations, series):
        if template is None:
            return None
        correlations = np.zeros(len(self.interval_lengths))
        for i, peak_location in enumerate(peak_locations):
            interval_length = self.interval_lengths[i]
            if interval_length == 0:
                correlations[i] = 0
            else:
                heartbeat = series.bcg.filtered_data[peak_location: int(peak_location + interval_length)]
                try:
                    curr_corr = correlate(template, heartbeat, method='auto')
                    correlations[i] = np.sum(curr_corr) / len(curr_corr)
                except:
                    print(template)
                    print(heartbeat)
                    print(self.interval_lengths)
                    print(peak_location)
                    import traceback
                    traceback.print_exc()
                    SystemExit(-1)
        return correlations

    @staticmethod
    def get_feature_name_array():
        segment_array = SegmentStatistical.get_feature_name_array()
        own_array = np.array([
            'hf_ratio_acf',
            'hf_ratio_data',
            'hf_diff_acf',
            'hf_diff_data',
            'peak_frequency_acf',
            'peak_frequency_data',
            'abs_energy',
            'interval_lengths_std',
            'interval_lengths_range',
            'interval_lengths_mean',
            'sqi_std',
            'sqi_min',
            'sqi_max',
            'sqi_median',
            'sqi_mean',
            'peak_max',
            'peak_min',
            'peak_mean',
            'peak_std',
            'template_corr_highest_sqi_mean',
            'template_corr_highest_sqi_std',
            'template_corr_median_sqi_mean',
            'template_corr_median_sqi_std',
            'interval_means_std',
            'interval_stds_std',
            'interval_ranges_std',
            'sqi_coverage_03',
            'sqi_coverage_04',
            'sqi_coverage_05'
        ])
        return np.concatenate((segment_array, own_array), axis=0)

    def get_feature_array(self):
        """
        :return: array representation of the segment
        """
        segment_array = super().get_feature_array()
        own_array = np.array([
            self.hf_ratio_acf,
            self.hf_ratio_data,
            self.hf_diff_acf,
            self.hf_diff_data,
            self.peak_frequency_acf,
            self.peak_frequency_data,
            self.abs_energy,
            self.interval_lengths_std,
            self.interval_lengths_range,
            self.interval_lengths_mean,
            self.sqi_std,
            self.sqi_min,
            self.sqi_max,
            self.sqi_median,
            self.sqi_mean,
            self.peak_max,
            self.peak_min,
            self.peak_mean,
            self.peak_std,
            self.template_correlation_highest_sqi_mean,
            self.template_correlation_highest_sqi_std,
            self.template_correlation_median_sqi_mean,
            self.template_correlation_median_sqi_std,
            self.interval_means_std,
            self.interval_stds_std,
            self.interval_ranges_std,
            self.sqi_coverage_03,
            self.sqi_coverage_04,
            self.sqi_coverage_05
        ])
        return np.concatenate((segment_array, own_array), axis=0)


if __name__ == "__main__":
    # DataSet()
    DataSetOwn()
    # DataSetStatistical()
    # DataSetPino()
    # DataSetPino(4, 0.75)
    # DataSetBrueser()
    # DataSetBrueser(sqi_threshold=0.3)
    # DataSetBrueser(sqi_threshold=0.2)
    pass
