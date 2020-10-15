import json
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

import utils as utils
from data_processing import get_ecg_processing, get_brueser_from_id


class BCGSeries:
    coverage_threshold = 80

    def __init__(self, bcg_id, raw_data, sqi, bbi_bcg, bbi_ecg, indices, data_folder, sample_rate=100):
        self.bcg_id = bcg_id
        self.raw_data = raw_data
        self.filtered_data = utils.butter_bandpass_filter(raw_data, 1, 12, sample_rate)
        self.sqi = sqi
        self.bbi_bcg = bbi_bcg
        self.bbi_ecg = bbi_ecg
        self.indices = indices
        self.sample_rate = sample_rate
        self.length = len(raw_data) / sample_rate  # in seconds
        self.brueser_df = get_brueser_from_id(self.sample_rate, self.bcg_id, data_folder=data_folder).dropna()
        self.medians = self.brueser_df['medians'].to_numpy()
        self.unique_peaks = self.brueser_df['unique_peaks'].to_numpy()
        self.brueser_sqi = self.brueser_df['qualities'].to_numpy()

    def get_hr(self, start, end):
        """Calculates heart rate in given interval by calculating the mean length of the detected intervals
        """
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        if len(indices) > 0 and self.get_coverage(start, end) > self.coverage_threshold:
            hr = 60 / (np.median(self.medians[indices]) / self.sample_rate)
        else:
            hr = np.nan
        return hr

    def get_mean_sqi(self, start, end):
        """Returns mean brueser sqi in given interval
        """
        sqi_array = self.get_sqi_array(start, end)
        if len(sqi_array[np.isfinite(sqi_array)]) == 0:
            return 0
        return np.mean(self.get_sqi_array(start, end))

    def get_sqi_array(self, start, end):
        """Returns brueser sqis in given interval
        """
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        return self.brueser_sqi[indices]

    def get_unique_peak_locations(self, start, end):
        """Returns the locations of the unique peaks in the given interval
        """
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        return self.unique_peaks[indices]

    def get_unique_peak_values(self, start, end):
        """Returns the values at the unique peaks in the given window"""
        return self.filtered_data[self.get_unique_peak_locations(start, end)]

    def get_filtered_signal(self, start, end):
        """Returns filtered bcg in given window"""
        return self.filtered_data[start:end]

    def get_interval_lengths(self, start, end):
        """Returns estimated interval lengths in given window"""
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        return self.medians[indices]

    def get_coverage(self, start, end):
        """Returns coverage on given interval
        """
        est_lengths = self.get_interval_lengths(start, end)
        coverage = np.sum(est_lengths) / (end - start)
        if coverage >= 1:
            return 100
        else:
            return coverage * 100


class ECGSeries:

    def __init__(self, patient_id, r_peaks, length, sample_rate=1000):
        self.sample_rate = sample_rate
        self.r_peaks = r_peaks
        self.length = length / sample_rate  # in seconds
        self.patient_id = patient_id

    def get_hr(self, start, end, lower_threshold=30, upper_threshold=200):
        """Calculates heart rate in given interval by calculating the mean length of the detected intervals.
        It chooses the lead with the lowest difference to the hr of a slightly bigger area.
        """
        curr_hr = []
        curr_hr_env = []
        start_env = start - round(1.5 * (end - start))
        end_env = end + round(1.5 * (end - start))
        interval_ranges = []
        for r_peaks_single in self.r_peaks.transpose():
            indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
            indices_env = np.argwhere(np.logical_and(start_env <= r_peaks_single, r_peaks_single < end_env))
            interval_lengths = [r_peaks_single[indices[i]] - r_peaks_single[indices[i - 1]] for i in
                                range(1, len(indices))]
            interval_lengths_env = [r_peaks_single[indices_env[i]] - r_peaks_single[indices_env[i - 1]] for i in
                                    range(1, len(indices_env))]
            if interval_lengths:
                hr_guess = 60 / (np.median(interval_lengths) / self.sample_rate)
                hr_guess_env = np.median(interval_lengths_env)
                lower_threshold_count = lower_threshold * (end - start) / (self.sample_rate * 60)
                upper_threshold_count = upper_threshold * (end - start) / (self.sample_rate * 60)
                if lower_threshold < hr_guess < upper_threshold and lower_threshold_count < len(
                        indices) < upper_threshold_count:
                    curr_hr.append(hr_guess)
                    curr_hr_env.append(hr_guess_env)
                    interval_ranges.append(np.max(interval_lengths_env) - np.min(interval_lengths_env))
        if curr_hr:
            id_min_range = np.argwhere(interval_ranges == np.min(interval_ranges))
            hr_env = np.median(np.array(curr_hr_env)[id_min_range])
            id_min_diff = np.argwhere(np.abs(curr_hr - hr_env) == np.min(np.abs(curr_hr - hr_env)))
            hr = np.median(np.array(curr_hr)[id_min_diff])
        else:
            hr = np.nan
        return hr


class DataSeries:
    bcg_sample_rate = 100
    reference_threshold = 90

    def __init__(self, ecg_series: ECGSeries):
        self.ecg = ecg_series
        self.bcg = None
        self.patient_id = ecg_series.patient_id
        self.drift = None

    def reference_exists(self, bcg_start, bcg_end) -> bool:
        """Checks if more than reference_threshold % of the values are not nan
        """
        start_second = np.floor(bcg_start / self.bcg_sample_rate)
        end_second = np.floor(bcg_end / self.bcg_sample_rate)
        area = self.drift.loc[start_second:end_second]
        if len(area.index.values) == 0 or 100 / len(area.index.values) * area.count() < self.reference_threshold:
            return False
        if np.isnan(self.get_ecg_hr(bcg_start, bcg_end)):
            return False
        return True

    def get_ecg_area(self, bcg_start, bcg_end):
        """Returns corresponding ecg indices for the given bcg area
        """
        start_second = np.floor(bcg_start / self.bcg_sample_rate)
        end_second = np.floor(bcg_end / self.bcg_sample_rate)
        area = self.drift.loc[start_second:end_second].dropna()  # end incl.
        diff = np.mean(area.values - area.index.values)
        start_ecg = np.floor((start_second + diff) * self.ecg.sample_rate)
        end_ecg = start_ecg + (bcg_end - bcg_start) * (self.ecg.sample_rate / self.bcg_sample_rate)
        return start_ecg, end_ecg

    def get_first_reference_index(self):
        """Returns first bcg index, where reference exists
        """
        return self.drift.first_valid_index() * self.ecg.sample_rate

    def get_last_reference_index(self):
        """Returns first bcg index, where reference exists
        """
        return self.drift.last_valid_index() * self.ecg.sample_rate

    def get_ecg_hr(self, bcg_start, bcg_end):
        ecg_start, ecg_end = self.get_ecg_area(bcg_start=bcg_start, bcg_end=bcg_end)
        return self.ecg.get_hr(ecg_start, ecg_end)

    def get_bcg_hr(self, bcg_start, bcg_end):
        return self.bcg.get_hr(bcg_start, bcg_end)

    def get_mean_brueser_sqi(self, bcg_start, bcg_end):
        return self.bcg.get_mean_sqi(bcg_start, bcg_end)

    def is_informative(self, bcg_start, bcg_end, threshold):
        """Returns if signal is informative on given window depending on threshold
        :param bcg_start: sample of bcg signal where window starts
        :param bcg_end: sample of bcg signal where window ends
        :param threshold: threshold for relative error in percent, abs threshold is threshold/2
        """
        if self.get_error(bcg_start, bcg_end) > threshold:
            return False
        return True

    def get_error(self, bcg_start, bcg_end):
        """Returns relative error in given window.

        If heart rate is under 50, relative error is abs_err/2"""
        bcg_hr = self.get_bcg_hr(bcg_start, bcg_end)
        ecg_hr = self.get_ecg_hr(bcg_start, bcg_end)
        abs_err = np.abs(ecg_hr - bcg_hr)
        rel_err = 100 / ecg_hr * abs_err
        if np.isnan(bcg_hr):
            return 667  # max possible error
        if ecg_hr > 50:  # if error in percentage is larger
            return rel_err
        else:
            return abs_err / 0.5

    def get_best_est_int(self, bcg_start, bcg_end):
        idx = self.bcg.get_unique_peak_locations(bcg_start, bcg_end)
        if len(idx) == 0:
            return None
        sqis = self.bcg.get_sqi_array(bcg_start, bcg_end)
        interval_lengths = self.bcg.get_interval_lengths(bcg_start, bcg_end)
        if len(np.isfinite(sqis) > 0):
            max_id = np.argmax(sqis)
            return self.bcg.filtered_data[idx[max_id]:int(idx[max_id] + interval_lengths[max_id])]
        return None

    def get_median_est_int(self, bcg_start, bcg_end):
        idx = self.bcg.get_unique_peak_locations(bcg_start, bcg_end)
        if len(idx) == 0:
            return None
        sqis = self.bcg.get_sqi_array(bcg_start, bcg_end)
        interval_lengths = self.bcg.get_interval_lengths(bcg_start, bcg_end)
        if len(np.isfinite(sqis) > 0):
            median_id = np.argwhere(np.median(sqis))
            return self.bcg.filtered_data[idx[median_id]:int(idx[median_id] + interval_lengths[median_id])]
        return None


class Data:
    sample_rate = 100

    def __init__(self, data_folder='data_patients'):
        """
        :param data_folder: folder_name of all data
        """
        self.data_folder = data_folder
        self.mapping = self._load_mapping()
        self.data_series = {}
        self._create_data_series()
        self._load_bcg_data()
        self._load_drift_compensation()

    def _load_drift_compensation(self):
        paths = [path for path in os.listdir(utils.get_drift_path(self.data_folder)) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(utils.get_drift_path(self.data_folder), path) for path in paths]
        for path in paths:
            mat_dict = loadmat(path)
            patient_id = path.lower().split("_")[-1].replace(".mat", "")
            drift = pd.Series(index=mat_dict['t_bcg_samp'][0], data=mat_dict['t_ecg_corresp'][0])
            self.data_series[patient_id].drift = drift

    def _load_mapping(self):
        """
        :return: mapping from bcg to ecg
        """
        path = os.path.join(utils.get_data_path(self.data_folder), 'mapping.json')
        with open(path, encoding='utf-8') as file:
            mapping = json.load(file)
        return mapping

    def _create_data_series(self):
        paths = [path for path in os.listdir(utils.get_ecg_data_path(self.data_folder)) if
                 path.lower().endswith(".edf")]
        for path in paths:
            path = os.path.join(utils.get_ecg_data_path(self.data_folder), path)
            r_peaks, ecg_id, sample_rate, length = get_ecg_processing(path=path, use_existing=True,
                                                                      data_folder=self.data_folder)
            self.data_series[ecg_id] = DataSeries(ECGSeries(
                patient_id=ecg_id,
                r_peaks=r_peaks.to_numpy(),
                length=length,
                sample_rate=sample_rate
            )
            )

    def _load_bcg_data(self):
        """Loads all bcg data
        :return: array of found bcg_series
        """
        paths = [path for path in os.listdir(utils.get_bcg_data_path(self.data_folder)) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(utils.get_bcg_data_path(self.data_folder), path) for path in paths]
        for path in paths:
            mat_dict = loadmat(path)
            bcg_id = path.lower().split("_")[-1].replace(".mat", "")
            if self.data_folder == 'data_patients' and bcg_id == '14':  # skip file without drift vector
                continue
            bcg = BCGSeries(
                raw_data=mat_dict['BCG_raw_data'][0],
                sqi=mat_dict['q_BCG'][:, 0],
                bbi_bcg=mat_dict['BBI_BCG'][:, 0],
                bbi_ecg=mat_dict['BBI_ECG'][:, 0],
                indices=mat_dict['indx'][:, 0],
                sample_rate=self.sample_rate,
                bcg_id=bcg_id,
                data_folder=self.data_folder
            )
            ecg_id = self.mapping[bcg_id]
            self.data_series[str(ecg_id)].bcg = bcg

    def get_total_time(self):
        """
        :return: total recorded time in hours
        """
        time = np.array([[bcg_series.length for bcg_series in ecg_series] for ecg_series in self.data_series]).sum()
        return time / 60 / 60
