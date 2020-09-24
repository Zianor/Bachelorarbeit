from scipy.io import loadmat
import src.utils as utils
from src.data_processing import ecg_csv, get_brueser
import pandas as pd
import numpy as np
import os
import json


class BCGSeries:

    def __init__(self, patient_id, ecg_id, raw_data, sqi, bbi_bcg, bbi_ecg, indices, sample_rate=100):
        self.bcg_id = patient_id
        self.ecg_id = ecg_id
        self.raw_data = raw_data
        self.sqi = sqi
        self.bbi_bcg = bbi_bcg
        self.bbi_ecg = bbi_ecg
        self.indices = indices
        self.sample_rate = sample_rate
        self.length = len(raw_data) / sample_rate  # in seconds
        self.brueser_df = get_brueser(self.sample_rate, self.bcg_id)
        self.medians = self.brueser_df['medians'].to_numpy()
        self.unique_peaks = self.brueser_df['unique_peaks'].to_numpy()
        self.brueser_sqi = self.brueser_df['qualities'].to_numpy()

    def get_hr(self, start, end):
        """Calculates heartrate in given interval by calculating the mean length of the detected intervals
        """
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        if len(indices) > 0:
            hr = 60 / (np.mean(self.medians[indices]) / self.sample_rate)
        else:
            hr = np.nan
        return hr

    def get_sqi(self, start, end):
        """Returns mean brueser sqi in given interval"""
        indices = np.where(np.logical_and(start <= self.unique_peaks, self.unique_peaks < end))
        sqi = np.mean(self.brueser_sqi[indices])
        return sqi


class ECGSeries:

    def __init__(self, patient_id, r_peaks, length, sample_rate=1000):
        self.sample_rate = sample_rate
        self.r_peaks = r_peaks
        self.length = length / sample_rate  # in seconds
        self.patient_id = patient_id

    def get_hr(self, start, end, lower_threshold=30, upper_threshold=200):
        """Calculates heartrate in given interval by calculating the mean length of the detected intervals.
        It calculates the mean of all leads
        """
        curr_hr = []
        for r_peaks_single in self.r_peaks.transpose():
            indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
            if len(indices) > 1:
                hr_guess = (len(indices) - 1) / (
                        r_peaks_single[indices[-1]] - r_peaks_single[indices[0]]) * self.sample_rate * 60
                lower_threshold_count = lower_threshold * (end - start) / (self.sample_rate * 60)
                upper_threshold_count = upper_threshold * (end - start) / (self.sample_rate * 60)
                if lower_threshold < hr_guess < upper_threshold and lower_threshold_count < len(
                        indices) < upper_threshold_count:
                    curr_hr.append(hr_guess)
        if curr_hr:
            hr = np.mean(curr_hr)
        else:
            hr = 0
        return hr

    def get_hr_std(self, start, end, lower_threshold=30, upper_threshold=200):
        """Calculates heartrate in given interval by calculating the mean length of the detected intervals.
        It chooses lead with lowest std.
        """
        curr_hr = []
        curr_std = []
        for r_peaks_single in self.r_peaks.transpose():
            indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
            interval_lengths = [r_peaks_single[indices[i]] - r_peaks_single[indices[i - 1]] for i in
                                range(1, len(indices))]
            if interval_lengths:
                hr_guess = np.mean(interval_lengths) / self.sample_rate * 60
                std = np.std(interval_lengths)
                lower_threshold_count = lower_threshold * (end - start) / (self.sample_rate * 60)
                upper_threshold_count = upper_threshold * (end - start) / (self.sample_rate * 60)
                if lower_threshold < hr_guess < upper_threshold and lower_threshold_count < len(
                        indices) < upper_threshold_count:
                    curr_hr.append(hr_guess)
                    curr_std.append(std)
        if curr_hr:
            i = np.argmin(curr_std)
            hr = curr_hr[i]
        else:
            hr = 0
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
        if len(area.index.values) == 0 or 100 / len(
                area.index.values) * area.count() < self.reference_threshold or self.get_ecg_hr(bcg_end, bcg_end) == 0:
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

    def get_ecg_hr_std(self, bcg_start, bcg_end):
        ecg_start, ecg_end = self.get_ecg_area(bcg_start=bcg_start, bcg_end=bcg_end)
        return self.ecg.get_hr_std(ecg_start, ecg_end)

    def get_bcg_hr(self, bcg_start, bcg_end):
        return self.bcg.get_hr(bcg_start, bcg_end)

    def get_brueser_sqi(self, bcg_start, bcg_end):
        return self.bcg.get_sqi(bcg_start, bcg_end)


class Data:
    sample_rate = 100

    def __init__(self):
        self.mapping = Data.load_mapping()
        self.data_series = {}
        self.create_data_series()
        self.load_bcg_data()
        self.load_drift_compensation()

    def load_drift_compensation(self):
        paths = [path for path in os.listdir(utils.get_drift_path()) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(utils.get_drift_path(), path) for path in paths]
        for path in paths:
            mat_dict = loadmat(path)
            patient_id = path.lower().split("_")[-1].replace(".mat", "")
            drift = pd.Series(index=mat_dict['t_bcg_samp'][0], data=mat_dict['t_ecg_corresp'][0])
            self.data_series[patient_id].drift = drift

    @staticmethod
    def load_mapping():
        """
        :return: mapping from bcg to ecg
        """
        path = os.path.join(utils.get_data_path(), 'mapping.json')
        with open(path, encoding='utf-8') as file:
            mapping = json.load(file)
        return mapping

    def create_data_series(self):
        paths = [path for path in os.listdir(utils.get_ecg_data_path()) if
                 path.lower().endswith(".edf")]
        for path in paths:
            path = os.path.join(utils.get_ecg_data_path(), path)
            r_peaks, ecg_id, sample_rate, length = ecg_csv(path=path, use_existing=True)
            self.data_series[ecg_id] = DataSeries(ECGSeries(
                patient_id=ecg_id,
                r_peaks=r_peaks.to_numpy(),
                length=length,
                sample_rate=sample_rate
            )
            )

    def load_bcg_data(self):
        """Loads all bcg data
        :return: array of found bcg_series
        """
        paths = [path for path in os.listdir(utils.get_bcg_data_path()) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(utils.get_bcg_data_path(), path) for path in paths]
        for path in paths:
            mat_dict = loadmat(path)
            bcg_id = path.lower().split("_")[-1].replace(".mat", "")
            if bcg_id == '14':  # skip file without drift vector
                continue
            bcg = BCGSeries(
                ecg_id=self.mapping[bcg_id],
                raw_data=mat_dict['BCG_raw_data'][0],
                sqi=mat_dict['q_BCG'][:, 0],
                bbi_bcg=mat_dict['BBI_BCG'][:, 0],
                bbi_ecg=mat_dict['BBI_ECG'][:, 0],
                indices=mat_dict['indx'][:, 0],
                sample_rate=self.sample_rate,
                patient_id=bcg_id
            )
            ecg_id = bcg.ecg_id
            self.data_series[str(ecg_id)].bcg = bcg

    def get_total_time(self):
        """
        :return: total recorded time in hours
        """
        time = np.array([[bcg_series.length for bcg_series in ecg_series] for ecg_series in self.data_series]).sum()
        return time / 60 / 60
