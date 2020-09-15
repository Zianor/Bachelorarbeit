from scipy.io import loadmat
import utils
from data_processing import ecg_csv, get_brueser
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
        self.medians = self.brueser_df['medians']
        self.unique_peaks = self.brueser_df['unique_peaks']


class ECGSeries:

    def __init__(self, patient_id, r_peaks, length, sample_rate=1000):
        self.sample_rate = sample_rate
        self.r_peaks = r_peaks
        self.length = length / sample_rate  # in seconds
        self.patient_id = patient_id


class DataSeries:
    bcg_sample_rate = 100
    reference_threshold = 90

    def __init__(self, ecg_series: ECGSeries):
        self.ecg = ecg_series
        self.bcg_series = {}
        self.patient_id = ecg_series.patient_id
        self.drift = None

    def add_bcg(self, bcg: BCGSeries):
        if bcg not in self.bcg_series:
            self.bcg_series[bcg.bcg_id] = bcg

    def reference_exists(self, bcg_start, bcg_end) -> bool:
        """Checks if more than reference_threshold % of the values are not nan
        """
        start_second = np.floor(bcg_start / self.bcg_sample_rate)
        end_second = np.floor(bcg_end / self.bcg_sample_rate)
        area = self.drift.loc[start_second:end_second]
        if len(area.index.values) == 0 or 100/len(area.index.values) * area.count() < self.reference_threshold:
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


class Data:
    sample_rate = 100

    def __init__(self):
        self.mapping = Data.load_mapping()
        bcg_series = self.load_bcg_data()
        self.data_series = {}
        self.create_data_series()
        for series in bcg_series:  # TODO: do it while loading?
            curr_id = series.ecg_id
            self.data_series[str(curr_id)].add_bcg(series)
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
        bcg_series = []
        for path in paths:
            mat_dict = loadmat(path)
            bcg_id = path.lower().split("_")[-1].replace(".mat", "")
            bcg_series.append(
                BCGSeries(
                    ecg_id=self.mapping[bcg_id],
                    raw_data=mat_dict['BCG_raw_data'][0],
                    sqi=mat_dict['q_BCG'][:, 0],
                    bbi_bcg=mat_dict['BBI_BCG'][:, 0],
                    bbi_ecg=mat_dict['BBI_ECG'][:, 0],
                    indices=mat_dict['indx'][:, 0],
                    sample_rate=self.sample_rate,
                    patient_id=bcg_id
                )
            )
        return bcg_series

    def get_total_time(self):
        """
        :return: total recorded time in hours
        """
        time = np.array([[bcg_series.length for bcg_series in ecg_series] for ecg_series in self.data_series]).sum()
        return time/60/60
