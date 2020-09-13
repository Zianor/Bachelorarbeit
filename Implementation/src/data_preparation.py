from scipy.io import loadmat
from src.utils import get_project_root
from data_processing import ecg_csv, get_brueser
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


class ECGSeries:

    def __init__(self, patient_id, r_peaks, length, sample_rate=1000):
        self.sample_rate = sample_rate
        self.r_peaks = r_peaks
        self.length = length / sample_rate  # in seconds
        self.patient_id = patient_id


class DataSeries:

    def __init__(self, ecg_series: ECGSeries):
        self.ecg = ecg_series
        self.bcg_series = {}
        self.patient_id = ecg_series.patient_id

    def add_bcg(self, bcg: BCGSeries):
        if bcg not in self.bcg_series:
            self.bcg_series[bcg.bcg_id] = bcg


class Data:
    sample_rate = 100

    def __init__(self):
        self.mapping = Data.load_mapping()
        bcg_series = self.load_bcg_data()
        self.data_series = {}
        self.create_data_series()
        for series in bcg_series:
            curr_id = series.ecg_id
            self.data_series[str(curr_id)].add_bcg(series)

    @staticmethod
    def load_mapping():
        """
        :return: mapping from bcg to ecg
        """
        path = os.path.join(get_project_root(), 'data/mapping.json')
        with open(path, encoding='utf-8') as file:
            mapping = json.load(file)
        return mapping

    def create_data_series(self):
        data_path = os.path.join(get_project_root(), 'data/ecg/')
        paths = [path for path in os.listdir(data_path) if
                 path.lower().endswith(".edf")]
        for path in paths:
            path = os.path.join(os.path.join(get_project_root(), 'data/ecg/'), path)
            r_peaks, ecg_id, sample_rate, length = ecg_csv(data_path=data_path, path=path, use_existing=True)
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
        paths = [path for path in os.listdir(os.path.join(get_project_root(), 'data/bcg/ml_data')) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(os.path.join(get_project_root(), 'data/bcg/ml_data'), path) for path in paths]
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
