from scipy.io import loadmat
from src.utils import get_project_root
import os


class DataSeries:

    def __init__(self, raw_data, sqi, bbi_bcg, bbi_ecg, indices, samplerate):
        self.raw_data = raw_data
        self.sqi = sqi
        self.bbi_bcg = bbi_bcg
        self.bbi_ecg = bbi_ecg
        self.indices = indices
        self.samplerate = samplerate

    def calculate_bbi_error(self):
        return abs(self.bbi_bcg - self.bbi_ecg)


class BcgData:
    samplerate = 100

    def __init__(self):
        paths = [path for path in os.listdir(os.path.join(get_project_root(), 'data/')) if
                 path.lower().endswith(".mat")]
        paths = [os.path.join(os.path.join(get_project_root(), 'data/'), path) for path in paths]
        self.data_series = []
        for path in paths:
            mat_dict = loadmat(path)
            self.add_file_to_data(mat_dict)

    def add_file_to_data(self, mat_dict):
        """
        Adds the content of dict of bcg data to data_series
        """
        self.data_series.append(
            DataSeries(
                mat_dict['BCG_raw_data'][0],
                mat_dict['q_BCG'][:, 0],
                mat_dict['BBI_BCG'][:, 0],
                mat_dict['BBI_ECG'][:, 0],
                mat_dict['indx'][:, 0],
                self.samplerate
            )
        )
