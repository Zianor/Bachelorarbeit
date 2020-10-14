import csv
import os

import numpy as np
import pandas as pd
import scipy
from ecgdetectors import Detectors
from pyedflib import highlevel
from scipy import signal
from scipy.io import loadmat

import brueser
import utils

from multiprocessing import Pool
import multiprocessing as mp


def brueser_process_all(fs, use_existing=True, data_folder='data_patients'):
    paths = [path for path in os.listdir(utils.get_bcg_data_path(data_folder)) if
             path.lower().endswith(".mat")]
    pool = Pool(10)
    for path in paths:
        path = os.path.join(utils.get_bcg_data_path(data_folder), path)
        pool.apply_async(get_brueser, kwds={'fs': fs, 'path': path, 'use_existing': use_existing})
        # brueser_csv(fs=fs, path=path, use_existing=use_existing)
    pool.close()
    pool.join()


def get_brueser(fs, path, use_existing=True, data_folder='data_patients'):
    """
    :return: Dataframe of unique_peaks, medians and qualities, where medians is the estimated length
    :rtype: pandas.Dataframe
    """
    data = loadmat(path)['BCG_raw_data'][0]
    number = path.lower().split("_")[-1]
    number = number.replace("mat", "")
    filename = 'brueser' + str(number) + 'csv'
    path_csv = os.path.join(utils.get_brueser_path(data_folder), filename)

    if not use_existing or not os.path.isfile(path_csv):
        win = np.arange(0.3 * fs, 2 * fs + 1, dtype=np.int32)
        data = brueser.preprocess(data, fs)
        result, est_len, quality_arr = brueser.interval_probabilities(data, win, estimate_lengths=True)
        peaks, _ = scipy.signal.find_peaks(data, distance=win[0])
        unique_peaks, medians, qualities = brueser.rr_intervals_from_est_len(est_len, peaks, data, quality_arr,
                                                                             win[0])
        with open(path_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["unique_peaks", "medians", "qualities"])
            for i, peak in enumerate(unique_peaks):
                writer.writerow([peak, medians[i], qualities[i]])
    return pd.read_csv(path_csv)


def get_brueser_from_id(fs, brueser_id, use_existing=True, data_folder='data_patients'):
    filename = 'ML_data_patient_' + brueser_id + '.mat'
    path = os.path.join(utils.get_bcg_data_path(data_folder), filename)
    return get_brueser(fs=fs, path=path, use_existing=use_existing)


def ecg_process_all(use_existing=True, data_folder='data_patients'):
    paths = [path for path in os.listdir(utils.get_ecg_data_path(data_folder)) if path.lower().endswith(".edf")]
    for path in paths:
        path = os.path.join(utils.get_ecg_data_path(data_folder), path)
        get_ecg_processing(path=path, use_existing=use_existing)


def get_ecg_processing(path, use_existing=True, data_folder='data_patients'):
    ecg_id = path.lower().split("_")[-1].replace(".edf", "")
    filename = 'rpeaks' + str(ecg_id) + '.csv'
    path_csv = os.path.join(utils.get_rpeaks_path(data_folder), filename)

    length = None
    sample_rate = None

    if not use_existing or not os.path.isfile(path_csv):
        signals, signal_headers, header = highlevel.read_edf(path)
        r_peaks = {}
        for i, s in enumerate(signals):
            if signal_headers[i]['transducer'] == 'ECG electrode':
                sample_rate = signal_headers[i]['sample_rate']
                length = len(s)
                detectors = Detectors(sample_rate)
                r_peaks[signal_headers[i]['label']] = detectors.two_average_detector(s)
        r_peaks_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in r_peaks.items()]))
        r_peaks_data.to_csv(path_csv)
    r_peaks_data = pd.read_csv(path_csv)

    if (not sample_rate) or (not length):
        signals, signal_headers, header = highlevel.read_edf(path)
        for i, s in enumerate(signals):
            if signal_headers[i]['transducer'] == 'ECG electrode':
                sample_rate = signal_headers[i]['sample_rate']
                length = len(s)
                break

    return r_peaks_data, ecg_id, sample_rate, length


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    brueser_process_all(fs=100, use_existing=True, data_folder='data_healthy')
    pass
