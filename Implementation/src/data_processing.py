import csv
import os

import numpy as np
import pandas as pd
import scipy
from ecgdetectors import Detectors
from pyedflib import highlevel
from scipy import signal
from scipy.io import loadmat

import src.brueser as brueser
import src.utils as utils

from multiprocessing import Pool
import multiprocessing as mp


def get_brueser_hr(unique_peaks, medians, segment_length, sample_rate):
    unique_peaks = unique_peaks.to_numpy()
    last_peak = unique_peaks[-1]
    segment_count = last_peak // segment_length
    hr = np.zeros(segment_count)

    for i, _ in enumerate(hr):
        start = i * segment_length
        end = (i + 1) * segment_length
        hr[i] = get_brueser_segment_hr(start, end, unique_peaks, medians, sample_rate)
    return hr


def get_brueser_segment_hr(start, end, unique_peaks, medians, sample_rate):
    """Calculates Heartrate in given interval by calculating the mean of the detected intervals
    """
    indices = np.where(np.logical_and(start <= unique_peaks, unique_peaks < end))
    hr = 60 / (np.mean(medians.to_numpy()[indices]) / sample_rate)
    return hr


def brueser_csv_all(fs, use_existing=True):
    paths = [path for path in os.listdir(utils.get_bcg_data_path()) if
             path.lower().endswith(".mat")]
    # pool = Pool(10)
    for path in paths:
        path = os.path.join(utils.get_bcg_data_path(), path)
        # pool.apply_async(brueser_csv, kwds={'fs': fs, 'path': path, 'use_existing': use_existing})
        brueser_csv(fs=fs, path=path, use_existing=use_existing)
    # pool.close()
    # pool.join()


def brueser_csv(fs, path, use_existing=True):
    """
    :return: Dataframe of unique_peaks, medians and qualities, where medians is the estimated length
    :rtype: pandas.Dataframe
    """
    data = loadmat(path)['BCG_raw_data'][0]
    number = path.lower().split("_")[-1]
    number = number.replace("mat", "")
    filename = 'brueser' + str(number) + 'csv'
    path_csv = os.path.join(utils.get_brueser_path(), filename)
    print(path_csv)

    if not use_existing or not os.path.isfile(path_csv):
        win = np.arange(0.3 * fs, 2 * fs + 1, dtype=np.int32)
        result, est_len, quality_arr = brueser.interval_probabilities(data, win, estimate_lengths=True)
        print(result)
        peaks, _ = scipy.signal.find_peaks(data, distance=win[0])
        unique_peaks, medians, qualities = brueser.rr_intervals_from_est_len(est_len, peaks, data, quality_arr,
                                                                             win[0])
        with open(path_csv, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["unique_peaks", "medians", "qualities"])
            for i, peak in enumerate(unique_peaks):
                writer.writerow([peak, medians[i], qualities[i]])
        print("Wrote " + path_csv)
    return pd.read_csv(path_csv)


def get_brueser(fs, brueser_id, use_existing=True):
    filename = 'ML_data_patient_' + brueser_id + '.mat'
    path = os.path.join(utils.get_bcg_data_path(), filename)
    return brueser_csv(fs=fs, path=path, use_existing=use_existing)


def get_ecg_hr(r_peaks, segment_length, sample_rate, lower_threshold=30, upper_threshold=200):
    last_peak = np.nanmax(r_peaks[-1, :])
    segment_count = int(last_peak // segment_length)
    hr = np.zeros(segment_count)
    for i, _ in enumerate(hr):
        start = i * segment_length
        end = (i + 1) * segment_length
        hr[i] = get_ecg_segment_hr(start, end, r_peaks, sample_rate, lower_threshold, upper_threshold)
    return hr


def get_std_ecg_hr(r_peaks, segment_length, sample_rate, lower_threshold=30, upper_threshold=200):
    last_peak = np.nanmax(r_peaks[-1, :])
    segment_count = int(last_peak // segment_length)
    hr = np.zeros(segment_count)
    for i, _ in enumerate(hr):
        start = i * segment_length
        end = (i + 1) * segment_length
        hr[i] = get_std_ecg_segment_hr(start, end, r_peaks, sample_rate, lower_threshold, upper_threshold)
    return hr


def get_std_ecg_segment_hr(start, end, r_peaks, sample_rate, lower_threshold=30, upper_threshold=200):
    curr_hr = []
    curr_std = []
    for r_peaks_single in r_peaks.transpose():
        indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
        interval_lengths = [r_peaks_single[indices[i]] - r_peaks_single[indices[i - 1]] for i in range(1, len(indices))]
        if interval_lengths:
            hr_guess = np.mean(interval_lengths) / sample_rate * 60
            std = np.std(interval_lengths)
            lower_threshold_count = lower_threshold * (end - start) / (sample_rate * 60)
            upper_threshold_count = upper_threshold * (end - start) / (sample_rate * 60)
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


def get_ecg_segment_hr(start, end, r_peaks, sample_rate, lower_threshold=30, upper_threshold=200):
    curr_hr = []
    for r_peaks_single in r_peaks.transpose():
        indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
        if len(indices) > 1:
            hr_guess = (len(indices) - 1) / (
                    r_peaks_single[indices[-1]] - r_peaks_single[indices[0]]) * sample_rate * 60
            lower_threshold_count = lower_threshold * (end - start) / (sample_rate * 60)
            upper_threshold_count = upper_threshold * (end - start) / (sample_rate * 60)
            if lower_threshold < hr_guess < upper_threshold and lower_threshold_count < len(
                    indices) < upper_threshold_count:
                curr_hr.append(hr_guess)
    if curr_hr:
        hr = np.mean(curr_hr)
    else:
        hr = 0
    return hr


def ecg_csv_all(use_existing=True):
    paths = [path for path in os.listdir(utils.get_ecg_data_path()) if path.lower().endswith(".edf")]
    for path in paths:
        path = os.path.join(utils.get_ecg_data_path(), path)
        ecg_csv(path=path, use_existing=use_existing)


def ecg_csv(path, use_existing=True):
    ecg_id = path.lower().split("_")[-1].replace(".edf", "")
    filename = 'rpeaks' + str(ecg_id) + '.csv'
    path_csv = os.path.join(utils.get_rpeaks_path(), filename)

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
    else:
        r_peaks_data = pd.read_csv(path_csv)

    if (not sample_rate) or (not length):
        signals, signal_headers, header = highlevel.read_edf(path)
        for i, s in enumerate(signals):
            if signal_headers[i]['transducer'] == 'ECG electrode':
                sample_rate = signal_headers[i]['sample_rate']
                length = len(s)
                break

    return r_peaks_data, ecg_id, sample_rate, length


def serialize_ecg_hrs():
    paths = [path for path in os.listdir(utils.get_ecg_data_path()) if
             path.lower().endswith(".edf")]
    for path in paths:
        path = os.path.join(utils.get_ecg_data_path(), path)
        ecg_csv(path=path)
    paths = [path for path in os.listdir(utils.get_rpeaks_path()) if
             path.lower().endswith(".csv") and path.lower().startswith("rpeaks")]
    ecg_hrs = {}
    for path in paths:
        path = os.path.join(utils.get_rpeaks_path(), path)
        data = pd.read_csv(path)
        ecg_hrs[path] = get_ecg_hr(data.to_numpy(), 10 * 1000, 1000)
    ecg_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ecg_hrs.items()]))
    ecg_data.to_csv(os.path.join(utils.get_ecg_path(), 'ecg_hrs.csv'))


def serialize_bcg_hrs():
    paths = [path for path in os.listdir(utils.get_brueser_path()) if
             path.lower().endswith(".csv") and path.lower().startswith('brueser')]
    bcg_hrs = {}
    for path in paths:
        path = os.path.join(utils.get_brueser_path(), path)
        data = pd.read_csv(path)
        bcg_hrs[path] = get_brueser_hr(data['unique_peaks'], data['medians'], 10 * 100, 100)
    bcg_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in bcg_hrs.items()]))
    bcg_data.to_csv(os.path.join(utils.get_bcg_path(), 'bcg_hrs.csv'))


def compare_ecg_hr_methods():
    paths = [path for path in os.listdir(utils.get_rpeaks_path()) if
             path.lower().endswith(".csv") and path.lower().startswith("rpeaks")]
    for path in paths:
        hrs = {}
        path = os.path.join(utils.get_rpeaks_path(), path)
        data = pd.read_csv(path)
        path_comparison = path.replace("rpeaks", "comparison")
        hrs["mean"] = get_ecg_hr(data.to_numpy(), 10 * 1000, 1000)
        hrs["std"] = get_std_ecg_hr(data.to_numpy(), 10 * 1000, 1000)
        hr_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in hrs.items()]))
        hr_data.to_csv(path_comparison)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    brueser_csv_all(fs=100, use_existing=False)
    pass
