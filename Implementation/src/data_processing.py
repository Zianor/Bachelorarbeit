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
from utils import get_project_root


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
    indices = np.where(np.logical_and(start <= unique_peaks, unique_peaks < end))
    hr = 60 / (np.mean(medians.to_numpy()[indices]) / sample_rate)
    return hr


def brueser_csv(fs, use_existing=True):
    data_path = os.path.join(get_project_root(), 'data/')
    paths = [path for path in os.listdir(data_path) if
             path.lower().endswith(".mat")]

    for path in paths:
        if use_existing:
            path = os.path.join(os.path.join(get_project_root(), 'data/'), path)

            data = loadmat(path)['BCG_raw_data'][0]

            number = path.lower().split("_")[-1]
            number = number.replace("mat", "")
            filename = 'brueser' + str(number) + 'csv'
            path_csv = os.path.join(data_path, filename)

            if not use_existing or not os.path.isfile(path_csv):
                win = np.arange(0.3 * fs, 2 * fs + 1, dtype=np.int32)
                result, est_len, quality_arr = brueser.interval_probabilities(data, win, estimate_lengths=True)
                peaks, _ = scipy.signal.find_peaks(data, distance=win[0])
                unique_peaks, medians, qualities = brueser.rr_intervals_from_est_len(est_len, peaks, data, quality_arr,
                                                                                     win[0])

                with open(path_csv, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["unique_peaks", "medians", "qualities"])
                    for i, peak in enumerate(unique_peaks):
                        writer.writerow([peak, medians[i], qualities[i]])


def get_ecg_hr(r_peaks, segment_length, sample_rate, lower_threshold=30, upper_threshold=200):
    last_peak = np.nanmax(r_peaks[-1, :])
    segment_count = int(last_peak // segment_length)
    hr = np.zeros(segment_count)
    for i, _ in enumerate(hr):
        start = i * segment_length
        end = (i + 1) * segment_length
        curr_hr = []
        for r_peaks_single in r_peaks.transpose():
            indices = np.argwhere(np.logical_and(start <= r_peaks_single, r_peaks_single < end))
            hr_guess = len(indices) / segment_length * sample_rate * 60
            if lower_threshold < hr_guess < upper_threshold:
                curr_hr.append(hr_guess)
        if curr_hr:
            hr[i] = np.mean(curr_hr)
        else:
            hr[i] = 0
    return hr


def ecg_csv(use_existing=True):
    data_path = os.path.join(get_project_root(), 'data/ecg/')
    paths = [path for path in os.listdir(data_path) if
             path.lower().endswith(".edf")]
    for path in paths:
        path = os.path.join(os.path.join(get_project_root(), 'data/ecg/'), path)

        number = path.lower().split("_")[-1]
        number = number.replace("edf", "")
        filename = 'rpeaks' + str(number) + 'csv'
        path_csv = os.path.join(data_path, filename)

        if not use_existing or not os.path.isfile(path_csv):
            signals, signal_headers, header = highlevel.read_edf(path)
            detectors = Detectors(signal_headers[0]['sample_rate'])
            r_peaks = {}
            for i, s in enumerate(signals):
                if signal_headers[i]['transducer'] == 'ECG electrode':
                    r_peaks[signal_headers[i]['label']] = detectors.pan_tompkins_detector(s)
            r_peaks_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in r_peaks.items()]))
            r_peaks_data.to_csv(path_csv)


def serialize_ecg_hrs():
    ecg_csv()
    brueser_csv(100)
    data_path = os.path.join(get_project_root(), 'data/ecg/')
    paths = [path for path in os.listdir(data_path) if path.lower().endswith(".csv")]
    ecg_hrs = {}
    for path in paths:
        path = os.path.join(os.path.join(get_project_root(), 'data/ecg/'), path)
        data = pd.read_csv(path)
        ecg_hrs[path] = get_ecg_hr(data.to_numpy(), 10 * 1000, 1000)
    ecg_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in ecg_hrs.items()]))
    ecg_data.to_csv(os.path.join(get_project_root(), 'data/ecg/ecg_hrs.csv'))


def serialize_bcg_hrs():
    data_path = os.path.join(get_project_root(), 'data/')
    paths = [path for path in os.listdir(data_path) if
             path.lower().endswith(".csv") and path.lower().startswith('brueser')]
    bcg_hrs = {}
    for path in paths:
        path = os.path.join(os.path.join(get_project_root(), 'data/'), path)
        data = pd.read_csv(path)
        bcg_hrs[path] = get_brueser_hr(data['unique_peaks'], data['medians'], 10 * 100, 100)
    bcg_data = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in bcg_hrs.items()]))
    bcg_data.to_csv(os.path.join(get_project_root(), 'data/bcg_hrs.csv'))
