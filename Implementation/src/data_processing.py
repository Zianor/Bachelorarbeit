import csv

import numpy as np
import scipy
from pyedflib import highlevel

import brueser
import os
from utils import get_project_root
from scipy.io import loadmat
from scipy import signal
from ecgdetectors import Detectors
import pandas as pd


def get_brueser_hr(unique_peaks, medians, segment_length, segment_count, sample_rate):
    unique_peaks = unique_peaks.to_numpy()
    last_peak = unique_peaks[-1]
    segment_count = last_peak//segment_length
    hr = np.zeros(segment_count)

    for i in range(segment_count):
        start = i*segment_length
        end = (i+1)*segment_length
        indices = np.where(np.logical_and(start <= unique_peaks, unique_peaks < end))
        hr[i] = np.mean(medians.to_numpy()[indices])

    hr = [60 / (curr / sample_rate) for curr in hr]  # TODO: use broadcasting # convert to bpm
    return hr


def brueser_csv(fs, use_existing=True):
    data_path = os.path.join(get_project_root(), 'data/')
    paths = [path for path in os.listdir(data_path) if
             path.lower().endswith(".mat")]

    for path in paths:
        if(use_existing):
            number = path.lower().split("_")[-1]
            number = number.replace("mat", "")
            path = os.path.join(os.path.join(get_project_root(), 'data/'), path)

            data = loadmat(path)['BCG_raw_data'][0]

            filename = 'brueser' + str(number) + 'csv'
            path_csv = os.path.join(data_path, filename)

            if not os.path.isfile(path_csv):
                win = np.arange(0.3 * fs, 2 * fs + 1, dtype=np.int32)
                result, est_len, quality_arr = brueser.interval_probabilities(data, win, estimate_lengths=True)
                peaks, _ = scipy.signal.find_peaks(data, distance=win[0])
                unique_peaks, medians, qualities = brueser.rr_intervals_from_est_len(est_len, peaks, data, quality_arr, win[0])
                # alle 3 serialisieren

                with open(path_csv, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow(["unique_peaks", "medians", "qualities"])
                    for i, peak in enumerate(unique_peaks):
                        writer.writerow([peak, medians[i], qualities[i]])


brueser_csv(100)
