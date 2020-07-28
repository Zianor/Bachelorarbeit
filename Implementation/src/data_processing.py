import csv

import numpy as np
import scipy
import brueser
import os
from utils import get_project_root
from scipy.io import loadmat
from scipy import signal


def get_brueser_hr(unique_peaks, medians, segment_length, segment_count, samplerate):
    hr = np.zeros(segment_count)

    for i, _ in enumerate(hr):
        start = i*segment_length
        end = (i+1)*segment_length
        indices = np.argwhere(np.logical_and(start < unique_peaks, unique_peaks < end))[0]
        hr[i] = np.mean(medians[indices])

    hr = 60 * hr / samplerate  # convert to bpm
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
