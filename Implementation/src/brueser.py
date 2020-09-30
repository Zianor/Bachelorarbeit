import numpy as np
import scipy
from numba import jit


# Usage:
# data = brueser.filter(data, FS)
# win = np.arange(0.3 * FS, 2 * FS + 1, dtype=np.int32)
# result, est_len, quality_arr = brueser.interval_probabilities(data, win, estimate_lengths = True)
# peaks, _ = scipy.signal.find_peaks(data, distance=win[0])
# unique_peaks, medians, qualities = brueser.rr_intervals_from_est_len(est_len, peaks, data, quality_arr, win[0])

# rr_baseline = medians[qualities > np.max(qualities) / 5]
# unique_peaks = unique_peaks[qualities > np.max(qualities) / 5]
from data_statistical_features import SegmentStatistical


def _bandpass_order(f_stop1, f_pass1, f_pass2, f_stop2, dpass_dB, dstop_dB, fsamp = 1):
    """
    Optimal FIR (equal ripple) Bandpass Order Determination

    Text reference: Ifeachor, Digital Signal Processing a Practical Approach,
    second edition, Prentice Hall, 2002.
    Journal paper reference: F. Mintzer & B. Liu, Practical Design Rules for Optimum
    FIR Bandpass Digital Filters, IEEE Transactions on Acoustics and Speech, pp.
    204-206, April,1979.

    Source: https://scikit-dsp-comm.readthedocs.io/en/latest/_modules/sk_dsp_comm/fir_design_helper.html#fir_remez_bpf
    Basic Linear Phase Digital Filter Design Helper

    Copyright (c) March 2017, Mark Wickert
    All rights reserved.
    """
    dpass = 1 - 10**(-dpass_dB/20)
    dstop = 10**(-dstop_dB/20)
    Df1 = (f_pass1 - f_stop1)/fsamp
    Df2 = (f_stop2 - f_pass2)/fsamp
    b1 = 0.01201
    b2 = 0.09664
    b3 = -0.51325
    b4 = 0.00203
    b5 = -0.5705
    b6 = -0.44314

    Df = min(Df1, Df2)
    Cinf = np.log10(dstop)*(b1*np.log10(dpass)**2 + b2*np.log10(dpass) + b3) \
           + (b4*np.log10(dpass)**2 + b5*np.log10(dpass) + b6)
    g = -14.6*np.log10(dpass/dstop) - 16.9
    N = Cinf/Df + g*Df + 1
    ff = 2*np.array([0, f_stop1, f_pass1, f_pass2, f_stop2, fsamp/2])/fsamp
    aa = np.array([0, 0, 1, 1, 0, 0])
    wts = np.array([dpass/dstop, 1, dpass/dstop])
    return int(N), ff, aa, wts


def fir_remez_bpf(f_stop1, f_pass1, f_pass2, f_stop2, d_pass, d_stop, fs, n_bump=5):
    """
    Design an FIR bandpass filter using remez with order
    determination. The filter order is determined based on
    f_stop1 Hz, f_pass1 Hz, f_pass2 Hz, f_stop2 Hz, and the
    desired passband ripple d_pass dB and stopband attenuation
    d_stop dB all relative to a sampling rate of fs Hz.

    Mark Wickert October 2016, updated October 2018
    Source: https://scikit-dsp-comm.readthedocs.io/en/latest/_modules/sk_dsp_comm/fir_design_helper.html#fir_remez_bpf

    Basic Linear Phase Digital Filter Design Helper

    Copyright (c) March 2017, Mark Wickert
    All rights reserved.
    """
    n, ff, aa, wts = _bandpass_order(f_stop1, f_pass1, f_pass2, f_stop2,
                                  d_pass, d_stop, fsamp=fs)
    # Bump up the order by N_bump to bring down the final d_pass & d_stop
    n_taps = n
    n_taps += n_bump
    b = scipy.signal.remez(n_taps, ff, aa[0::2], wts, Hz=2)
    return b


def filter(data, sample_rate):
    """Filters signal
    """
    b = fir_remez_bpf(f_pass1=1, f_stop1=0.5, f_pass2=10, f_stop2=12, d_pass=0.1, d_stop=80, fs=sample_rate)
    return scipy.signal.lfilter(b=b, a=1, x=data)
    return data
    # FIR Filter 0.5 und 20 Hz
    # erste Ableitung mit Savitzky-Golay filter


def interval_probabilities(data, win, estimate_lengths=True):
    """
    Processes the whole data with given window
    :param data: bcg data
    :param win: window of all n for HR of 30 to 200
    """
    corr_arr = np.zeros((data.size, win.size))
    est_len_arr = np.zeros(data.size)
    quality_arr = np.zeros(data.size)

    # normal range
    for l in range(win[-1], data.size - win[-1]):  # iterate over all windows
        win_sig = data[l - win[-1]:l + win[-1] + 1]
        probabilities, est_len, quality = prob_estimator(win_sig, win, estimate_lengths)
        corr_arr[l, :] = probabilities
        est_len_arr[l] = est_len
        quality_arr[l] = quality

    return corr_arr, est_len_arr, quality_arr


@jit(nopython=True, fastmath=True)
def calc_all_coeffs(signal, signal_length, min_lag, min_window_size):
    """
    :param min_lag: minimal interval length in samples
    :param signal: signal without mean, starting at k in window
    :param signal_length: size of window
    :param min_window_size: is 0, does nothing?
    """
    mid = signal_length // 2
    coeffs = np.zeros((mid + 1 - min_lag, 3))

    for i in range(min_lag, mid + 1):  # N_min to N_max + 1, i is equivalent to N
        win_len = max(i, min_window_size)
        b_start = mid - i  # equivalent to v-N if v = 0
        w = 1.0

        cov = 0.0
        diff = 0.0
        max_val = -np.inf
        for j in range(win_len):  # for each j in range 0 to i, equivalent to v in the sum
            cov += signal[mid + j] * signal[b_start + j]
            diff += abs(signal[mid + j] - signal[b_start + j])
            max_val = max(max_val, (signal[mid + j] + signal[b_start + j]))

        coeffs[i - min_lag, :] = np.array([cov * (w / win_len), max_val / 2, diff * (w / win_len)])
        # second is mean of max values

    return coeffs


@jit(nopython=True, fastmath=True)
def find_largest_peak(signal, min_win):
    """
    :return max_ampl: value at max amplitude
    :return idx: position of max amplitude
    :return est_len: estimated length in samples
    """
    extrs = []
    for i in range(1, signal.shape[0] - 1):
        if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
            extrs.append(i)

    extrs = np.array(extrs)  # indices of peaks
    ampls = signal[extrs]  # amplitudes
    if ampls.size == 0:
        max_ampl = 0
        idx = 0
    else:
        idx = np.argmax(ampls)
        max_ampl = np.max(ampls)

    if ampls.size > 1:
        sort_idx = np.argsort(ampls)  # small to large
        ampls_sorted = ampls[sort_idx]
        extrs_sorted = extrs[sort_idx]
        if (ampls_sorted[-1] - ampls_sorted[-2]) / ampls_sorted[-1] < 0.999 \
                and abs((extrs_sorted[-1] + min_win - 1) / 2 - (extrs_sorted[-2] + min_win - 1)) < 5:
            max_ampl = ampls_sorted[-2]
            idx = np.flatnonzero(extrs == extrs_sorted[-2])[0]

    if extrs.size == 0:
        est_len = 0
    else:
        try:
            est_len = round(min_win + extrs[idx] - 1)
        except:
            return

    return max_ampl, idx, est_len


@jit(nopython=True, fastmath=True)
def rr_intervals_from_est_len(est_len, peaks, data, quality, min_win):
    est_len_adj = est_len.astype(np.int32)
    corresponding_peaks = np.zeros((data.shape[0]), dtype=np.int32)

    for data_idx in range(min_win, data.shape[0] - min_win):  # iterate over whole signal again
        estimated_interval = est_len_adj[data_idx]
        max_sum = -np.inf
        for peak in peaks[np.searchsorted(peaks, data_idx):np.searchsorted(peaks, data_idx + estimated_interval) + 1]:
            # all peaks between data_idx and data_idx + estimated_interval
            # search peak with largest combined amplitude of data at peak and data at (peak - estimated_interval)
            if max_sum < data[peak] + data[peak - estimated_interval]:
                max_sum = data[peak] + data[peak - estimated_interval]
                corresponding_peaks[data_idx] = peak

    unique_peaks = np.unique(corresponding_peaks)
    medians = np.zeros(unique_peaks.shape[0])
    qualities = np.zeros(unique_peaks.shape[0])

    for idx, peak in enumerate(unique_peaks):
        # calculate median of all lengths and qualities associated with this peak
        medians[idx] = np.median(est_len_adj[corresponding_peaks == peak])
        qualities[idx] = np.median(quality[corresponding_peaks == peak]) * 10

    return unique_peaks, medians, qualities


def prob_estimator(win_sig, win, estimate_lengths=True):
    """
    :param win_sig: signal in the given window
    :param win: win = np.arange(0.3 * FS, 2 * FS + 1, dtype=np.int32)
    """
    # Expand to make Matlab transition easier
    win_sig = win_sig[np.newaxis, :]

    #  Remove mean
    win_sig_modified = win_sig - np.mean(win_sig, axis=1, keepdims=True)

    # Compute Scores
    # combined = np.ones(win.size)
    probabilities = np.zeros((win_sig.shape[0], win.size))
    quality = np.zeros(win_sig.shape[0])
    est_len = np.zeros(win_sig.shape[0])

    min_window_size = 0

    signal_length = win_sig.shape[1] - (win_sig.shape[1] % 2)  # Signal length must be even
    min_lag = win[0]

    # Make sure minlag is valid
    assert min_lag <= signal_length // 2

    for k in range(win_sig.shape[0]):  # only one runthrough with a single channel signal

        coeffs = calc_all_coeffs(win_sig_modified[k, :], signal_length, min_lag, min_window_size).T

        eps = np.spacing(1)

        # coeffs = [xc, ms, ad]
        coeffs[2, :] = 1 / (coeffs[2, :] + eps)  # Inverses von AMDF

        # aus Matlab so von Brueser uebernommen
        coeffs -= np.min(coeffs, axis=1, keepdims=True)
        coeffs *= 0.9 / np.sum(coeffs, axis=1, keepdims=True) + eps
        coeffs += 0.1 / coeffs.shape[1]

        probabilities[k, :] = coeffs[0, :] * coeffs[1, :] * coeffs[2, :]
        # combined *= probabilities[k, :]  # not needed for single channel

        if estimate_lengths:
            max_ampl, _, est_len_peak = find_largest_peak(probabilities[k, :], win[0])
            quality[k] = max_ampl / np.sum(probabilities[k, :])
            # assert quality[k] <= 1 or np.isnan(quality[k]), f"erste Qualitaet bei {quality[k]}, array {probabilities[k, :]}"
            est_len[k] = est_len_peak

    return probabilities, est_len, quality

