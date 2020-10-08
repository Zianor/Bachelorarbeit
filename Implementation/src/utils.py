from pathlib import Path
import os

from scipy.signal import lfilter, butter


def get_project_root() -> Path:
    """Returns project root folder.
    """
    return Path(__file__).parent.parent


def get_thesis_pic_path() -> Path:
    return os.path.join(get_project_root().parent, 'Thesis/pic')


def get_bcg_data_path() -> Path:
    return os.path.join(get_bcg_path(), 'ml_data')


def get_brueser_path() -> Path:
    return os.path.join(get_bcg_path(), 'brueser')


def get_bcg_path() -> Path:
    return os.path.join(get_data_path(), 'bcg')


def get_grid_params_path() -> Path:
    return os.path.join(get_data_path(), 'grid_params')


def get_ecg_path() -> Path:
    return os.path.join(get_data_path(), 'ecg')


def get_ecg_data_path() -> Path:
    return get_ecg_path()


def get_rpeaks_path() -> Path:
    return get_ecg_path()


def get_drift_path() -> Path:
    return os.path.join(get_data_path(), 'drift_compensation')


def get_data_path() -> Path:
    return os.path.join(get_project_root(), 'data')


def get_data_object_path() -> Path:
    return os.path.join(get_data_path(), 'data.set')


def get_data_set_folder(segment_length, overlap_amount) -> Path:
    folder_name = 'data_set_l' + str(segment_length) + '_o' + str(overlap_amount)
    return os.path.join(get_data_path(), folder_name)


def get_features_csv_path(segment_length, overlap_amount, hr_threshold=None):
    if hr_threshold:
        filename = 'data_' + str(hr_threshold) + '.csv'
    else:
        filename = 'data.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def get_own_features_csv_path(segment_length, overlap_amount, hr_threshold=None):
    if hr_threshold:
        filename = 'data_own_features_hr' + str(hr_threshold) + '.csv'
    else:
        filename = 'data_own_features.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def get_statistical_features_csv_path(segment_length, overlap_amount, hr_threshold=None) -> Path:
    if hr_threshold:
        filename = 'data_statistical_features_hr' + str(hr_threshold) + '.csv'
    else:
        filename = 'data_statistical_features.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def get_brueser_features_csv_path(segment_length, overlap_amount, sqi_threshold, hr_threshold=None) -> Path:
    if hr_threshold:
        filename = 'data_brueser_features_sqi' + str(sqi_threshold) + '_hr' + str(hr_threshold) + '.csv'
    else:
        filename = 'data_brueser_features_sqi' + str(sqi_threshold) + '.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def get_pino_features_csv_path(segment_length, overlap_amount, hr_threshold=None) -> Path:
    if hr_threshold:
        filename = 'data_pino_features_hr' + str(hr_threshold) + '.csv'
    else:
        filename = 'data_pino_features.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def get_image_folder(segment_length, overlap_amount, hr_threshold):
    filename = 'images_hr' + str(hr_threshold) + '.csv'
    return os.path.join(get_data_set_folder(segment_length, overlap_amount), filename)


def seconds_to_frames(duration_seconds, frequency):
    """
    Converts a given duration in seconds to the number of frames in a given frequency
    :param duration_seconds: given duration in seconds
    :type duration_seconds: float
    :param frequency: frequency in Hz
    :type frequency: int
    :return: duration in number of frames
    :rtype: int
    """
    duration_frames = duration_seconds * frequency
    return int(duration_frames)


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Butterworth Bandpass
    :param data: data to be filtered
    :param lowcut: lowcut frequency in Hz
    :param highcut: highcut frequency in Hz
    :param fs: sample rate in Hz
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Used internally for Butterworth Bandpass
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def get_plt_settings():
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.serif": "Times",
        "font.family": "serif",
        # Use 12pt font in plots, to match 12pt font in document
        "axes.labelsize": 12,
        "font.size": 12,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.loc": 'upper right'
    }
    return tex_fonts


def get_plt_normal_size():
    width = 298.76 / 72.27
    height = width / 1.618
    return width, height


def get_plt_big_size():
    width = 426.8 / 72.27
    height = width / 1.618
    return width, height

