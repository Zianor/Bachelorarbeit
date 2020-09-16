from pathlib import Path
import os


def get_project_root() -> Path:
    """Returns project root folder.
    """
    return Path(__file__).parent.parent


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


def get_statistical_features_csv_path(segment_length, overlap_amount, hr_threshold) -> Path:
    filename = 'data_statistical_features_l' + str(segment_length) + '_o' + str(overlap_amount) + '_hr' + \
                   str(hr_threshold) + '.csv'
    return os.path.join(get_data_path(), filename)

