import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import utils as utils
from data_statistical_features import Segment, DataSetBrueser, DataSetStatistical, DataSetPino, SegmentStatistical


class QualityEstimator:

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        self.segment_length = segment_length
        self.overlap_amount = overlap_amount
        self.hr_threshold = hr_threshold
        self.data = self._load_segments()
        self.informative_info = self.data[QualityEstimator._get_informative_names()]
        self.features = self._get_features()
        self.target = self.data['informative']
        self.patient_id = self.data['patient_id']

    @staticmethod
    def _get_informative_names():
        names = Segment.get_feature_name_array()
        return np.delete(names, np.where(names == "brueser_sqi"))

    def _get_features(self):
        raise Exception("Not implemented in base class")

    def _load_segments(self):
        raise Exception("Not implemented in base class")

    def get_5percent_coverage(self, indices, labels=None):
        return self.get_percent_coverage(5, indices, labels)

    def get_10percent_coverage(self, indices, labels=None):
        return self.get_percent_coverage(10, indices, labels)

    def get_15percent_coverage(self, indices, labels=None):
        return self.get_percent_coverage(15, indices, labels)

    def get_20percent_coverage(self, indices, labels=None):
        return self.get_percent_coverage(20, indices, labels)

    def get_unusable_percentage(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        return 100 / len(data_subset.index) * len(data_subset[data_subset['rel_error'] == np.finfo(np.float32).max])

    def get_percent_coverage(self, threshold, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        covered = data_subset[np.logical_or(data_subset['rel_err'] < threshold, data_subset['abs_err'] < threshold/2)]
        return 100 / len(data_subset.index) * len(covered)

    def print_model_test_report(self):
        y_pred = self.predict_test_set()
        _, x2, _, y2, _, _ = self._get_patient_split()
        test_indices = x2.indices
        print(f"Fehler < 5 Prozent/2.5bpm insgesamt: {self.get_5percent_coverage(test_indices):.2f}")
        print("Fehler < 5 Prozent/2.5bpm klassifiziert: %.2f" % self.get_5percent_coverage(test_indices, y_pred))
        print("Fehler < 10 Prozent/5bpm insgesamt: %.2f" % self.get_10percent_coverage(test_indices))
        print("Fehler < 10 Prozent/5bpm klassifiziert: %.2f" % self.get_10percent_coverage(test_indices, y_pred))
        print("Fehler < 15 Prozent/7.5bpm insgesamt: %.2f" % self.get_15percent_coverage(test_indices))
        print("Fehler < 15 Prozent/7.5bpm klassifiziert: %.2f" % self.get_15percent_coverage(test_indices, y_pred))
        print("Fehler < 20 Prozent/10bpm insgesamt: %.2f" % self.get_20percent_coverage(test_indices))
        print("Fehler < 20 Prozent/10bpm klassifiziert: %.2f" % self.get_20percent_coverage(test_indices, y_pred))
        print("Keine Schaetzung insgesamt: %.2f" % self.get_unusable_percentage(test_indices))
        print("Keine Schaetzung klassifiziert: %.2f" % self.get_unusable_percentage(test_indices, y_pred))

    def _get_patient_split(self, test_size=0.33):
        patient_ids_1, patient_ids_2 = train_test_split(self.patient_id.unique(), random_state=1, test_size=test_size)
        x1 = self.features[np.isin(self.patient_id, patient_ids_1)]
        x2 = self.features[np.isin(self.patient_id, patient_ids_2)]
        y1 = self.target[np.isin(self.patient_id, patient_ids_1)]
        y2 = self.target[np.isin(self.patient_id, patient_ids_2)]
        groups1 = self.patient_id[np.isin(self.patient_id, patient_ids_1)]
        groups2 = self.patient_id[np.isin(self.patient_id, patient_ids_2)]
        return x1, x2, y1, y2, groups1, groups2

    def predict_all_labels(self):
        return self.predict(self.features)

    def predict_test_set(self):
        x1, x2, y1, y2, groups1, groups2 = self._get_patient_split()
        return self.predict(x2)

    def predict(self, x):
        raise Exception("Not implemented in base class")

    def get_mean_error_abs(self, indices, labels):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
            data_subset['abs_err'] = data_subset['abs_err'].replace(np.inf, np.nan)
        return np.nanmean(data_subset['abs_err'])

    def get_mean_error_rel(self, indices, labels):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        data_subset['rel_err'] = data_subset['rel_err'].replace(np.inf, np.nan)
        return np.nanmean(data_subset['rel_err'])


class BrueserSingleSQI(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4,
                 coverage_threshold=85):
        self.sqi_threshold = sqi_threshold
        super(BrueserSingleSQI, self).__init__(segment_length, overlap_amount, hr_threshold)
        self.features['sqi_hr_diff_abs'] = np.abs(self.informative_info['ecg_hr'] - self.features['sqi_hr'])
        self.features['sqi_hr_diff_rel'] = 100 / self.informative_info['ecg_hr'] * self.features['sqi_hr_diff_abs']
        self.coverage_threshold = coverage_threshold

    def _load_segments(self):
        """
        Loads BCG data as Dataframe
        :return: Dataframe
        """
        path_hr = utils.get_brueser_features_csv_path(self.segment_length, self.overlap_amount, self.sqi_threshold,
                                                      self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_features_csv_path(self.segment_length, self.overlap_amount,
                                               self.sqi_threshold)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['rel_err'] < self.hr_threshold
                data.to_csv(path_hr, index=False)
            else:
                warnings.warn('No csv, data needs to be reproduced. This may take some time')
                DataSetBrueser(segment_length=self.segment_length, overlap_amount=self.overlap_amount,
                               hr_threshold=self.hr_threshold)
        return pd.read_csv(path_hr, index_col=False)

    def _get_features(self):
        features = ['sqi_hr', 'sqi_coverage']
        return self.data[features].copy()

    def predict_all_labels(self):
        return self.predict(self.features)

    def predict(self, x):
        data_subset = self.features[x.indices]
        labels = data_subset['sqi_coverage'] >= self.coverage_threshold
        return labels


class PinoMinMaxStd(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(PinoMinMaxStd, self).__init__(segment_length, overlap_amount, hr_threshold)

    def _load_segments(self):
        path_hr = utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_pino_features_csv_path(self.segment_length, self.overlap_amount)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['rel_err'] < self.hr_threshold
                data.to_csv(path_hr, index=False)
            else:
                warnings.warn('No csv, data needs to be reproduced. This may take some time')
                DataSetPino(segment_length=self.segment_length, overlap_amount=self.overlap_amount,
                            hr_threshold=self.hr_threshold)
        return pd.read_csv(path_hr, index_col=False)

    def _get_features(self):
        features = ['T1', 'T2']
        return self.data[features].copy()

    def predict_all_labels(self):
        labels = self.features['T1'] <= self.features['T2']
        return labels

    def predict_test_set(self):
        x1, x2, y1, y2, groups1, groups2 = self._get_patient_split()
        return self.predict(x2)

    def predict(self, x):
        data_subset = self.features[x.indices]
        labels = data_subset['T1'] <= data_subset['T2']
        return labels


class MLStatisticalEstimator(QualityEstimator):

    def __init__(self, path, segment_length=10, overlap_amount=0.9, hr_threshold=10):
        super(MLStatisticalEstimator, self).__init__(segment_length, overlap_amount, hr_threshold)
        if os.path.isfile(os.path.join(path, 'fitted_model.sav')):
            with open(os.path.join(path, 'fitted_model.sav'), 'rb') as file:
                grid_search = pickle.load(file)
                self.model = grid_search.best_estimator_
        else:
            raise Exception("No model found at given path")

    def _load_segments(self):
        path_hr = utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount, self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_statistical_features_csv_path(self.segment_length, self.overlap_amount)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['rel_err'] < self.hr_threshold
                data.to_csv(path_hr, index=False)
            else:
                warnings.warn('No csv, data needs to be reproduced. This may take some time')
                DataSetStatistical(segment_length=self.segment_length, overlap_amount=self.overlap_amount,
                                   hr_threshold=self.hr_threshold)
        return pd.read_csv(path_hr, index_col=False)

    def _get_features(self):
        features = np.delete(SegmentStatistical.get_feature_name_array(), Segment.get_feature_name_array())
        return features

    def predict(self, x):
        data_subset = self.features[x.indices]
        labels = self.model.predict(data_subset)
        return labels


if __name__ == "__main__":
    pass
