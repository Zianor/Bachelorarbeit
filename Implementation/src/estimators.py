import os
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import utils as utils
from data_statistical_features import DataSet, Segment, DataSetBrueser, DataSetStatistical, DataSetPino


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

    def get_3bpm_coverage(self, indices, labels=None):
        return self.get_bpm_coverage(self, 3, indices, labels)

    def get_5bpm_coverage(self, indices, labels=None):
        return self.get_bpm_coverage(5, indices, labels)

    def get_bpm_coverage(self, threshold, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        return 100 / len(data_subset.index) * len(data_subset[data_subset['abs_err'] < threshold])

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
        labels = ((self.features['sqi_hr_diff_rel'] < self.hr_threshold) | (self.features['sqi_hr_diff_abs'] <
                  self.hr_threshold/2)) & (self.features['sqi_coverage'] >= self.coverage_threshold)
        return labels

    def predict(self, X):
        data_subset = self.features[X.indices]
        labels = (data_subset['sqi_hr_diff_rel'] < self.hr_threshold) & (
                data_subset['sqi_coverage'] >= self.coverage_threshold)
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

    def predict(self, X):
        data_subset = self.features[X.indices]
        labels = data_subset['T1'] <= data_subset['T2']
        return labels


if __name__ == "__main__":
    brueser = BrueserSingleSQI()
    pred_labels = brueser.predict_all_labels()
    indices = brueser.informative_info.index
    brueser_5bpm_coverage = brueser.get_5bpm_coverage(indices, pred_labels)
    annotation_5bpm_coverage = brueser.get_5bpm_coverage(indices, brueser.informative_info['informative'])
    print("5 bpm Coverage Brueser Classification: %.2f" % brueser_5bpm_coverage)
    print("5 bpm Coverage Annotation: %.2f" % annotation_5bpm_coverage)

    pino = PinoMinMaxStd()
    pred_labels = pino.predict_all_labels()
    indices = pino.informative_info.index
    pino_5bpm_coverage = pino.get_5bpm_coverage(indices, pred_labels)
    annotation_5bpm_coverage = pino.get_5bpm_coverage(indices, pino.informative_info['informative'])
    print("5 bpm Coverage Pino Classification: %.2f" % pino_5bpm_coverage)
    print("5 bpm Coverage Annotation: %.2f" % annotation_5bpm_coverage)
    pass
