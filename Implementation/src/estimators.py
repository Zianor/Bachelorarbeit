import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import utils as utils
from data_statistical_features import Segment, DataSetBrueser, DataSetStatistical, DataSetPino, SegmentStatistical


class QualityEstimator:

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients'):
        self.data_folder = data_folder
        self.segment_length = segment_length
        self.overlap_amount = overlap_amount
        self.hr_threshold = hr_threshold
        self.data = self._load_segments()
        self.informative_info = self.data[QualityEstimator._get_informative_names()].copy()
        self.features = self._get_features().copy()
        self.target = self.data['informative'].copy()
        self.patient_id = self.data['patient_id'].copy()

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
        return 100 / len(data_subset.index) * len(data_subset[data_subset['quality_class'] == 0])

    def get_percent_coverage(self, threshold, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        covered = data_subset[np.logical_or(data_subset['rel_err'] < threshold, data_subset['abs_err'] < threshold/2)]
        return 100 / len(data_subset.index) * len(covered)

    def print_model_test_report(self):
        y_pred, y_true = self.predict_test_set()
        _, x2, _, y2, _, _ = self._get_patient_split()
        if type(y_pred) != pd.Series:
            y_pred = pd.Series(y_pred, index=y_true.index)
        test_indices = x2.index

        fp_indices = y_true[np.logical_and(~y_true, y_pred)].index
        fp = y_pred.loc[fp_indices]

        fn_indices = y_true[np.logical_and(y_true, ~y_pred)].index
        fn = y_pred.loc[fn_indices]

        print(f"Coverage klassifiziert: {len(y_pred[y_pred])/len(y_pred)*100:.2f} %")
        print(f"Coverage annotiert: {len(y_true[y_true]) / len(y_true)*100:.2f} %")
        print(f"Fehler < 5 Prozent/2.5bpm insgesamt: {self.get_5percent_coverage(test_indices):.2f} %")
        print(f"Fehler < 5 Prozent/2.5bpm klassifiziert: {self.get_5percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 10 Prozent/5bpm insgesamt: {self.get_10percent_coverage(test_indices):.2f} %")
        print(f"Fehler < 10 Prozent/5bpm klassifiziert: {self.get_10percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 15 Prozent/7.5bpm insgesamt: {self.get_15percent_coverage(test_indices):.2f} %")
        print(f"Fehler < 15 Prozent/7.5bpm klassifiziert: {self.get_15percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 20 Prozent/10bpm insgesamt: {self.get_20percent_coverage(test_indices):.2f} %")
        print(f"Fehler < 20 Prozent/10bpm klassifiziert: {self.get_20percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Keine Schaetzung insgesamt: {self.get_unusable_percentage(test_indices):.5f} %")
        print(f"Keine Schaetzung klassifiziert: {self.get_unusable_percentage(test_indices, y_pred):.5f} %")
        print(f"Durchschnittlicher Fehler von False Positives: {self.get_mean_error_abs(fp_indices, fp)}")
        # print(f"Standardabweichung des Fehlers von False Positives: {self.get_mean_error_abs(fp_indices, fp)}")
        print(f"Durchschnittlicher Fehler von False Negatives: {self.get_mean_error_abs(fn_indices, fn)}")

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
        return self.predict(x2), y2

    def predict(self, x):
        raise Exception("Not implemented in base class")

    def get_mean_error_abs(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        data_subset = data_subset[data_subset['informative']]
        data_subset = data_subset[data_subset['quality_class'] != 0]
        return np.nanmean(data_subset['abs_err'])

    def get_mean_error_rel(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        data_subset = data_subset[data_subset['quality_class'] != 0]
        data_subset = data_subset[data_subset['informative']]
        return np.nanmean(data_subset['rel_err'])


class BrueserSingleSQI(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4,
                 coverage_threshold=85, data_folder='data_patients'):
        self.sqi_threshold = sqi_threshold
        super(BrueserSingleSQI, self).__init__(segment_length, overlap_amount, hr_threshold, data_folder)
        self.informative_info['sqi_hr_diff_abs'] = np.abs(self.informative_info['ecg_hr'] - self.features['sqi_hr'])
        self.informative_info['sqi_hr_diff_abs'] = self.informative_info['sqi_hr_diff_abs'].replace(np.nan, np.finfo(np.float32).max)
        self.informative_info['sqi_hr_diff_rel'] = 100 / self.informative_info['ecg_hr'] * self.informative_info['sqi_hr_diff_abs']
        self.informative_info['sqi_hr_diff_rel'] = self.informative_info['sqi_hr_diff_rel'].replace(np.nan, np.finfo(np.float32).max)
        self.coverage_threshold = coverage_threshold

    def _load_segments(self):
        """
        Loads BCG data as Dataframe
        :return: Dataframe
        """
        path_hr = utils.get_brueser_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount, self.sqi_threshold,
                                                      self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
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
        data_subset = self.features.loc[x.index]
        labels = data_subset['sqi_coverage'] >= self.coverage_threshold
        return labels

    def get_mean_error_abs(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
            return np.nanmean(data_subset['sqi_hr_diff_abs'])
        data_subset = data_subset[data_subset['quality_class'] != 0]
        data_subset = data_subset[data_subset['informative']]
        return np.nanmean(data_subset['abs_err'])

    def get_mean_error_rel(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
            return np.nanmean(data_subset['sqi_hr_diff_rel'])
        data_subset = data_subset[data_subset['quality_class'] != 0]
        data_subset = data_subset[data_subset['informative']]
        return np.nanmean(data_subset['rel_err'])

    def get_unusable_percentage(self, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
            unusable = data_subset[data_subset['sqi_hr_diff_rel'] == np.finfo(np.float32).max]
        else:
            unusable = data_subset[data_subset['quality_class'] == 0]
        return 100 / len(data_subset.index) * len(unusable)

    def get_percent_coverage(self, threshold, indices, labels=None):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
            covered = data_subset[np.logical_or(data_subset['sqi_hr_diff_rel'] < threshold,
                                                data_subset['sqi_hr_diff_abs'] < threshold/2)]
        else:
            covered = data_subset[np.logical_or(data_subset['rel_err'] < threshold, data_subset['abs_err']
                                                < threshold / 2)]
        return 100 / len(data_subset.index) * len(covered)


class PinoMinMaxStd(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients'):
        super(PinoMinMaxStd, self).__init__(segment_length, overlap_amount, hr_threshold, data_folder=data_folder)

    def _load_segments(self):
        path_hr = utils.get_pino_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount, self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_pino_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)  # other threshold?
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

    def predict(self, x):
        data_subset = self.features.loc[x.index]
        labels = data_subset['T1'] <= data_subset['T2']
        return labels


class MLStatisticalEstimator(QualityEstimator):

    def __init__(self, path, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients'):
        super(MLStatisticalEstimator, self).__init__(segment_length, overlap_amount, hr_threshold, data_folder)
        if os.path.isfile(os.path.join(utils.get_grid_params_path(self.data_folder), path, 'fitted_model.sav')):
            with open(os.path.join(utils.get_grid_params_path(self.data_folder), path, 'fitted_model.sav'), 'rb') as file:
                grid_search = pickle.load(file)
                self.model = grid_search.best_estimator_
        else:
            raise Exception("No model found at given path")

    def _load_segments(self):
        path_hr = utils.get_statistical_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount, self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_statistical_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount)  # other threshold?
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
        to_remove = [np.any(Segment.get_feature_name_array()[:] == v)
                     for v in SegmentStatistical.get_feature_name_array()]
        features = np.delete(SegmentStatistical.get_feature_name_array(), to_remove)
        return self.data[features]

    def predict(self, x):
        data_subset = self.features.loc[x.index]
        labels = self.model.predict(data_subset)
        return labels


if __name__ == "__main__":
    pass
