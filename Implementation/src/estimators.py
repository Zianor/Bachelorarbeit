import jsonplus as json
import os
import pickle
import warnings

import numpy as np
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, max_error, \
    mean_absolute_error, mean_squared_error, r2_score, plot_roc_curve, accuracy_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin

import utils as utils
from data_statistical_features import Segment, DataSetBrueser, DataSetStatistical, DataSetPino, SegmentStatistical, \
    SegmentOwn, DataSetOwn


class QualityEstimator:

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients'):
        plt.rcParams.update(utils.get_plt_settings())
        plt.rcParams['axes.grid'] = False
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

    def get_5percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(5, indices, labels, informative_only, use_brueser_hr)

    def get_10percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(10, indices, labels, informative_only, use_brueser_hr)

    def get_15percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(15, indices, labels, informative_only, use_brueser_hr)

    def get_20percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(20, indices, labels, informative_only, use_brueser_hr)

    def get_unusable_percentage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        """
        :param indices: indices of used data
        :param labels: if None all data is assumed as informative and used
        :param informative_only: if true percentage only of data[labels]
        :param use_brueser_hr: only for brueser classification
        """
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if informative_only:
            all_length = len(data_subset.index)
            if all_length == 0:
                raise Exception("informative_only not supported when no informative Segments are given")
        else:
            all_length = len(indices)
        return 100 / all_length * len(data_subset[data_subset['quality_class'] == 0])

    def get_percent_coverage(self, threshold, indices, labels=None, informative_only=False, use_brueser_hr=False):
        """
        :param indices: indices of used data
        :param labels: if None all data is assumed as informative and used
        :param informative_only: if true percentage only of data[labels]
        :param use_brueser_hr: only for brueser classification
        """
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        covered = data_subset[data_subset['error'] < threshold]
        if informative_only:
            all_length = len(data_subset.index)
            if all_length == 0:
                raise Exception("informative_only not supported when no informative Segments are given")
        else:
            all_length = len(indices)
        return 100 / all_length * len(covered)

    def plot_bland_altman(self, indices, title=None, color=None):
        utils.bland_altman_plot(self.informative_info.loc[indices, 'ecg_hr'],
                                self.informative_info.loc[indices, 'bcg_hr'], title=title, color=color)

    def print_report_all_signal(self, test_indices, y_pred):
        print("\n Coverage bestimmter Fehler des genutzten Signals auf Gesamtsignal")
        print(f"Fehler < 5 FE gesamt           : {self.get_5percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 5 FE klassifiziert    : {self.get_5percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 10 FE gesamt          : {self.get_10percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 10 FE klassifiziert   : {self.get_10percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 15 FE gesamt          : {self.get_15percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 15 FE klassifiziert   : {self.get_15percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 20 FE gesamt          : {self.get_20percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 20 FE klassifiziert   : {self.get_20percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler = 667 FE gesamt         : {self.get_unusable_percentage(test_indices, use_brueser_hr=False):.5f} %")
        print(f"Fehler = 667 FE klassifiziert  : {self.get_unusable_percentage(test_indices, y_pred):.5f} %")
        # TODO: plot

    def print_report_informative_signal(self, test_indices, y_pred, path=None):
        print("\n Anteil bestimmter Fehler auf informativem Signal")
        self.print_report_coverage(test_indices, y_pred, name="Informativ klassifiziertes Signal", path=path)
        # TODO: plot

    def print_report_coverage(self, test_indices, y_pred, name=None, path=None):
        percent_5 = self.get_5percent_coverage(test_indices, y_pred, informative_only=True)
        percent_10 = self.get_10percent_coverage(test_indices, y_pred, informative_only=True)
        percent_15 = self.get_15percent_coverage(test_indices, y_pred, informative_only=True)
        percent_20 = self.get_20percent_coverage(test_indices, y_pred, informative_only=True)
        percent_unusuable = self.get_unusable_percentage(test_indices, y_pred, informative_only=True)

        arr = np.array([percent_5, percent_10 - percent_5, percent_15 - percent_10, percent_20 - percent_15,
                       100 - percent_20, percent_unusuable])
        data = pd.Series(arr, index=['$<5$', '$5-10$', '$10-15$', '$15-20$', '$20-666$', '$667$'])
        plt.figure(figsize=utils.get_plt_normal_size())
        plt.bar(x=data.index, height=data.values)
        plt.xlabel('$E\\textsubscript{HR}$ in FE')
        if name:
            plt.title(name)
        if path is not None:
            plt.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
        print(
            f"Fehler < 5 FE   : {percent_5:.2f} %")
        print(
            f"Fehler < 10 FE   : {percent_10:.2f} %")
        print(
            f"Fehler < 15 FE   : {percent_15:.2f} %")
        print(
            f"Fehler < 20 FE   : {percent_20:.2f} %")
        print(
            f"Fehler = 667 FE : {percent_unusuable:.5f} %")

    def print_model_test_report(self, save_title=None):
        y_pred, y_true = self.predict_test_set()
        _, x2, _, y2, _, _ = self._get_patient_split()
        if type(y_pred) != pd.Series:
            y_pred = pd.Series(y_pred, index=y_true.index)
        test_indices = x2.index

        fp_indices = y_true[np.logical_and(~y_true, y_pred)].index
        fp_labels = y_pred.loc[fp_indices]

        fn_indices = y_true[np.logical_and(y_true, ~y_pred)].index
        fn_labels = y_pred.loc[fn_indices]

        print("F1-Score: %.2f" % f1_score(y_true, y_pred))

        print("\n Testset insgesamt")
        if save_title is not None:
            file = save_title + '-testset.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        all_true = [True for i in range(len(test_indices))]
        self.print_report_coverage(test_indices, all_true, name="Validierungsset", path=file)

        class_names = ['non-informative', 'informative']
        plt.figure(figsize=utils.get_plt_normal_size())
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=class_names).plot()
        if save_title is not None:
            file = save_title + '-conf-matrix.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)
        print(classification_report(y_true, y_pred, target_names=class_names))

        print("MAE auf als informativ klassifizierten Segmenten: %.2f" % self.get_mean_error(
            y_true.index, y_pred, use_brueser_hr=False))
        print("MAE auf als informativ annotierten Segmenten:  %.2f" % self.get_mean_error(
            y_true.index, y_true, use_brueser_hr=False))
        print("MAE insgesamt:  %.2f" % self.get_mean_error(
            y_true.index, use_brueser_hr=False))
        print("MSE auf als informativ klassifizierten Segmenten: %.2f" % self.get_mean_squared_error(
            y_true.index, y_pred, use_brueser_hr=False))
        print("MSE auf als informativ annotierten Segmenten:  %.2f" % self.get_mean_squared_error(
            y_true.index, y_true, use_brueser_hr=False))
        print("MSE insgesamt:  %.2f" % self.get_mean_squared_error(
            y_true.index, use_brueser_hr=False))
        print("\n")

        print(f"Coverage klassifiziert      : {len(y_pred[y_pred]) / len(y_pred) * 100:.2f} %")
        print(f"Coverage annotiert          : {len(y_true[y_true]) / len(y_true) * 100:.2f} %")
        self.print_report_all_signal(test_indices, y_pred)
        if save_title is not None:
            file = save_title + '-positives.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        self.print_report_informative_signal(test_indices, y_pred, path=file)
        p_indices = test_indices[y_pred]
        is_tp = np.logical_and(y_true.loc[p_indices], y_pred.loc[p_indices])
        self.plot_bland_altman(p_indices, "Informativ klassifiziert", color=is_tp)
        if save_title is not None:
            file = save_title + '-bland-altman-inf.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)

        self.plot_bland_altman(fn_indices, "Falsch-Negative")
        if save_title is not None:
            file = save_title + '-bland-altman-fn.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)

        print("\n False Positives")
        print(f"Durchschnittlicher Fehler von False Positives: {self.get_mean_error(fp_indices):.2f}")
        if save_title is not None:
            file = save_title + '-fp.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        self.print_report_coverage(fp_indices, fp_labels, name="Falsch-Positive", path=file)
        # print(f"Standardabweichung des Fehlers von False Positives: {self.get_mean_error_abs(fp_indices, fp)}")

        print("\n False Negatives")
        print(f"Durchschnittlicher Fehler von False Negatives: {self.get_mean_error(fn_indices):.2f}")
        if save_title is not None:
            file = save_title + '-fn.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        self.print_report_coverage(fn_indices, ~fn_labels, name="Falsch-Negative", path=file)

    def _get_patient_split(self, test_size=0.33, reproduce=True):
        patient_ids_1, patient_ids_2 = train_test_split(self.patient_id.unique(), random_state=1, test_size=test_size)
        if reproduce:
            patient_ids_2 = [23, 13, 11,  8, 16]  # to make it for all algorithms similar
            patient_ids_1 = [36, 9, 28, 14, 27, 26, 22, 35, 5]
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

    def get_mean_error(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        # data_subset = data_subset[data_subset['quality_class'] != 0]
        return np.mean(data_subset['error'])

    def get_mean_squared_error(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        # data_subset = data_subset[data_subset['quality_class'] != 0]
        return np.mean(data_subset['error'].pow(2))

    def get_mean_error_abs(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        # data_subset = data_subset[data_subset['quality_class'] != 0]
        return np.mean(data_subset['abs_err'])

    def get_mean_error_rel(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        # data_subset = data_subset[data_subset['quality_class'] != 0]
        data_subset = data_subset[data_subset['informative']]
        return np.mean(data_subset['rel_err'])


class BrueserSingleSQI(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, sqi_threshold=0.4,
                 coverage_threshold=75, data_folder='data_patients'):
        self.sqi_threshold = sqi_threshold
        super(BrueserSingleSQI, self).__init__(segment_length, overlap_amount, hr_threshold, data_folder)
        self.informative_info['sqi_hr_diff_abs'] = np.abs(self.informative_info['ecg_hr'] - self.features['sqi_hr'])
        mask_less_50 = self.features['sqi_hr'] < 50
        self.informative_info['sqi_hr_diff_rel'] = 100 / self.informative_info['ecg_hr'] * self.informative_info[
            'sqi_hr_diff_abs']
        self.informative_info['sqi_hr_error'] = self.informative_info['sqi_hr_diff_rel']
        self.informative_info.loc[mask_less_50, 'sqi_hr_error'] = self.informative_info.loc[mask_less_50,
                                                                                            'sqi_hr_diff_abs'] / 0.5
        self.informative_info['sqi_hr_diff_abs'] = self.informative_info['sqi_hr_diff_abs'].replace(np.nan, 170)
        self.informative_info['sqi_hr_diff_rel'] = self.informative_info['sqi_hr_diff_rel'].replace(np.nan, 667)
        self.informative_info['sqi_hr_error'] = self.informative_info['sqi_hr_error'].replace(np.nan, 667)
        self.coverage_threshold = coverage_threshold

    def _load_segments(self):
        """
        Loads BCG data as Dataframe
        :return: Dataframe
        """
        path_hr = utils.get_brueser_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                      self.sqi_threshold,
                                                      self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                               self.sqi_threshold)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['error'] < self.hr_threshold
                data.to_csv(path_hr, index=False)
            else:
                warnings.warn('No csv, data needs to be reproduced. This may take some time')
                DataSetBrueser(segment_length=self.segment_length, overlap_amount=self.overlap_amount,
                               hr_threshold=self.hr_threshold, sqi_threshold=self.sqi_threshold)
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

    def get_mean_error(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            # data_subset = data_subset[~np.isclose(data_subset['sqi_hr_error'], 667)]
            return np.mean(data_subset['sqi_hr_error'])
        # data_subset = data_subset[~np.isclose(data_subset['error'], 667)]
        return np.mean(data_subset['error'])

    def get_mean_squared_error(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            # data_subset = data_subset[~np.isclose(data_subset['sqi_hr_error'], 667)]
            return np.mean(data_subset['sqi_hr_error'].pow(2))
        # data_subset = data_subset[~np.isclose(data_subset['error'], 667)]
        return np.mean(data_subset['error'].pow(2))

    def get_mean_error_abs(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            # data_subset = data_subset[~np.isclose(data_subset['sqi_hr_error'], 667)]
            return np.mean(data_subset['sqi_hr_diff_abs'])
        # data_subset = data_subset[~np.isclose(data_subset['error'], 667)]
        return np.mean(data_subset['abs_err'])

    def get_mean_error_rel(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            # data_subset = data_subset[~np.isclose(data_subset['sqi_hr_error'], 667)]
            return np.mean(data_subset['sqi_hr_diff_rel'])
        # data_subset = data_subset[~np.isclose(data_subset['error'], 667)]
        return np.mean(data_subset['rel_err'])

    def get_unusable_percentage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            unusable = data_subset[np.isclose(data_subset['sqi_hr_diff_rel'], 677)]
        else:
            unusable = data_subset[data_subset['quality_class'] == 0]
        if informative_only:
            all_length = len(data_subset.index)
        else:
            all_length = len(indices)
        return 100 / all_length * len(unusable)

    def get_5percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(5, indices, labels, informative_only, use_brueser_hr)

    def get_10percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(10, indices, labels, informative_only, use_brueser_hr)

    def get_15percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(15, indices, labels, informative_only, use_brueser_hr)

    def get_20percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=False):
        return self.get_percent_coverage(20, indices, labels, informative_only, use_brueser_hr)

    def get_percent_coverage(self, threshold, indices, labels=None, informative_only=False, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            covered = data_subset[data_subset['sqi_hr_error'] < threshold]
        else:
            covered = data_subset[data_subset['error'] < threshold]
        if informative_only:
            all_length = len(data_subset.index)
        else:
            all_length = len(indices)
        return 100 / all_length * len(covered)


class PinoMinMaxStd(QualityEstimator):

    def __init__(self, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients'):
        super(PinoMinMaxStd, self).__init__(segment_length, overlap_amount, hr_threshold, data_folder=data_folder)

    def _load_segments(self):
        path_hr = utils.get_pino_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                   self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_pino_features_csv_path(self.data_folder, self.segment_length,
                                                    self.overlap_amount)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['error'] < self.hr_threshold
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
            with open(os.path.join(utils.get_grid_params_path(self.data_folder), path, 'fitted_model.sav'),
                      'rb') as file:
                grid_search = pickle.load(file)
                self.model = grid_search.best_estimator_
        else:
            raise Exception("No model found at given path")

    def _load_segments(self):
        path_hr = utils.get_statistical_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                          self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_statistical_features_csv_path(self.data_folder, self.segment_length,
                                                           self.overlap_amount)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['error'] < self.hr_threshold
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

    def print_model_test_report(self, save_title=None):
        x1, x2, y1, y2, groups1, groups2 = self._get_patient_split()
        plt.figure(figsize=utils.get_plt_normal_size())
        plot_roc_curve(self.model, x2, y2)
        super(MLStatisticalEstimator, self).print_model_test_report(save_title=save_title)


class RegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model=None, threshold=10):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        """Fits the underlying model
        :param y: needs to be continuous target
        """
        self.model.fit(X, y)
        self.classes_ = [False, True]  # order important for AUC?
        return self

    def predict(self, X):
        y_continuous = self.model.predict(X)
        y = [False if curr > self.threshold else True for curr in y_continuous]
        return y

    def get_params(self, deep=True):
        return {'model': self.model,
                'threshold': self.threshold}

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def predict_proba(self, X):
        y_continuous = self.model.predict(X)
        proba_true = np.array([np.math.exp(np.log(0.5)/10 * y) for y in y_continuous])  # e function with f(th)=0.5
        proba_false = 1 - proba_true
        ret = np.ones(shape=(len(y_continuous), 2))
        ret[:, 0] = proba_false  # TODO: clean up
        ret[:, 1] = proba_true
        return ret

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


class OwnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model=None, threshold=10, **params):
        self.model = model
        self.model.set_params(**params)
        self.threshold = threshold

    def fit(self, X, y):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        if type(y) is not pd.Series:
            y = pd.Series(y, index=X.index)
        mask_nan = X.isna().any(axis=1)
        y_true = y.loc[X[~mask_nan].index]
        if type(self.model) == xgb.XGBClassifier:  # class weight
            scale_pos_weight = len(y[~y].index) / len(y[y].index)
            self.model.set_params(scale_pos_weight=scale_pos_weight)
        self.model.fit(X.loc[~mask_nan], y_true)
        self.classes_ = [False, True]  # order important for AUC?
        return self

    def predict(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        mask_nan = X.isna().any(axis=1)
        X_not_na = X.loc[~mask_nan]
        y_pred = self.model.predict(X_not_na)
        y = pd.Series(index=X.index, data=np.full((len(X.index),), False), name='pred')
        y.loc[X_not_na.index] = pd.Series(y_pred, X_not_na.index, dtype=bool)
        return y.to_numpy()

    def get_params(self, deep=True):
        return {'model': self.model,
                'threshold': self.threshold}

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def predict_proba(self, X):
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        mask_nan = X.isna().any(axis=1)
        X_not_na = X.loc[~mask_nan]
        y_proba_not_na = self.model.predict_proba(X_not_na)
        y_proba = pd.DataFrame(index=X.index, columns=self.model.classes_)
        y_proba.loc[mask_nan, True] = np.array([0])
        y_proba.loc[mask_nan, False] = np.array([1])
        y_proba.loc[X_not_na.index, self.model.classes_[0]] = y_proba_not_na[:, 0]
        y_proba.loc[X_not_na.index, self.model.classes_[1]] = y_proba_not_na[:, 1]
        return y_proba.to_numpy()


class OwnEstimator(QualityEstimator):

    def __init__(self, clf, path, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients',
                 feature_selection=None, hyperparameter=None, log=True):
        self.feature_selection = feature_selection
        super(OwnEstimator, self).__init__(segment_length=segment_length, overlap_amount=overlap_amount,
                                           hr_threshold=hr_threshold, data_folder=data_folder)
        pd.options.mode.use_inf_as_na = True
        self.error_target = self.data['error']
        self.path = os.path.join(utils.get_model_path(), path)
        if clf is not None:
            self.clf = OwnClassifier(model=clf, threshold=self.hr_threshold)
            if not os.path.isdir(self.path):
                os.mkdir(path=self.path)
            if hyperparameter is not None:
                print("Hyperparameter optimization, this may need some time")
                self.optimize_hyperparameter(hyperparameter)
            else:
                print("Model is trained, this may need some time")
                self._train()
            self._save_model()
        else:
            self._load_model()

    def optimize_hyperparameter(self, hyperparameter):
        x_g1, x_g2, y_g1, y_g2, groups1, groups2 = self._get_patient_split()
        cv = LeaveOneGroupOut()
        grid_search = RandomizedSearchCV(
            estimator=self.clf, param_distributions=hyperparameter, scoring=['f1', 'roc_auc'],
            cv=cv, n_jobs=6, verbose=2, refit='roc_auc', n_iter=15)
        grid_search.fit(x_g1, y_g1, groups=groups1)
        self.clf = grid_search.best_estimator_
        params = grid_search.best_params_
        with open(os.path.join(self.path, 'grid.sav'), 'wb') as file:
            pickle.dump(grid_search, file=file)
        with open(os.path.join(self.path, 'params.json'), 'w') as file:
            file.write(json.dumps(params))
            file.flush()
        self._get_dataframe_from_cv_results(grid_search.cv_results_).to_csv(os.path.join(self.path, 'grid.csv'))

    def _train(self):
        x1, x2, y1, y2, groups1, groups2 = self._get_patient_split()
        self.clf.fit(x1, y1)

    @staticmethod
    def _get_dataframe_from_cv_results(res):
        data = pd.DataFrame(res)
        scores = ['f1', 'roc_auc']
        columns = ['params']
        for scoring in scores:
            rank = 'rank_test_' + scoring
            mean = 'mean_test_' + scoring
            columns.append(rank)
            columns.append(mean)
        return data.filter(items=columns, axis='columns')

    def _save_model(self):
        model_file = os.path.join(self.path, 'model.sav')
        with open(model_file, 'wb') as file:
            pickle.dump(self.clf, file=file)

    def _load_model(self):
        model_file = os.path.join(self.path, 'model.sav')
        if os.path.isfile(model_file):
            with open(model_file, 'rb') as file:
                self.clf = pickle.load(file)
        else:  # TODO better exeption
            raise Exception("No model given and not found at given path")

    def _get_features(self):
        if self.feature_selection is not None:
            return self.data[self.feature_selection]
        to_remove = [np.any(Segment.get_feature_name_array()[:] == v)
                     for v in SegmentOwn.get_feature_name_array()]
        features = np.delete(SegmentOwn.get_feature_name_array(), to_remove)
        return self.data[features]

    def _load_segments(self):
        path_hr = utils.get_own_features_csv_path(self.data_folder, self.segment_length, self.overlap_amount,
                                                  self.hr_threshold)
        if not os.path.isfile(path_hr):
            path = utils.get_own_features_csv_path(self.data_folder, self.segment_length,
                                                   self.overlap_amount)  # other threshold?
            if os.path.isfile(path):
                data = pd.read_csv(path, index_col=False)
                warnings.warn('Labels are recalculated')
                data['informative'] = data['error'] < self.hr_threshold
                data.to_csv(path_hr, index=False)
            else:
                warnings.warn('No csv, data needs to be reproduced. This may take some time')
                DataSetOwn(segment_length=self.segment_length, overlap_amount=self.overlap_amount,
                           hr_threshold=self.hr_threshold)
        return pd.read_csv(path_hr, index_col=False)

    def predict(self, x):
        return self.clf.predict(x)

    def print_model_test_report(self, save_title=None):
        _, x2, _, y2, _, _ = self._get_patient_split()
        print("AUC: %.2f" % roc_auc_score(y2, self.clf.predict_proba(x2)[:, 1]))
        super(OwnEstimator, self).print_model_test_report(save_title=save_title)

    def print_short_report(self):
        x1, x2, y1, y2, _, _ = self._get_patient_split()
        print("AUC: %.2f" % roc_auc_score(y2, self.clf.predict_proba(x2)[:, 1]))
        y_pred, y_true = self.predict_test_set()
        print("F1: %.2f" % f1_score(y_true, y_pred))
        print(f"Coverage klassifiziert      : {len(y_pred[y_pred]) / len(y_pred) * 100:.2f} %")
        print(f"Coverage annotiert          : {len(y_true[y_true]) / len(y_true) * 100:.2f} %")

        print("MAE auf als informativ klassifizierten Segmenten: %.2f" % self.get_mean_error(y_true.index, y_pred))
        print("MAE auf als informativ annotierten Segmenten:  %.2f" % self.get_mean_error(y_true.index, y_true))
        print("MAE insgesamt:  %.2f" % self.get_mean_error(y_true.index))


class OwnEstimatorRegression(OwnEstimator):

    def __init__(self, clf, path, segment_length=10, overlap_amount=0.9, hr_threshold=10, data_folder='data_patients',
                 feature_selection=None, hyperparameter=None, log=True):
        if clf is not None:
            clf = RegressionClassifier(model=clf, threshold=10)
        super(OwnEstimatorRegression, self).__init__(clf, path, segment_length, overlap_amount, hr_threshold,
                                                     data_folder, feature_selection=feature_selection,
                                                     hyperparameter=hyperparameter, log=log)

    def _train(self):
        x1, x2, y1, y2, groups1, groups2 = self._get_patient_split()
        self.clf.fit(x1, self.informative_info.loc[y1.index, 'error'])


if __name__ == "__main__":
    pass
