import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
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

    def plot_bland_altman(self, indices, title=None):
        utils.bland_altman_plot(self.informative_info.loc[indices, 'ecg_hr'],
                                self.informative_info.loc[indices, 'bcg_hr'], title=title)

    def print_report_all_signal(self, test_indices, y_pred):
        print("\n Coverage bestimmter Fehler des genutzten Signals auf Gesamtsignal")
        print(f"Fehler < 5 gesamt           : {self.get_5percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 5 klassifiziert    : {self.get_5percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 10 gesamt          : {self.get_10percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 10 klassifiziert   : {self.get_10percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 15 gesamt          : {self.get_15percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 15 klassifiziert   : {self.get_15percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler < 20 gesamt          : {self.get_20percent_coverage(test_indices, use_brueser_hr=False):.2f} %")
        print(f"Fehler < 20 klassifiziert   : {self.get_20percent_coverage(test_indices, y_pred):.2f} %")
        print(f"Fehler = 667 gesamt         : {self.get_unusable_percentage(test_indices, use_brueser_hr=False):.5f} %")
        print(f"Fehler = 667 klassifiziert  : {self.get_unusable_percentage(test_indices, y_pred):.5f} %")
        # TODO: plot

    def print_report_informative_signal(self, test_indices, y_pred, path=None):
        print("\n Anteil bestimmter Fehler auf informativem Signal")
        self.print_report_coverage(test_indices, y_pred, name="Informatives Signal", path=path)
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
        plt.xlabel('$E\\textsubscript{HR}$')
        if name:
            plt.title(name)
        if path is not None:
            plt.savefig(path, transparent=True, bbox_inches='tight', dpi=300)
        print(
            f"Fehler < 5    : {percent_5:.2f} %")
        print(
            f"Fehler < 10   : {percent_10:.2f} %")
        print(
            f"Fehler < 15   : {percent_15:.2f} %")
        print(
            f"Fehler < 20   : {percent_20:.2f} %")
        print(
            f"Fehler = 667  : {percent_unusuable:.5f} %")

    def print_model_test_report(self, save_title=None):
        y_pred, y_true = self.predict_test_set()
        _, x2, _, y2, _, _ = self._get_patient_split()
        if type(y_pred) != pd.Series:
            y_pred = pd.Series(y_pred, index=y_true.index)
        test_indices = x2.index

        fp_indices = y_true[np.logical_and(~y_true, y_pred)].index
        fp_labels = y_pred[fp_indices]

        fn_indices = y_true[np.logical_and(y_true, ~y_pred)].index
        fn_labels = y_pred[fn_indices]

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
        self.plot_bland_altman(test_indices[y_pred], "Informativ klassifiziert")
        if save_title is not None:
            file = save_title + '-bland-altman-inf.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)
        self.plot_bland_altman(test_indices[np.logical_and(y_pred, y_true)], "True Positives")
        if save_title is not None:
            file = save_title + '-bland-altman-tp.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)

        print("\n False Positives")
        print(f"Durchschnittlicher Fehler von False Positives: {self.get_mean_error(fp_indices):.2f}")
        if save_title is not None:
            file = save_title + '-fp.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        self.print_report_coverage(fp_indices, fp_labels, name="Falsch Positive", path=file)
        self.plot_bland_altman(fp_indices, "Falsch Positive")
        if save_title is not None:
            file = save_title + '-bland-altman-fp.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)
        # print(f"Standardabweichung des Fehlers von False Positives: {self.get_mean_error_abs(fp_indices, fp)}")
        print("\n False Negatives")
        print(f"Durchschnittlicher Fehler von False Negatives: {self.get_mean_error(fn_indices):.2f}")
        if save_title is not None:
            file = save_title + '-fn.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
        else:
            file = None
        self.print_report_coverage(fn_indices, ~fn_labels, name="Falsch Negative", path=file)
        self.plot_bland_altman(fn_indices, title="Falsch Negative")
        if save_title is not None:
            file = save_title + '-bland-altman-fn.pdf'
            file = os.path.join(utils.get_thesis_pic_path(), file)
            plt.savefig(file, transparent=True, bbox_inches='tight', dpi=300)

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

    def plot_bland_altman(self, indices, title=None):
        utils.bland_altman_plot(self.informative_info.loc[indices, 'ecg_hr'],
                                self.features.loc[indices, 'sqi_hr'], title=title)

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
        return np.mean(data_subset['error'])

    def get_mean_error_rel(self, indices, labels=None, use_brueser_hr=False):
        data_subset = self.informative_info.loc[indices]
        if labels is not None:
            data_subset = data_subset[labels]
        if use_brueser_hr:
            # data_subset = data_subset[~np.isclose(data_subset['sqi_hr_error'], 667)]
            return np.mean(data_subset['sqi_hr_diff_rel'])
        # data_subset = data_subset[~np.isclose(data_subset['error'], 667)]
        return np.mean(data_subset['error'])

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

    def get_5percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=True):
        return self.get_percent_coverage(5, indices, labels, informative_only, use_brueser_hr)

    def get_10percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=True):
        return self.get_percent_coverage(10, indices, labels, informative_only, use_brueser_hr)

    def get_15percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=True):
        return self.get_percent_coverage(15, indices, labels, informative_only, use_brueser_hr)

    def get_20percent_coverage(self, indices, labels=None, informative_only=False, use_brueser_hr=True):
        return self.get_percent_coverage(20, indices, labels, informative_only, use_brueser_hr)

    def get_percent_coverage(self, threshold, indices, labels=None, informative_only=False, use_brueser_hr=True):
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
