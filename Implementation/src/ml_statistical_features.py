import jsonplus as json
import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_statistical_features import DataSet
import src.utils as utils


def load_data(segment_length=10, overlap_amount=0.9, hr_threshold=10):
    """
    Loads BCG data features with its target labels
    :return: BCG data features, target labels
    """
    df = load_data_as_dataframe(segment_length=segment_length, overlap_amount=overlap_amount, hr_threshold=hr_threshold)
    features = df.iloc[:, 0:13]
    target = df['informative_hr']  # TODO: add second label
    mean_error = df['mean error']
    coverage = df['coverage']
    patient_id = df['patient_id']  # TODO: do sth with it
    return features, target, mean_error, coverage, patient_id


def load_data_as_dataframe(segment_length=10, overlap_amount=0.9, hr_threshold=10):
    """
    Loads BCG data as Dataframe
    :return: Dataframe
    """
    path = utils.get_statistical_features_csv_path(segment_length, overlap_amount, hr_threshold)
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet(segment_length=10, overlap_amount=0.9, hr_threshold=10)
    return pd.read_csv(path, index_col=False)


def evaluate_model(y_actual, y_pred):
    """
    Evaluates model performance
    :param y_actual: Actual labels from test set
    :param y_pred: Predicted labels from test set
    :return: evaluation of the model
    :rtype: String
    """
    string_representation = ["Results", os.linesep,
                             "Misclassified examples: %d" % (y_actual != y_pred).sum(), os.linesep,
                             "Accuracy: %.3f" % accuracy_score(y_actual, y_pred), os.linesep,
                             "Confusion matrix", os.linesep, str(confusion_matrix(y_actual, y_pred)), os.linesep,
                             "Classification report", os.linesep,
                             str(classification_report(y_actual, y_pred, target_names=['non-informative',
                                                                                       'informative']))]

    _, _, mean_error, coverage, _ = load_data()

    avg_mean_error = calc_avg_mean_error(mean_error, y_pred, y_actual)
    string_representation.extend(["Average mean error: ", str(avg_mean_error), os.linesep])

    coverage = calc_coverage(coverage, y_pred, y_actual)
    string_representation.extend(["Coverage: ", str(coverage), os.linesep])

    return ''.join(string_representation)


def get_svm_grid_params():
    """Support vector machine
    :return: base estimator and dict of parameters for grid search
    """
    parameters = {
        'clf__kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'clf__C': [1, 10],
        'clf__class_weight': (None, 'balanced')
    }
    svm = SVC(random_state=1)

    return svm, parameters


def get_lda_grid_params():
    """Linear Discriminant Analysis
    :return: base estimator and dict of parameters for grid search
    """
    lda = LinearDiscriminantAnalysis()

    parameters = {
        'clf__solver': ('svd', 'lsqr', 'eigen')
    }

    return lda, parameters


def get_dt_grid_params():
    """
    Decision tree
    :return: base estimator and dict of parameters for grid search
    """
    dt = DecisionTreeClassifier(random_state=1)

    parameters = {
        'clf__criterion': ("gini", "entropy"),
        'clf__splitter': ("best", "random"),
        'clf__class_weight': (None, 'balanced')
    }

    return dt, parameters


def get_rf_grid_params():
    """
    Random forest
    :return: base estimator and dict of parameters for grid search
    """
    rf = RandomForestClassifier(random_state=1)

    parameters = {
        'clf__n_estimators': [10, 30, 50, 75, 100],
        'clf__criterion': ("gini", "entropy"),
        'clf__class_weight': ("balanced", None)
    }

    return rf, parameters


def get_mlp_grid_params():
    """
    Multilayer Perceptron
    :return: base estimator and dict of parameters for grid search
    """
    mlp = MLPClassifier()

    parameters = {
        'clf__hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,)],
        'clf__activation': ('identity', 'logistic', 'tanh', 'relu'),
        'clf__alpha': [0.0001],
        'clf__learning_rate': ('constant', 'invscaling', 'adaptive'),
        'clf__learning_rate_init': [0.0001]
    }

    return mlp, parameters


def calc_avg_mean_error(mean_error, predicted, actual=None):
    """
    Calculates mean bbi error of all segments predicted as informative
    :param mean_error: Array containing each segments mean bbi error
    :param predicted: Predicted labels for all segments
    :param actual: If predicted is not of type pandas.core.series.Series, Series of actual labels needed for mapping
    :return: average mean bbi error, None if mapping was not possible
    """
    if not isinstance(predicted, pd.Series):
        if actual is None:
            warnings.warn(
                'Predicted Labels not of type pandas.core.series.Series but no Series actual for mapping given')
            return None
        actual.update = predicted
        predicted = actual
    mean_error = mean_error[predicted.index]
    avg_mean_error = np.mean(mean_error[predicted.values == 1])
    return avg_mean_error


def calc_coverage(coverage, predicted, actual=None):
    """
    Calculates coverage over all segments
    :param coverage: Array containing each segments mean bbi error
    :param predicted: Predicted labels for all segments
    :param actual: If predicted is not of type pandas.core.series.Series, Series of actual labels needed for mapping
    :return: coverage, None if mapping was not possible
    """
    if not isinstance(predicted, pd.Series):
        if actual is None:
            warnings.warn(
                'Predicted Labels not of type pandas.core.series.Series but no Series actual for mapping given')
            return None
        actual.update = predicted
        predicted = actual
    coverage = coverage[predicted.index]
    coverage_sum = np.sum(coverage[predicted.values == 1])  # coverage over all informative segments
    coverage = coverage_sum / len(coverage)
    return coverage


def _create_list_dict(params):
    list_dict = {}
    for k, v in params.items():
        list_dict[k] = [v]
    return list_dict


def get_dataframe_from_cv_results(res):
    return pd.concat([pd.DataFrame(res["params"]),
                      pd.DataFrame(res["mean_test_score"], columns=["Accuracy"])], axis=1)


def get_patient_split(features, target, patient_id, test_size):
    patient_ids_1, patient_ids_2 = train_test_split(patient_id.unique(), random_state=1, test_size=test_size)
    x1 = features[np.isin(patient_id, patient_ids_1)]
    x2 = features[np.isin(patient_id, patient_ids_2)]
    y1 = target[np.isin(patient_id, patient_ids_1)]
    y2 = target[np.isin(patient_id, patient_ids_2)]
    return x1, x2, y1, y2


def eval_classifier(features, target, patient_id, pipe_with_params, grid_folder_name, test_size=0.33, grid_params=None,
                    patient_cv=True):
    if not os.path.isdir(os.path.join(utils.get_grid_params_path(), grid_folder_name)):
        if not grid_params:
            raise Exception("No existing folder and no params given")
        else:
            os.mkdir(path=os.path.join(utils.get_grid_params_path(), grid_folder_name))

    path = os.path.join(utils.get_grid_params_path(), grid_folder_name)
    model_filename = 'fitted_model.sav'
    params_filename = 'params.json'
    score_filename = 'score.json'

    # split in g1 and g2
    if patient_cv:
        x_g1, x_g2, y_g1, y_g2 = get_patient_split(features, target, patient_id, test_size)
    else:
        x_g1, x_g2, y_g1, y_g2 = train_test_split(features, target, test_size=test_size, random_state=1,
                                                  stratify=target)

    # create pipeline for standardization
    pipe = Pipeline([('scaler', StandardScaler()), ('clf', pipe_with_params)])

    if patient_cv:
        cv = LeaveOneGroupOut()
        groups = patient_id
    else:
        cv = KFold(n_splits=10, shuffle=True, random_state=1)
        groups = None

    # initialize parameters
    score = {}
    best_params = {}
    grid_search = None

    # load or reconstruct fitted grid_search, best_params and score with G1 as training
    if not grid_params:  # no parameters for grid search, load existing data
        if os.path.isfile(os.path.join(path, score_filename)):
            with open(os.path.join(path, score_filename)) as file:
                score = json.loads(file.read())
        if os.path.isfile(os.path.join(path, params_filename)):
            with open(os.path.join(path, params_filename)) as file:
                best_params = json.loads(file.read())
        if os.path.isfile(os.path.join(path, 'fitted_model.sav')):
            with open(os.path.join(path, 'fitted_model.sav'), 'rb') as file:
                grid_search = pickle.load(file)
        if not grid_search:
            if not best_params:  # if params are missing data can't be reconstructed
                raise Exception('Expected Data is missing in folder')
            else:
                warnings.warn('Model is missing and needs to be reconstructed. This may need some time')
                grid_params = _create_list_dict(best_params)

    if not grid_search:  # either not loaded or didn't performed yet
        grid_search = GridSearchCV(estimator=pipe, param_grid=grid_params, cv=cv, n_jobs=-2, verbose=2)
        grid_search.fit(x_g1, y_g1, groups=groups)
        # save fitted model
        with open(os.path.join(path, model_filename), 'wb') as file:
            pickle.dump(grid_search, file)
        if not best_params:  # if params weren't loaded (means full grid search)
            get_dataframe_from_cv_results(grid_search.cv_results_).to_csv(os.path.join(path, 'grid.csv'))

    if not best_params:  # if params weren't loaded
        # save params
        with open(os.path.join(path, params_filename), 'w') as file:
            best_params = grid_search.best_params_
            file.write(json.dumps(best_params))

    if not score:  # if score wasn't loaded
        score['mean_score_g1'] = grid_search.best_score_

    # scoring with G1 as training
    g2_predicted = grid_search.predict(x_g2)
    g2_actual = y_g2
    mean_score_g1 = score['mean_score_g1']
    score['accuracy_g1'] = accuracy_score(y_true=y_g2, y_pred=g2_predicted)

    # use G2 as training
    # cross validation
    pipe_with_params = pipe.set_params(**best_params)
    mean_score_g2 = np.mean(cross_val_score(pipe_with_params, x_g2, y=y_g2, cv=cv, groups=groups))
    score['mean_score_g2'] = mean_score_g2
    # train model with g2
    pipe_with_params.fit(x_g2, y_g2)
    g1_predicted = pipe_with_params.predict(x_g1)
    score['accuracy_g2'] = accuracy_score(y_true=y_g1, y_pred=g1_predicted)

    # save score
    score['mean_score_g1'] = grid_search.best_score_
    with open(os.path.join(path, score_filename), 'w') as file:
        file.write(json.dumps(score))

    return grid_search.best_estimator_, mean_score_g1, g2_predicted, y_g2, mean_score_g2, g1_predicted, y_g1


def eval_classifier_paper(features, target, patient_id, clf, grid_folder_name, grid_params=None):
    return eval_classifier(features, target, patient_id, clf, grid_folder_name, test_size=0.43,
                                 grid_params=grid_params, patient_cv=False)


def reconstruct_models_paper(grid_search: bool):
    paths = ['LDA_0916_hr10', 'DT_0916_hr10', 'RF_0916_hr10', 'MLP_0916_hr10', 'SVC_0916_hr10']
    functions = (get_lda_grid_params, get_dt_grid_params, get_rf_grid_params, get_mlp_grid_params, get_svm_grid_params)

    x, y, mean_error, coverage, patient_id = load_data(segment_length=10, overlap_amount=0)

    for path, function in zip(paths, functions):
        clf, params = function()
        if not grid_search:
            params = None
        print(path)
        eval_classifier_paper(x, y, patient_id, clf=clf, grid_folder_name=path, grid_params=params)


def get_all_scores(reconstruct: bool):
    if reconstruct:
        reconstruct_models_paper(grid_search=False)
    score_dict = {}
    filename = 'score.json'
    paths = ['RF_0916_hr10', 'SVC_0916_hr10', 'MLP_0916_hr10', 'LDA_0916_hr10', 'DT_0916_hr10']
    clf_names = ['RF', 'SVM', 'MLP', 'LDA', 'DT']
    for clf_name, folder in zip(clf_names, paths):
        location = folder + '/' + filename
        path = os.path.join(utils.get_grid_params_path(), location)
        if os.path.isfile(path):
            with open(path) as file:
                score = json.loads(file.read())
        else:
            raise Exception('No score file found')
        score_dict[clf_name] = score
    return score_dict


if __name__ == "__main__":
    pass
