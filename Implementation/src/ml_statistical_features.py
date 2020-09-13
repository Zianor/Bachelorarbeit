import jsonplus as json
import os
import pickle
import warnings

import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_statistical_features import DataSet
from utils import get_project_root


def load_data(segment_length=10, overlap_amount=0.9):
    """
    Loads BCG data features with its target labels
    :return: BCG data features, target labels
    """
    filename = 'data/data_statistical_features_l' + str(segment_length) + '_o' + str(overlap_amount) + '.csv'
    path = os.path.join(get_project_root(), filename)
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet()
    df = pd.read_csv(path)
    features = df.iloc[:, 0:13]
    target = df['informative']
    mean_error = df['mean error']
    coverage = df['coverage']
    patient_id = df['patient_id']  # TODO: do sth with it
    return features, target, mean_error, coverage, patient_id


def load_data_as_dataframe(segment_length=10, overlap_amount=0.9):
    """
    Loads BCG data as Dataframe
    :return: Dataframe
    """
    filename = 'data/data_statistical_features_l' + str(segment_length) + '_o' + str(overlap_amount) + '.csv'
    path = os.path.join(get_project_root(), filename)
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet()
    return pd.read_csv(path)


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
        'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'C': [1, 10],
        'class_weight': (None, 'balanced')
    }
    svm = SVC(random_state=1)

    return svm, parameters


def get_lda_grid_params():
    """Linear Discriminant Analysis
    :return: base estimator and dict of parameters for grid search
    """
    lda = LinearDiscriminantAnalysis()

    parameters = {
        'solver': ('svd', 'lsqr', 'eigen')
    }

    return lda, parameters


def get_dt_grid_params():
    """
    Decision tree
    :return: base estimator and dict of parameters for grid search
    """
    dt = DecisionTreeClassifier(random_state=1)

    parameters = {
        'criterion': ("gini", "entropy"),
        'splitter': ("best", "random"),
        'class_weight': (None, 'balanced')
    }

    return dt, parameters


def get_rf_grid_params():
    """
    Random forest
    :return: base estimator and dict of parameters for grid search
    """
    rf = RandomForestClassifier(random_state=1)

    parameters = {
        'n_estimators': [10, 30, 50, 75, 100],
        'criterion': ("gini", "entropy"),
        'class_weight': ("balanced", None)
    }

    return rf, parameters


def get_mlp_grid_params():
    """
    Multilayer Perceptron
    :return: base estimator and dict of parameters for grid search
    """
    mlp = MLPClassifier()

    parameters = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,)],
        'activation': ('identity', 'logistic', 'tanh', 'relu'),
        'alpha': [0.0001],
        'learning_rate': ('constant', 'invscaling', 'adaptive'),
        'learning_rate_init': [0.0001]
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


def eval_classifier_paper(features, target, clf, grid_folder_name, grid_params=None):
    grid_folder_name = 'data/grid_params/' + grid_folder_name
    if not os.path.isdir(os.path.join(get_project_root(), grid_folder_name)):
        if not grid_params:
            raise Exception("No existing folder and no params given")
        else:
            os.mkdir(path=os.path.join(get_project_root(), grid_folder_name))

    path = os.path.join(get_project_root(), grid_folder_name)
    model_filename = 'fitted_model.sav'
    params_filename = 'params.json'
    score_filename = 'score.json'

    # split in g1 and g2
    x_g1, x_g2, y_g1, y_g2 = train_test_split(features, target, test_size=0.43, random_state=1, stratify=target)

    # standardize with g1 as trainings set
    sc = StandardScaler()
    sc.fit(x_g1)
    x_train = sc.transform(x_g1)
    x_test = sc.transform(x_g2)
    y_train = y_g1
    y_test = y_g2

    k_fold = KFold(n_splits=10, shuffle=True, random_state=1)

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
        grid_search = GridSearchCV(estimator=clf, param_grid=grid_params, cv=k_fold, n_jobs=10)
        grid_search.fit(x_train, y_train)
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
        # save score
        score['score'] = grid_search.best_score_
        with open(os.path.join(path, score_filename), 'w') as file:
            file.write(json.dumps(score))

    # scoring with G1 as training
    g2_predicted = grid_search.predict(x_test)
    g2_actual = y_g2
    mean_score_g1 = score['score']

    # use G2 as training
    # standardize with G2 as training
    sc = StandardScaler()
    sc.fit(x_g2)
    x_train = sc.transform(x_g2)
    x_test = sc.transform(x_g1)
    y_train = y_g2
    y_test = y_g1
    # cross validation
    clf = clf.set_params(**best_params)
    mean_score_g2 = np.mean(cross_val_score(clf, x_train, y=y_train, cv=k_fold))
    # train model with g2
    clf.fit(x_train, y_train)
    g1_predicted = clf.predict(x_test)
    g1_actual = y_g1

    return grid_search.best_estimator_, mean_score_g1, g2_predicted, g2_actual, mean_score_g2, g1_predicted, g1_actual


def reconstruct_models_paper(grid_search: bool):
    paths = ['SVC_0717', 'LDA_0717', 'DT_0717', 'RF_0717', 'MLP_0717']
    functions = (get_svm_grid_params, get_lda_grid_params, get_dt_grid_params, get_rf_grid_params, get_mlp_grid_params)

    x, y, mean_error, coverage, patient_id = load_data(segment_length=10, overlap_amount=0)

    for path, function in zip(paths, functions):
        clf, params = function()
        if not grid_search:
            params = None
        eval_classifier_paper(x, y, clf=clf, grid_folder_name=path, grid_params=params)


if __name__ == "__main__":
    pass
