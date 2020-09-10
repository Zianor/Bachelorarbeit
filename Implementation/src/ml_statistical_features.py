import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
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


def data_preparation(features, target, reverse=False, partial=True):
    """
    Splits data in training and test data and standardizes features
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param partial: If true cross validation is performed on only one partial set
    :return: string_representation, x_train_std, x_test_std, y_train, y_test
    """
    # Split dataset in 2/3 training and 1/3 test data
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.333, random_state=1, stratify=target)

    string_representation = ['Labels counts in y:', str(np.bincount(target)), os.linesep, 'Labels counts in y_train:',
                             str(np.bincount(y_train)), os.linesep, 'Labels counts in y_test:',
                             str(np.bincount(y_test))]
    string_representation = ''.join(string_representation)

    if reverse:
        x_test, x_train, y_test, y_train = x_train, x_test, y_train, y_test

    # Standardizing features
    sc = StandardScaler()
    sc.fit(x_train)
    if partial:
        x_train_std = sc.transform(x_train)
        x_test_std = sc.transform(x_test)
    else:
        x_train_std = sc.transform(features)
        x_test_std = None
        y_train = target
        y_test = None
    return string_representation, x_train_std, x_test_std, y_train, y_test


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
                             "Classification report", os.linesep, str(classification_report(y_actual, y_pred, target_names=['non-informative', 'informative']))
                             ]

    _, _, mean_error, coverage, _ = load_data()

    avg_mean_error = calc_avg_mean_error(mean_error, y_pred, y_actual)
    string_representation.extend(["Average mean error: ", str(avg_mean_error), os.linesep])

    coverage = calc_coverage(coverage, y_pred, y_actual)
    string_representation.extend(["Coverage: ", str(coverage), os.linesep])

    return ''.join(string_representation)


def support_vector_machine(features, target, reverse=False, plot_roc=False):
    """
    Support vector machine
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    parameters = {
        'kernel': ('linear', 'rbf', 'poly', 'sigmoid'),
        'C': [1, 10],
        'class_weight': (None, 'balanced')
    }
    svm = SVC(random_state=1)

    evaluation = classifier(svm, parameters, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Support Vector Machine - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Support Vector Machine", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def linear_discriminant_analysis(features, target, reverse=False, plot_roc=False):
    """
    Linear Discriminant Analysis
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    lda = LinearDiscriminantAnalysis()

    parameters = {
        'solver': ('svd', 'lsqr', 'eigen')
    }

    evaluation = classifier(lda, parameters, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Linear Discriminant Analysis - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Linear Discriminant Analysis", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def decision_tree(features, target, reverse=False, plot_roc=False):
    """
    Decision tree
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    dt = DecisionTreeClassifier(random_state=1)

    parameters = {
        'criterion': ("gini", "entropy"),
        'splitter': ("best", "random"),
        'class_weight': (None, 'balanced')
    }

    evaluation = classifier(dt, parameters, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Decision Tree - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Decision Tree", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def random_forest(features, target, reverse=False, plot_roc=False):
    """
    Random forest
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    rf = RandomForestClassifier(random_state=1)

    parameters = {
        'n_estimators': [10, 30, 50, 75, 100],
        'criterion': ("gini", "entropy"),
        'class_weight': ("balanced", None)
    }

    evaluation = classifier(rf, parameters, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Random Forest - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Random Forest", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def multilayer_perceptron(features, target, reverse=False, plot_roc=False):
    """
    Multilayer Perceptron
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    mlp = MLPClassifier()

    parameters = {
        'hidden_layer_sizes': [(10,), (20,), (30,), (40,), (50,), (60,), (70,), (80,), (90,), (100,)],
        'activation': ('identity', 'logistic', 'tanh', 'relu'),
        'alpha': [0.0001],
        'learning_rate': ('constant', 'invscaling', 'adaptive'),
        'learning_rate_init': [0.0001]
    }

    evaluation = classifier(mlp, parameters, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Multilayer Perceptron - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Multilayer Perceptron", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def classifier(clf, parameters, features, target, reverse=False, plot_roc=False):
    """
    Trains and tests a classifier
    :param clf: The classifier to be trained
    :param parameters: parameter dict for grid search
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: evaluation
    :rtype: String
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target, partial=True, reverse=reverse)

    y_train = y_train.to_numpy()

    k_fold = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)

    grid_search = GridSearchCV(estimator=clf, param_grid=parameters, cv=k_fold, n_jobs=10)  # TODO: evtl. anpassen
    grid_search.fit(x_train_std, y_train)

    evaluation = [os.linesep, "Best Score ", str(grid_search.best_score_), os.linesep,
                  "Best Params", str(grid_search.best_estimator_), os.linesep,
                  "Grid ", os.linesep, pd.DataFrame(grid_search.cv_results_).to_string(), os.linesep]

    y_pred = grid_search.predict(x_test_std)

    if plot_roc:
        roc_display = plot_roc_curve(clf, x_test_std, y_test)
        plt.show()

    evaluation.append(evaluate_model(y_test, y_pred))

    return ''.join(evaluation)


def evaluate_all(plot_roc=False):
    """
    Trains and tests all implemented models
    :return: evaluation
    :rtype: String
    """
    x, y, mean_error, coverage = load_data()
    string_representation = [get_data_metrics(x, y, mean_error, coverage), os.linesep, os.linesep,
                             support_vector_machine(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             linear_discriminant_analysis(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             decision_tree(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             random_forest(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             multilayer_perceptron(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             ]
    return ''.join(string_representation)


def evaluate_paper_statistical_features():
    """
    Evaluates all models according to the paper "https://ieeexplore.ieee.org/document/7591234 and prints results"
    """
    x, y, mean_error, coverage, patient_id = load_data()
    print(get_data_metrics(x, y, mean_error, coverage), os.linesep)
    print(support_vector_machine(x, y), os.linesep, os.linesep)
    print(linear_discriminant_analysis(x, y), os.linesep, os.linesep)
    print(decision_tree(x, y), os.linesep, os.linesep)
    print(random_forest(x, y), os.linesep, os.linesep)
    print(multilayer_perceptron(x, y), os.linesep, os.linesep)
    print(support_vector_machine(x, y, reverse=True), os.linesep, os.linesep)
    print(linear_discriminant_analysis(x, y, reverse=True), os.linesep, os.linesep)
    print(decision_tree(x, y, reverse=True), os.linesep, os.linesep)
    print(random_forest(x, y, reverse=True), os.linesep, os.linesep)
    print(multilayer_perceptron(x, y, reverse=True))


def get_data_metrics(features, target, mean_error, coverage):
    """
    Returns a description of the given data, incl. mean bbi error, coverage and training and test group
    :param features: feature matrix
    :param target: target vector
    :param mean_error: mean bbi error for each segment
    :type: pandas.Series
    :param coverage: coverage for each segment
    :type: pandas.Series
    :return: data description
    :rtype: String
    """
    data_description, _, _, y_train, y_test = data_preparation(features, target)

    string_representation = [data_description, os.linesep]

    string_representation.extend(
        ["Mean bbi error on all data: ", str(calc_avg_mean_error(mean_error, target)), os.linesep])
    string_representation.extend(["Coverage on all data: ", str(calc_coverage(coverage, target)), os.linesep])

    string_representation.extend(
        ["Mean bbi error on training set: ", str(calc_avg_mean_error(mean_error, y_train)), os.linesep])
    string_representation.extend(["Coverage on training set: ", str(calc_coverage(coverage, y_train)), os.linesep])

    string_representation.extend(
        ["Mean bbi error on test set: ", str(calc_avg_mean_error(mean_error, y_test)), os.linesep])
    string_representation.extend(["Coverage on test set: ", str(calc_coverage(coverage, y_test)), os.linesep])

    return ''.join(string_representation)


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


if __name__ == "__main__":
    # print(evaluate_all(True))
    evaluate_paper_statistical_features()
    sys.exit(0)
