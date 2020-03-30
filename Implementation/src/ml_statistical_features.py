import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_statistical_features import DataSet
from utils import get_project_root


def load_data():
    """
    Loads BCG data features with its target labels
    :return: BCG data features, target labels
    """
    path = os.path.join(get_project_root(), 'data/data.csv')
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet()
    df = pd.read_csv(path)
    features = df.iloc[:, 0:12]
    target = df.iloc[:, 13]
    return features, target


def data_preparation(features, target):
    """
    Splits data in training and test data and standardizes features
    :param features: feature matrix
    :param target: target vector
    :return: string_representation, x_train_std, x_test_std, y_train, y_test
    """
    # Split dataset in 2/3 training and 1/3 test data
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.333, random_state=1, stratify=target)

    string_representation = ['Labels counts in y:', str(np.bincount(
        target)), os.linesep, 'Labels counts in y_train:', str(np.bincount(
        y_train)), os.linesep, 'Labels counts in y_test:', str(np.bincount(y_test))]
    string_representation = ''.join(string_representation)

    # Standardizing features
    sc = StandardScaler()
    sc.fit(x_train)
    x_train_std = sc.transform(x_train)
    x_test_std = sc.transform(x_test)
    return string_representation, x_train_std, x_test_std, y_train, y_test


def evaluate(y_actual, y_pred):
    """
    Evaluates predicted labels
    :param y_actual: actual labels
    :param y_pred: predicted labels
    """
    string_representation = ["Misclassified examples: %d" % (y_actual != y_pred).sum(), os.linesep,
                             "Accuracy: %.3f" % accuracy_score(y_actual, y_pred), os.linesep,
                             "Confusion matrix", os.linesep, str(confusion_matrix(y_actual, y_pred))]
    return ''.join(string_representation)


def support_vector_machine(features, target):
    """
    Support vector machine
    :param features: feature matrix
    :param target: target vector
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=1)
    svm.fit(x_train_std, y_train)

    y_pred = svm.predict(x_test_std)

    string_representation = ["Support Vector machine", os.linesep, str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


def support_vector_machine_cross_validation(features, target, k=10):
    """
    Support vector machine with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train SVM
    svm = SVC(kernel='rbf', C=1.0, random_state=1)
    k_fold = model_selection.KFold(n_splits=k)
    y_train = y_train.to_numpy()
    results_k_fold = [
        svm.fit(x_train_std[train_index], y_train[train_index]).score(x_train_std[test_index], y_train[test_index]) for
        train_index, test_index in k_fold.split(x_train_std, y_train)]

    y_pred = svm.predict(x_test_std)

    string_representation = ["Support Vector Machine (cross validation)", os.linesep,
                             "Accuracy in cross validation: %.3f" % (np.mean(results_k_fold) * 100.0), os.linesep,
                             str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


def linear_discriminant_analysis(features, target):
    """
    Linear Discriminant Analysis
    :param features: feature matrix
    :param target: target vector
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train LDA
    lda = LinearDiscriminantAnalysis()  # no further information given
    lda.fit(x_train_std, y_train)

    y_pred = lda.predict(x_test_std)

    string_representation = ["Linear Discriminant Analysis", os.linesep, str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


def linear_discriminant_analysis_cross_validation(features, target, k=10):
    """
    Linear Discriminant Analysis with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train LDA
    lda = LinearDiscriminantAnalysis()  # no further information given
    k_fold = model_selection.KFold(n_splits=k)
    y_train = y_train.to_numpy()
    results_k_fold = [
        lda.fit(x_train_std[train_index], y_train[train_index]).score(x_train_std[test_index], y_train[test_index]) for
        train_index, test_index in k_fold.split(x_train_std, y_train)]

    y_pred = lda.predict(x_test_std)

    string_representation = ["Linear Discriminant Analysis (cross validation)", os.linesep,
                             "Accuracy in cross validation: %.3f" % (np.mean(results_k_fold) * 100.0), os.linesep,
                             str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


def decision_tree(features, target):
    """
    Decision tree
    :param features: feature matrix
    :param target: target vector
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train DT
    dt = DecisionTreeClassifier()  # no further information given
    dt.fit(x_train_std, y_train)

    y_pred = dt.predict(x_test_std)

    string_representation = ["Decision tree", os.linesep, str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


def decision_tree_cross_validation(features, target, k=10):
    """
    Decision tree with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :return: string representation of results
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target)

    # train DT
    dt = DecisionTreeClassifier()  # no further information given
    k_fold = model_selection.KFold(n_splits=k)
    y_train = y_train.to_numpy()
    results_k_fold = [
        dt.fit(x_train_std[train_index], y_train[train_index]).score(x_train_std[test_index], y_train[test_index]) for
        train_index, test_index in k_fold.split(x_train_std, y_train)]

    y_pred = dt.predict(x_test_std)

    string_representation = ["Decision (cross validation)", os.linesep,
                             "Accuracy in cross validation: %.3f" % (np.mean(results_k_fold) * 100.0), os.linesep,
                             str(evaluate(y_test, y_pred))]

    return ''.join(string_representation)


if __name__ == "__main__":
    X, y = load_data()
    data_string, _, _, _, _ = data_preparation(X, y)
    print(data_string)
    # print(support_vector_machine(X, y))
    # print(support_vector_machine_cross_validation(X, y))
    # print(linear_discriminant_analysis(X, y))
    # print(linear_discriminant_analysis_cross_validation(X, y))
    print(decision_tree(X, y))
    print(decision_tree_cross_validation(X, y))
    sys.exit(0)
