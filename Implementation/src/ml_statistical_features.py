import os
import warnings
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
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
    path = os.path.join(get_project_root(), 'data/data_statistical_features.csv')
    if not os.path.isfile(path):
        warnings.warn('No csv, data needs to be reproduced. This may take some time')
        DataSet()
    df = pd.read_csv(path)
    features = df.iloc[:, 0:13]
    target = df.iloc[:, 13]
    return features, target


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


def evaluate_model(y_actual_train, y_pred_train, y_actual_test, y_pred_test):
    """
    Evaluates model performance
    :param y_actual_train: Actual labels from training set
    :param y_pred_train: Predicted labels from training set
    :param y_actual_test: Actual labels from test set
    :param y_pred_test: Predicted labels from test set
    :return: evaluation of the model
    :rtype: String
    """
    string_representation = ["Training set", os.linesep,
                             evaluate(y_actual_train, y_pred_train), os.linesep,
                             "Test set", os.linesep,
                             evaluate(y_actual_test, y_pred_test)
                             ]

    return ''.join(string_representation)


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


def support_vector_machine(features, target, reverse=False, plot_roc=False):
    """
    Support vector machine
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    svm = SVC(kernel='rbf', C=1.0, random_state=1)

    evaluation = classifier(svm, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Support Vector Machine - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Support Vector Machine", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def support_vector_machine_cross_validation(features, target, k=10, partial=False, reverse=False):
    """
    Support vector machine with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: string representation of results
    """
    svm = SVC(kernel='rbf', C=1.0, random_state=1)

    evaluation = classifier_cross_validation(svm, features, target, k, partial=partial, reverse=reverse)

    if reverse:
        string_representation = ["Support Vector Machine (cross validation) - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Support Vector Machine (cross validation)", os.linesep,
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
    lda = LinearDiscriminantAnalysis()  # no further information given

    evaluation = classifier(lda, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Linear Discriminant Analysis - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Linear Discriminant Analysis", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def linear_discriminant_analysis_cross_validation(features, target, k=10, partial=False, reverse=False):
    """
    Linear Discriminant Analysis with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: string representation of results
    """
    lda = LinearDiscriminantAnalysis()  # no further information given

    evaluation = classifier_cross_validation(lda, features, target, k, partial=partial, reverse=reverse)

    if reverse:
        string_representation = ["Linear Discriminant Analysis (cross validation) - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Linear Discriminant Analysis (cross validation)", os.linesep,
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
    dt = DecisionTreeClassifier()  # no further information given

    evaluation = classifier(dt, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Decision Tree - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Decision Tree", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def decision_tree_cross_validation(features, target, k=10, partial=False, reverse=False):
    """
    Decision tree with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: string representation of results
    """
    dt = DecisionTreeClassifier()  # no further information given

    evaluation = classifier_cross_validation(dt, features, target, k, partial=partial, reverse=reverse)

    if reverse:
        string_representation = ["Decision Tree (cross validation) - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Decision Tree (cross validation)", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def random_forest(features, target, n_trees=50, reverse=False, plot_roc=False):
    """
    Random forest
    :param features: feature matrix
    :param target: target vector
    :param n_trees: The number of trees in the forest
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    rf = RandomForestClassifier(n_estimators=n_trees)  # no further information given

    evaluation = classifier(rf, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Random Forest - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Random Forest", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def random_forest_cross_validation(features, target, n_trees=50, k=10, partial=False, reverse=False):
    """
    Random Forest with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param n_trees: The number of trees in the forest
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: string representation of results
    """
    rf = RandomForestClassifier(n_estimators=n_trees)  # no further information given

    evaluation = classifier_cross_validation(rf, features, target, k, partial=partial, reverse=reverse)

    if reverse:
        string_representation = ["Random Forest (cross validation) - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Random Forest (cross validation)", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def multilayer_perceptron(features, target, hidden_nodes=50, reverse=False, plot_roc=False):
    """
    Multilayer Perceptron
    :param features: feature matrix
    :param target: target vector
    :param hidden_nodes: The number of neurons in the hidden layer
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: string representation of results
    """
    mlp = MLPClassifier(hidden_layer_sizes=hidden_nodes)

    evaluation = classifier(mlp, features, target, reverse=reverse, plot_roc=plot_roc)

    if reverse:
        string_representation = ["Multilayer Perceptron - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Multilayer Perceptron", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def multilayer_perceptron_cross_validation(features, target, hidden_nodes=50, k=10, partial=False, reverse=False):
    """
    Multilayer Perceptron with k-fold cross validation
    :param features: feature matrix
    :param target: target vector
    :param hidden_nodes: The number of neurons in the hidden layer
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: string representation of results
    """
    mlp = MLPClassifier(hidden_layer_sizes=hidden_nodes)

    evaluation = classifier_cross_validation(mlp, features, target, k, partial=partial, reverse=reverse)

    if reverse:
        string_representation = ["Multilayer Perceptron (cross validation) - Reversed", os.linesep,
                                 evaluation]
    else:
        string_representation = ["Multilayer Perceptron (cross validation)", os.linesep,
                                 evaluation]

    return ''.join(string_representation)


def classifier_cross_validation(clf, features, target, k=10, partial=False, reverse=False):
    """
    Trains and tests a classifier with k fold cross validation
    :param clf: The classifier to be trained
    :param features: feature matrix
    :param target: target vector
    :param k: number of folds
    :param partial: If true cross validation is performed on only one partial set
    :param reverse: Uses test set for training and training set for test
    :return: evaluation, results_k_fold
    :rtype: (String, array)
    """
    # even if partial is False, standardization is performed based on training set
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target, partial=partial, reverse=reverse)

    y_train = y_train.to_numpy()

    evaluation = []

    k_fold = model_selection.KFold(n_splits=k, shuffle=True, random_state=1)
    results_k_fold = cross_val_score(clf, x_train_std, y_train, cv=k_fold)

    evaluation.append("Accuracy in cross validation: %.3f" % (np.mean(results_k_fold) * 100.0))

    return ''.join(evaluation)


def classifier(clf, features, target, reverse=False, plot_roc=False):
    """
    Trains and tests a classifier
    :param clf: The classifier to be trained
    :param features: feature matrix
    :param target: target vector
    :param reverse: Uses test set for training and training set for test
    :param plot_roc: if True ROC curve will be plotted
    :return: evaluation
    :rtype: String
    """
    _, x_train_std, x_test_std, y_train, y_test = data_preparation(features, target, reverse=reverse)

    evaluation = []

    # train
    clf.fit(x_train_std, y_train)

    y_pred_train = clf.predict(x_train_std)
    y_pred_test = clf.predict(x_test_std)

    if plot_roc:
        roc_display = plot_roc_curve(clf, x_test_std, y_test)
        plt.show()

    evaluation.append(evaluate_model(y_train, y_pred_train, y_test, y_pred_test))

    return ''.join(evaluation)


def evaluate_all(plot_roc=False):
    """
    Trains and tests all implemented models
    :return: evaluation
    :rtype: String
    """
    x, y = load_data()
    data_string, _, _, _, _ = data_preparation(x, y)
    string_representation = [data_string, os.linesep, os.linesep,
                             support_vector_machine(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             support_vector_machine_cross_validation(x, y), os.linesep, os.linesep,
                             linear_discriminant_analysis(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             linear_discriminant_analysis_cross_validation(x, y), os.linesep, os.linesep,
                             decision_tree(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             decision_tree_cross_validation(x, y), os.linesep, os.linesep,
                             random_forest(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             random_forest_cross_validation(x, y), os.linesep, os.linesep,
                             multilayer_perceptron(x, y, plot_roc=plot_roc), os.linesep, os.linesep,
                             multilayer_perceptron_cross_validation(x, y)]
    return ''.join(string_representation)


def evaluate_paper_statistical_features():
    """
    Evaluates all models according to the paper "https://ieeexplore.ieee.org/document/7591234"
    :return: evaluation
    :rtype: String
    """
    x, y = load_data()
    data_string, _, _, _, _ = data_preparation(x, y)
    string_representation = [data_string, os.linesep,
                             "Cross Validation", os.linesep, os.linesep,
                             support_vector_machine_cross_validation(x, y, partial=True), os.linesep, os.linesep,
                             support_vector_machine_cross_validation(x, y, k=10, partial=True, reverse=True),
                             os.linesep, os.linesep,
                             linear_discriminant_analysis_cross_validation(x, y, partial=True), os.linesep, os.linesep,
                             linear_discriminant_analysis_cross_validation(x, y, k=10, partial=True, reverse=True),
                             os.linesep, os.linesep,
                             decision_tree_cross_validation(x, y, partial=True), os.linesep, os.linesep,
                             decision_tree_cross_validation(x, y, k=10, partial=True, reverse=True),
                             os.linesep, os.linesep,
                             random_forest_cross_validation(x, y, partial=True), os.linesep, os.linesep,
                             random_forest_cross_validation(x, y, k=10, partial=True, reverse=True),
                             os.linesep, os.linesep,
                             multilayer_perceptron_cross_validation(x, y, partial=True), os.linesep, os.linesep,
                             multilayer_perceptron_cross_validation(x, y, k=10, partial=True, reverse=True),
                             os.linesep, os.linesep,
                             "Accuracy", os.linesep, os.linesep,
                             support_vector_machine(x, y), os.linesep, os.linesep,
                             linear_discriminant_analysis(x, y), os.linesep, os.linesep,
                             decision_tree(x, y), os.linesep, os.linesep,
                             random_forest(x, y), os.linesep, os.linesep,
                             multilayer_perceptron(x, y), os.linesep, os.linesep,
                             support_vector_machine(x, y, reverse=True), os.linesep, os.linesep,
                             linear_discriminant_analysis(x, y, reverse=True), os.linesep, os.linesep,
                             decision_tree(x, y, reverse=True), os.linesep, os.linesep,
                             random_forest(x, y, reverse=True), os.linesep, os.linesep,
                             multilayer_perceptron(x, y, reverse=True)
                             ]
    return ''.join(string_representation)


if __name__ == "__main__":
    evaluate_all(True)
    sys.exit(0)
