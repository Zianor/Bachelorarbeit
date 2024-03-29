import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score


class OwnClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model):
        """Initialize self
        """
        self.model = model

    def fit(self, X, y):
        """Fits the underlying model to a binary label

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        param y: target label"""
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
        self.classes_ = [False, True]  # order important for AUC
        return self

    def predict(self, X):
        """Predict class for X

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        :returns: np array of shape (n_samples,) with binary labels"""
        if type(X) is not pd.DataFrame:
            X = pd.DataFrame(X)
        mask_nan = X.isna().any(axis=1)
        X_not_na = X.loc[~mask_nan]
        y_pred = self.model.predict(X_not_na)
        y = pd.Series(index=X.index, data=np.full((len(X.index),), False), name='pred')
        y.loc[X_not_na.index] = pd.Series(y_pred, X_not_na.index, dtype=bool)
        return y.to_numpy()

    def get_params(self, deep=True):
        return {'model': self.model}

    def set_params(self, **params):
        self.model.set_params(**params)
        return self

    def predict_proba(self, X):
        """Predicts the probability of both classes for given feature vectors

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        :returns: class probabilites in the order [False, True] in form (n_samples,2)"""
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


class RegressionClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, model, threshold=10, scale_sqrt=4):
        """Initialize self
        """
        self.model = model
        self.threshold = threshold
        self.scale_sqrt = scale_sqrt

    def fit(self, X, y):
        """Fits the underlying model to a continuous target

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        :param y: needs to be continuous target
        """
        y = np.power(y, 1/self.scale_sqrt)
        self.model.fit(X, y)
        self.classes_ = [False, True]  # order important for AUC?
        return self

    def predict(self, X):
        """Predict class for X

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        :returns: np array of shape (n_samples,) with binary labels"""
        y_continuous = self.model.predict(X)
        y_continuous = np.power(y_continuous, self.scale_sqrt)
        y = [False if curr > self.threshold else True for curr in y_continuous]
        return y

    def get_params(self, deep=True):
        return {'model': self.model,
                'threshold': self.threshold,
                'scale_sqrt': self.scale_sqrt}

    def set_params(self, **params):
        if "scale_sqrt" in params:
            self.scale_sqrt = params.pop("scale_sqrt")
        self.model.set_params(**params)
        return self

    def predict_proba(self, X):
        """Predicts the probability of both classes for given feature vectors

        :param X: input samples {array-like, sparse matrix} of shape (n_samples, n_features}
        :returns: class probabilites in the order [False, True] in form (n_samples,2)"""
        y_continuous = self.model.predict(X)
        y_continuous = np.power(y_continuous, self.scale_sqrt)
        # e function with f(th)=0.5
        proba_true = np.array([np.math.exp(np.log(0.5)/self.threshold * y) for y in y_continuous])
        proba_false = 1 - proba_true
        ret = np.ones(shape=(len(y_continuous), 2))
        ret[:, 0] = proba_false
        ret[:, 1] = proba_true
        return ret

    def score(self, X, y):
        """Calculates accuracy score for given input-output pairs

        :return: accuracy score"""
        y = [False if curr > self.threshold else True for curr in y]
        return accuracy_score(y, self.predict(X))
