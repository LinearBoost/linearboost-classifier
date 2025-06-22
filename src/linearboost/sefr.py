from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import _check_sample_weight

from ._utils import (
    SKLEARN_V1_6_OR_LATER,
    _check_feature_names,
    _check_n_features,
    _fit_context,
    check_X_y,
    validate_data,
)

__all__ = ["SEFR"]


class SEFR(LinearClassifierMixin, BaseEstimator):
    """A Scalable, Efficient, and Fast (SEFR) classifier.

    SEFR is an ultra-low power binary linear classifier designed specifically for
    resource-constrained devices. It operates by computing feature weights based on
    class-wise averages, achieving linear time complexity in both training and inference.
    The algorithm provides comparable accuracy to state-of-the-art methods while being
    significantly more energy efficient.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
        A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features)
        Coefficient of the features in the decision function.

    intercept_ : ndarray of shape (1,)
        Intercept (a.k.a. bias) added to the decision function.

        If `fit_intercept` is set to False, the intercept is set to zero.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Notes
    -----
    This classifier only supports binary classification tasks.

    Examples
    --------
    >>> from linearboost import SEFR
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = SEFR().fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[1.00...e+000, 2.04...e-154],
           [1.00...e+000, 1.63...e-165]])
    >>> clf.score(X, y)
    0.86...
    """

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
    }

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            tags.classifier_tags.multi_class = False
            tags.classifier_tags.poor_score = True
            return tags

    def _more_tags(self) -> dict[str, bool]:
        return {
            "binary_only": True,
            "requires_y": True,
            "poor_score": True,
            "_xfail_checks": {
                "check_sample_weight_equivalence_on_dense_data": (
                    "In SEFR, setting a sample's weight to 0 can produce a different result than omitting the sample. "
                    "Such samples intentionally still affect the calculation of the intercept."
                )
            },
        }

    def _check_X(self, X) -> np.ndarray:
        X = validate_data(
            self,
            X,
            dtype="numeric",
            force_all_finite=True,
            reset=False,
        )
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "Expected input with %d features, got %d instead."
                % (self.n_features_in_, X.shape[1])
            )
        return X

    def _check_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(
            X,
            y,
            dtype="numeric",
            force_all_finite=True,
            estimator=self,
        )
        check_classification_targets(y)

        if np.unique(y).shape[0] == 1:
            raise ValueError("Classifier can't train when only one class is present.")
        if (y_type := type_of_target(y)) != "binary":
            if SKLEARN_V1_6_OR_LATER:
                msg = f"Only binary classification is supported. The type of the target is {y_type}."
            else:
                msg = "Unknown label type: non-binary"
            raise ValueError(msg)

        return X, y

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None) -> Self:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like of shape (n_samples,) default=None
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self
            Fitted estimator.
        """
        _check_n_features(self, X=X, reset=True)
        _check_feature_names(self, X=X, reset=True)

        X, y = self._check_X_y(X, y)
        self.classes_, y_ = np.unique(y, return_inverse=True)

        pos_labels = y_ == 1
        neg_labels = y_ == 0

        pos_sample_weight, neg_sample_weight = None, None
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)
            pos_sample_weight = (
                sample_weight[pos_labels]
                if len(sample_weight[pos_labels]) > 0
                else None
            )
            neg_sample_weight = (
                sample_weight[neg_labels]
                if len(sample_weight[neg_labels]) > 0
                else None
            )
            if np.all(pos_sample_weight == 0) or np.all(neg_sample_weight == 0):
                raise ValueError("SEFR requires 2 classes; got only 1 class.")

        avg_pos = np.average(X[pos_labels, :], axis=0, weights=pos_sample_weight)
        avg_neg = np.average(X[neg_labels, :], axis=0, weights=neg_sample_weight)
        self.coef_ = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1e-7)
        self.coef_ = np.reshape(self.coef_, (1, -1))

        if self.fit_intercept:
            scores = safe_sparse_dot(X, self.coef_.T, dense_output=True)
            pos_score_avg = np.average(
                scores[pos_labels][:, 0], weights=pos_sample_weight
            )
            neg_score_avg = np.average(
                scores[neg_labels][:, 0], weights=neg_sample_weight
            )
            pos_label_count = np.count_nonzero(y_)
            neg_label_count = y_.shape[0] - pos_label_count
            bias = (
                neg_label_count * pos_score_avg + pos_label_count * neg_score_avg
            ) / (neg_label_count + pos_label_count)
            self.intercept_ = np.array([-bias])
        else:
            self.intercept_ = np.zeros(1)

        return self

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        score = self.decision_function(X) / np.linalg.norm(self.coef_)
        proba = 1.0 / (1.0 + np.exp(-score))
        return np.column_stack((1.0 - proba, proba))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Vector to be scored, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))
