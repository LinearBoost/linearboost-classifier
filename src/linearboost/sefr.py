from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self

import numpy as np
from numbers import Integral, Real
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import _check_sample_weight, check_is_fitted
from sklearn.utils._param_validation import Interval, StrOptions

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

    kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='linear'
      Specifies the kernel type to be used in the algorithm.
      If a callable is given, it is used to pre-compute the kernel matrix.
      If 'precomputed', X is assumed to be a kernel matrix.

    gamma : float, default=None
      Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If None, then it is
      set to 1.0 / n_features. Ignored when kernel='precomputed'.

    degree : int, default=3
      Degree for 'poly' kernels. Ignored by other kernels.

    coef0 : float, default=1
      Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes, )
      A list of class labels known to the classifier.

    coef_ : ndarray of shape (1, n_features) or (1, n_samples)
      Coefficient of the features in the decision function. When a kernel is used,
      the shape will be (1, n_samples).

    intercept_ : ndarray of shape (1,)
      Intercept (a.k.a. bias) added to the decision function.

      If `fit_intercept` is set to False, the intercept is set to zero.

    n_features_in_ : int
      Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
      Names of features seen during :term:`fit`. Defined only when `X`
      has feature names that are all strings.

    X_fit_ : ndarray of shape (n_samples, n_features)
      The training data, stored when a kernel is used (except for 'precomputed').

    Notes
    -----
    This classifier only supports binary classification tasks.

    Examples
    --------
    >>> from linearboost import SEFR
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = SEFR(kernel='rbf').fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.score(X, y)
    0.89...
    """

    _parameter_constraints: dict = {
        "fit_intercept": ["boolean"],
        "kernel": [
            StrOptions({"linear", "poly", "rbf", "sigmoid", "precomputed"}),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Integral, 1, None, closed="left"), None],
        "coef0": [Real, None],
    }

    def __init__(
        self,
        *,
        fit_intercept=True,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
    ):
        self.fit_intercept = fit_intercept
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

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
        if self.kernel == "precomputed":
            X = validate_data(
                self,
                X,
                dtype="numeric",
                force_all_finite=True,
                reset=False,
            )
            # For precomputed kernels during prediction, X should be (n_test_samples, n_train_samples)
            if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"Precomputed kernel matrix should have {self.n_features_in_} columns "
                    f"(number of training samples), got {X.shape[1]}."
                )
        else:
            X = validate_data(
                self,
                X,
                dtype="numeric",
                force_all_finite=True,
                reset=False,
            )
            if hasattr(self, "n_features_in_") and X.shape[1] != self.n_features_in_:
                raise ValueError(
                    "Expected input with %d features, got %d instead."
                    % (self.n_features_in_, X.shape[1])
                )
        return X

    def _check_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        if self.kernel == "precomputed":
            # For precomputed kernels, X should be a square kernel matrix
            X, y = check_X_y(
                X,
                y,
                dtype="numeric",
                force_all_finite=True,
                estimator=self,
            )
            if X.shape[0] != X.shape[1]:
                raise ValueError(
                    f"Precomputed kernel matrix should be square, got shape {X.shape}."
                )
        else:
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

    def _get_kernel_matrix(self, X, Y=None):
        if self.kernel == "precomputed":
            # X is already a kernel matrix
            return X

        if Y is None:
            Y = self.X_fit_

        if callable(self.kernel):
            return self.kernel(X, Y)
        else:
            return pairwise_kernels(
                X,
                Y,
                metric=self.kernel,
                filter_params=True,
                gamma=self.gamma,
                degree=self.degree,
                coef0=self.coef0,
            )

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None) -> Self:
        """
        Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or (n_samples, n_samples)
          Training vector, where `n_samples` is the number of samples and
          `n_features` is the number of features.
          If kernel='precomputed', X should be a square kernel matrix.

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
        if self.kernel == "precomputed":
            _check_n_features(self, X=X, reset=True)
            _check_feature_names(self, X=X, reset=True)
        else:
            _check_n_features(self, X=X, reset=True)
            _check_feature_names(self, X=X, reset=True)

        X, y = self._check_X_y(X, y)

        # Store training data only for non-precomputed kernels
        if self.kernel != "precomputed":
            self.X_fit_ = X

        self.classes_, y_ = np.unique(y, return_inverse=True)

        if self.kernel == "linear":
            K = X
        elif self.kernel == "precomputed":
            K = X  # X is already the kernel matrix
        else:
            K = self._get_kernel_matrix(X)

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

        avg_pos = np.average(K[pos_labels, :], axis=0, weights=pos_sample_weight)
        avg_neg = np.average(K[neg_labels, :], axis=0, weights=neg_sample_weight)
        self.coef_ = (avg_pos - avg_neg) / (avg_pos + avg_neg + 1e-7)
        self.coef_ = np.reshape(self.coef_, (1, -1))

        if self.fit_intercept:
            scores = safe_sparse_dot(K, self.coef_.T, dense_output=True)
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

    def decision_function(self, X):
        check_is_fitted(self)
        X = self._check_X(X)

        if self.kernel == "linear":
            K = X
        elif self.kernel == "precomputed":
            K = X  # X is already a kernel matrix
        else:
            K = self._get_kernel_matrix(X)

        return (
            safe_sparse_dot(K, self.coef_.T, dense_output=True) + self.intercept_
        ).ravel()

    def predict_proba(self, X):
        """
        Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_train_samples)
          Vector to be scored, where `n_samples` is the number of samples and
          `n_features` is the number of features.
          If kernel='precomputed', X should have shape (n_samples, n_train_samples).

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
          Returns the probability of the sample for each class in the model,
          where classes are ordered as they are in ``self.classes_``.
        """
        check_is_fitted(self)
        norm_coef = np.linalg.norm(self.coef_)
        if norm_coef == 0:
            # Handle the case of a zero-norm coefficient vector to avoid division by zero
            score = self.decision_function(X)
        else:
            score = self.decision_function(X) / norm_coef
        proba = 1.0 / (1.0 + np.exp(-score))
        proba = np.clip(proba, 1e-9, 1 - 1e-9)
        return np.column_stack((1.0 - proba, proba))

    def predict_log_proba(self, X):
        """
        Predict logarithm of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or (n_samples, n_train_samples)
          Vector to be scored, where `n_samples` is the number of samples and
          `n_features` is the number of features.
          If kernel='precomputed', X should have shape (n_samples, n_train_samples).

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
          Returns the log-probability of the sample for each class in the
          model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))
