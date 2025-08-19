# This file is part of the LinearBoost project.
#
# Portions of this file are derived from scikit-learn
# Copyright (c) 2007–2024, scikit-learn developers (version 1.5)
# Licensed under the BSD 3-Clause License
# See https://github.com/scikit-learn/scikit-learn/blob/main/COPYING for details.
#
# Additional code and modifications:
#    - Hamidreza Keshavarz (hamid9@outlook.com) — machine learning logic, design, and new algorithms
#    - Mehdi Samsami (mehdisamsami@live.com) — software refactoring, compatibility with scikit-learn framework, and packaging
#
# The combined work is licensed under the MIT License.

from __future__ import annotations

import sys
import warnings
from abc import abstractmethod
from numbers import Integral, Real

if sys.version_info >= (3, 11):
    from typing import Self  # pragma: no cover
else:
    from typing_extensions import Self

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)
from sklearn.utils import compute_sample_weight
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from ._utils import SKLEARN_V1_6_OR_LATER, check_X_y, validate_data
from .sefr import SEFR

__all__ = ["LinearBoostClassifier"]

_scalers = {
    "minmax": MinMaxScaler(feature_range=(0, 1)),
    "quantile-uniform": QuantileTransformer(
        output_distribution="uniform", ignore_implicit_zeros=True
    ),
    "quantile-normal": QuantileTransformer(
        output_distribution="normal", ignore_implicit_zeros=True
    ),
    "normalizer-l1": Normalizer(norm="l1"),
    "normalizer-l2": Normalizer(norm="l2"),
    "normalizer-max": Normalizer(norm="max"),
    "standard": StandardScaler(),
    "power": PowerTransformer(method="yeo-johnson"),
    "maxabs": MaxAbsScaler(),
    "robust": RobustScaler(),
}


class _DenseAdaBoostClassifier(AdaBoostClassifier):
    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.sparse = False
            return tags

    def _check_X(self, X):
        # Only called to validate X in non-fit methods, therefore reset=False
        return validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
            reset=False,
        )

    @abstractmethod
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost.

        Warning: This method needs to be overridden by subclasses.

        Parameters
        ----------
        iboost : int
            The index of the current boost iteration.

        X : {array-like} of shape (n_samples, n_features) or (n_samples, n_samples)
            The training input samples. For kernel methods, this will be a
            precomputed kernel matrix.

        y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : array-like of shape (n_samples,)
            The current sample weights.

        random_state : RandomState
            The current random number generator

        Returns
        -------
        sample_weight : array-like of shape (n_samples,) or None
            The reweighted sample weights.
            If None then boosting has terminated early.

        estimator_weight : float
            The weight for the current boost.
            If None then boosting has terminated early.

        error : float
            The classification error for the current boost.
            If None then boosting has terminated early.
        """
        pass

    def staged_score(self, X, y, sample_weight=None):
        """Return staged scores for X, y.

        This generator method yields the ensemble score after each iteration of
        boosting and therefore allows monitoring, such as to determine the
        score on a test set after each boost.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            Labels for X.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Yields
        ------
        z : float
        """
        yield from super().staged_score(X, y, sample_weight)

    def staged_predict(self, X):
        """Return staged predictions for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        This generator method yields the ensemble prediction after each
        iteration of boosting and therefore allows monitoring, such as to
        determine the prediction on a test set after each boost.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Yields
        ------
        y : generator of ndarray of shape (n_samples,)
            The predicted classes.
        """
        yield from super().staged_predict(X)

    def staged_decision_function(self, X):
        """Compute decision function of ``X`` for each boosting iteration.

        This method allows monitoring (i.e. determine error on testing set)
        after each boosting iteration.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        score : generator of ndarray of shape (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        yield from super().staged_decision_function(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        return super().predict_proba(X)

    def staged_predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input sample is computed as
        the weighted mean predicted class probabilities of the classifiers
        in the ensemble.

        This generator method yields the ensemble predicted class probabilities
        after each iteration of boosting and therefore allows monitoring, such
        as to determine the predicted class probabilities on a test set after
        each boost.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Yields
        ------
        p : generator of ndarray of shape (n_samples,)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        yield from super().staged_predict_proba(X)

    def predict_log_proba(self, X):
        """Predict class log-probabilities for X.

        The predicted class log-probabilities of an input sample is computed as
        the weighted mean predicted class log-probabilities of the classifiers
        in the ensemble.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of
            outputs is the same of that of the :term:`classes_` attribute.
        """
        return super().predict_log_proba(X)


class LinearBoostClassifier(_DenseAdaBoostClassifier):
    """A LinearBoost classifier.

    A LinearBoost classifier is a meta-estimator based on AdaBoost and SEFR.
    It is a fast and accurate classification algorithm built to enhance the
    performance of the linear classifier SEFR.

    Parameters
    ----------
    n_estimators : int, default=200
        The maximum number of SEFR classifiers at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`, preferably `[10, 200]`.

    learning_rate : float, default=1.0
        Weight applied to each SEFR classifier at each boosting iteration. A higher
        learning rate increases the contribution of each SEFR classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`, preferably `(0.0, 1.0)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME'
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        If 'SAMME.R' then use the SAMME.R real boosting algorithm
        (implemented from scikit-learn = 1.5).
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

    scaler : str, default='minmax'
        Specifies the scaler to apply to the data. Options include:

        - 'minmax': Applies MinMaxScaler.
        - 'quantile-uniform': Uses QuantileTransformer with `output_distribution='uniform'`.
        - 'quantile-normal': Uses QuantileTransformer with `output_distribution='normal'`.
        - 'normalizer-l1': Applies Normalizer with `norm='l1'`.
        - 'normalizer-l2': Applies Normalizer with `norm='l2'`.
        - 'normalizer-max': Applies Normalizer with `norm='max'`.
        - 'standard': Uses StandardScaler.
        - 'power': Applies PowerTransformer with `method='yeo-johnson'`.
        - 'maxabs': Uses MaxAbsScaler.
        - 'robust': Applies RobustScaler.

    kernel : {'linear', 'poly', 'rbf', 'sigmoid'} or callable, default='linear'
        Specifies the kernel type to be used in the algorithm.
        If a callable is given, it is used to pre-compute the kernel matrix.

    gamma : float, default=None
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If None, then it is
        set to 1.0 / n_features.

    degree : int, default=3
        Degree for 'poly' kernels. Ignored by other kernels.

    coef0 : float, default=1
        Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.

    class_weight : {"balanced"}, dict or list of dicts, default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    loss_function : callable, default=None
        Custom loss function for optimization. Must follow the signature:

        ``loss_function(y_true, y_pred, sample_weight) -> float``

        where:
        - y_true: Ground truth (correct) target values.
        - y_pred: Estimated target values.
        - sample_weight: Sample weights (optional).

    Attributes
    ----------
    estimator_ : estimator
        The base estimator (SEFR) from which the ensemble is grown.

        .. versionadded:: scikit-learn 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: scikit-learn 1.2
            `base_estimator_` is deprecated and will be removed in scikit-learn 1.4.
            Use `estimator_` instead.

    estimators_ : list of classifiers
        The collection of fitted sub-estimators.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int
        The number of classes.

    estimator_weights_ : ndarray of floats
        Weights for each estimator in the boosted ensemble.

    estimator_errors_ : ndarray of floats
        Classification error for each estimator in the boosted
        ensemble.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    scaler_ : transformer
        The scaler instance used to transform the data.

    X_fit_ : ndarray of shape (n_samples, n_features)
        The training data after scaling, stored when kernel != 'linear'
        for prediction purposes.

    K_train_ : ndarray of shape (n_samples, n_samples)
        The precomputed kernel matrix on training data, stored when
        kernel != 'linear'.

    Notes
    -----
    This classifier only supports binary classification tasks.

    Examples
    --------
    >>> from linearboost import LinearBoostClassifier
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> clf = LinearBoostClassifier().fit(X, y)
    >>> clf.predict(X[:2, :])
    array([0, 0])
    >>> clf.predict_proba(X[:2, :])
    array([[0.88079708, 0.11920292],
           [0.88079708, 0.11920292]])
    >>> clf.score(X, y)
    0.97...
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0, None, closed="neither")],
        "algorithm": [StrOptions({"SAMME", "SAMME.R"})],
        "scaler": [StrOptions({s for s in _scalers})],
        "kernel": [StrOptions({"linear", "poly", "rbf", "sigmoid"}), callable],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "degree": [Interval(Integral, 1, None, closed="left"), None],
        "coef0": [Real, None],
        "class_weight": [
            StrOptions({"balanced"}),
            dict,
            list,
            None,
        ],
        "loss_function": [None, callable],
    }

    def __init__(
        self,
        n_estimators=200,
        *,
        learning_rate=1.0,
        algorithm="SAMME.R",
        scaler="minmax",
        class_weight=None,
        loss_function=None,
        kernel="linear",
        gamma=None,
        degree=3,
        coef0=1,
    ):
        # Create SEFR estimator with 'precomputed' kernel if we're using kernels
        # Use string comparison that's safe for arrays (will raise TypeError for arrays)
        try:
            if kernel == "linear":
                base_estimator = SEFR(kernel="linear")
            else:
                base_estimator = SEFR(kernel="precomputed")
        except (ValueError, TypeError):
            # If kernel is an array or invalid type, default to linear
            # Parameter validation will catch this later in fit()
            base_estimator = SEFR(kernel="linear")

        super().__init__(
            estimator=base_estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
        )
        self.algorithm = algorithm
        self.scaler = scaler
        self.class_weight = class_weight
        self.loss_function = loss_function
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
                    "In LinearBoostClassifier, setting a sample's weight to 0 can produce a different "
                    "result than omitting the sample. Such samples intentionally still affect the data scaling process."
                ),
                "check_sample_weights_invariance": (
                    "In LinearBoostClassifier, a zero sample_weight is not equivalent to removing the sample, "
                    "as samples with zero weight intentionally still affect the data scaling process."
                ),
            },
        }

    def _check_X_y(self, X, y) -> tuple[np.ndarray, np.ndarray]:
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            allow_nd=True,
            dtype=None,
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
        """Compute kernel matrix between X and Y.

        Parameters
        ----------
        X : array-like of shape (n_samples_X, n_features)
            Input samples.
        Y : array-like of shape (n_samples_Y, n_features), default=None
            Input samples. If None, use X.

        Returns
        -------
        K : ndarray of shape (n_samples_X, n_samples_Y)
            Kernel matrix.
        """
        if Y is None:
            Y = X

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

    def fit(self, X, y, sample_weight=None) -> Self:
        """Build a LinearBoost classifier from the training set (X, y).

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, the sample weights are initialized to
            1 / n_samples.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.algorithm not in {"SAMME", "SAMME.R"}:
            raise ValueError("algorithm must be 'SAMME' or 'SAMME.R'")

        if self.scaler not in _scalers:
            raise ValueError('Invalid scaler provided; got "%s".' % self.scaler)

        # Apply scaling
        if self.scaler == "minmax":
            self.scaler_ = clone(_scalers["minmax"])
        else:
            self.scaler_ = make_pipeline(
                clone(_scalers[self.scaler]), clone(_scalers["minmax"])
            )
        X_transformed = self.scaler_.fit_transform(X)
        y = np.asarray(y)

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.shape[0] != X_transformed.shape[0]:
                raise ValueError(
                    f"sample_weight.shape == {sample_weight.shape} is incompatible with X.shape == {X_transformed.shape}"
                )
            nonzero_mask = (
                sample_weight.sum(axis=1) != 0
                if sample_weight.ndim > 1
                else sample_weight != 0
            )
            X_transformed = X_transformed[nonzero_mask]
            y = y[nonzero_mask]
            sample_weight = sample_weight[nonzero_mask]

        X_transformed, y = self._check_X_y(X_transformed, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        # Store training data for kernel computation during prediction
        if self.kernel != "linear":
            self.X_fit_ = X_transformed
            # Precompute kernel matrix ONCE for all estimators
            self.K_train_ = self._get_kernel_matrix(X_transformed)
            training_data = self.K_train_
        else:
            training_data = X_transformed

        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight != "balanced":
                raise ValueError(
                    f'Valid preset for class_weight is "balanced". Given "{self.class_weight}".'
                )
            expanded_class_weight = compute_sample_weight(self.class_weight, y)

            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        with warnings.catch_warnings():
            if SKLEARN_V1_6_OR_LATER:
                warnings.filterwarnings(
                    "ignore",
                    category=FutureWarning,
                    message=".*parameter 'algorithm' may change in the future.*",
                )
            warnings.filterwarnings(
                "ignore",
                category=FutureWarning,
                message=".*parameter 'algorithm' is deprecated.*",
            )
            # Pass the precomputed kernel matrix (or raw features for linear)
            return super().fit(training_data, y, sample_weight)

    @staticmethod
    def _samme_proba(estimator, n_classes, X):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].

        References
        ----------
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

        """
        proba = estimator.predict_proba(X)

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        log_proba = np.log(proba)

        return (n_classes - 1) * (
            log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis]
        )

    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Implement a single boost using precomputed kernel matrix or raw features.

        Parameters
        ----------
        X : ndarray
            For kernel methods, this is the precomputed kernel matrix.
            For linear methods, this is the raw feature matrix.
        """
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        if self.algorithm == "SAMME.R":
            y_pred = estimator.predict(X)

            incorrect = y_pred != y
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weight, axis=0)
            )

            if estimator_error <= 0:
                return sample_weight, 1.0, 0.0
            elif estimator_error >= 0.5:
                if len(self.estimators_) > 1:
                    self.estimators_.pop(-1)
                return None, None, None

            # Compute SEFR-specific weight update
            estimator_weight = self.learning_rate * np.log(
                (1 - estimator_error) / estimator_error
            )

            if iboost < self.n_estimators - 1:
                sample_weight = np.exp(
                    np.log(sample_weight)
                    + estimator_weight * incorrect * (sample_weight > 0)
                )

            return sample_weight, estimator_weight, estimator_error

        else:  # standard SAMME
            y_pred = estimator.predict(X)
            incorrect = y_pred != y
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight))

            if estimator_error <= 0:
                return sample_weight, 1.0, 0.0
            if estimator_error >= 0.5:
                self.estimators_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError(
                        "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble cannot be fit."
                    )
                return None, None, None

            estimator_weight = self.learning_rate * np.log(
                (1.0 - estimator_error) / max(estimator_error, 1e-10)
            )

            sample_weight *= np.exp(estimator_weight * incorrect)

            # Normalize sample weights
            sample_weight /= np.sum(sample_weight)

            return sample_weight, estimator_weight, estimator_error

    def decision_function(self, X):
        """Compute the decision function of ``X``.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        score : ndarray of shape of (n_samples, k)
            The decision function of the input samples. The order of
            outputs is the same as that of the :term:`classes_` attribute.
            Binary classification is a special cases with ``k == 1``,
            otherwise ``k==n_classes``. For binary classification,
            values closer to -1 or 1 mean more like the first or second
            class in ``classes_``, respectively.
        """
        check_is_fitted(self)
        X_transformed = self.scaler_.transform(X)

        if self.kernel == "linear":
            # For linear kernel, pass raw features
            test_data = X_transformed
        else:
            # For kernel methods, compute kernel matrix between test and training data
            test_data = self._get_kernel_matrix(X_transformed, self.X_fit_)

        if self.algorithm == "SAMME.R":
            # Proper SAMME.R decision function
            classes = self.classes_
            n_classes = len(classes)

            pred = sum(
                self._samme_proba(estimator, n_classes, test_data)
                for estimator in self.estimators_
            )
            pred /= self.estimator_weights_.sum()
            if n_classes == 2:
                pred[:, 0] *= -1
                return pred.sum(axis=1)
            return pred

        else:
            # Standard SAMME algorithm from AdaBoostClassifier (discrete)
            return super().decision_function(test_data)

    def predict(self, X):
        """Predict classes for X.

        The predicted class of an input sample is computed as the weighted mean
        prediction of the classifiers in the ensemble.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes.
        """
        pred = self.decision_function(X)

        if self.n_classes_ == 2:
            return self.classes_.take(pred > 0, axis=0)

        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
