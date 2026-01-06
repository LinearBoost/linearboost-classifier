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
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

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
    
    kernel_approx : {'rff', 'nystrom'} or None, default=None
        Optional kernel approximation strategy for non-linear kernels.

        - 'rff': Use Random Fourier Features (RBFSampler). Only valid when
          ``kernel='rbf'``. Approximates the RBF kernel via an explicit
          low-dimensional feature map.
        - 'nystrom': Use Nyström approximation (Nystroem). Can be used with
          'rbf', 'poly', or 'sigmoid' kernels.
        - None: Use exact kernel with full Gram matrix (O(n^2) memory).

    n_components : int, default=256
        Dimensionality of the kernel feature map when using kernel approximation.
        Acts as the number of random features (for 'rff') or the rank of the
        approximation (for 'nystrom'). Must be >= 1.


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

    early_stopping : bool, default=False
        Whether to use early stopping to terminate training when validation
        score is not improving. If True, it requires ``n_iter_no_change`` to be set.
        
        If ``subsample < 1.0`` (subsampling is enabled), Out-of-Bag (OOB) evaluation
        is automatically used instead of a fixed validation split. This is more
        data-efficient as it uses all training data while still providing validation
        feedback. OOB evaluation uses samples not included in each iteration's
        subsample for validation.

    validation_fraction : float, default=0.1
        The proportion of training data to set aside as validation set for
        early stopping. Must be between 0 and 1. Only used if ``early_stopping`` is True
        and ``subsample >= 1.0`` (no subsampling). When subsampling is enabled,
        OOB evaluation is used instead and this parameter is ignored.

    n_iter_no_change : int, default=5
        Number of iterations with no improvement to wait before early stopping.
        Only used if ``early_stopping`` is True. Must be >= 1.

    tol : float, default=1e-4
        Tolerance for the optimization. When the loss or score is not improving
        by at least ``tol`` for ``n_iter_no_change`` consecutive iterations,
        convergence is considered to be reached and training stops.
        Only used if ``early_stopping`` is True. Must be >= 0.

    subsample : float, default=1.0
        The fraction of samples to be used for fitting the individual base
        learners. If smaller than 1.0 this results in Stochastic Gradient
        Boosting. `subsample` interacts with the parameter `n_estimators`.
        Choosing `subsample < 1.0` leads to a reduction of variance
        and an increase in bias. Values must be in the range `(0, 1]`.

    shrinkage : float, default=1.0
        Shrinkage parameter for regularization. Each estimator weight is
        multiplied by this factor. Values < 1.0 reduce the contribution of
        each base learner, helping to prevent overfitting and improve
        generalization. This is similar to the shrinkage used in gradient
        boosting methods.
        
        - If `shrinkage = 1.0`: no shrinkage (full weight)
        - If `shrinkage < 1.0`: apply shrinkage (e.g., 0.8 means 80% weight)
        
        Values must be in the range `(0, 1]`. Typical values are in the range
        `[0.8, 1.0]` for moderate regularization or `1.0` for no regularization.

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
        "kernel_approx": [StrOptions({"rff", "nystrom"}), None],
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "early_stopping": ["boolean"],
        "validation_fraction": [Interval(Real, 0, 1, closed="neither")],
        "n_iter_no_change": [Interval(Integral, 1, None, closed="left"), None],
        "tol": [Interval(Real, 0, None, closed="left")],
        "subsample": [Interval(Real, 0, 1, closed="right")],
        "shrinkage": [Interval(Real, 0, 1, closed="right")],
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
        kernel_approx=None,
        n_components=256,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=5,
        tol=1e-4,
        subsample=1.0,
        shrinkage=1.0,
    ):
        self.algorithm = algorithm
        self.scaler = scaler
        self.class_weight = class_weight
        self.loss_function = loss_function
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.kernel_approx = kernel_approx
        self.n_components = n_components
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.subsample = subsample
        self.shrinkage = shrinkage

        # Decide how SEFR sees the input:
        # - If we use a kernel approximation, the base estimator should work
        #   on explicit features (linear kernel).
        # - Otherwise:
        #     - 'linear' -> use SEFR with linear kernel
        #     - non-linear -> SEFR expects a precomputed Gram matrix
        try:
            if self.kernel_approx is not None or kernel == "linear":
                base_estimator = SEFR(kernel="linear")
            else:
                base_estimator = SEFR(kernel="precomputed")
        except (ValueError, TypeError):
            # If kernel is an array or invalid type, default to linear
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
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.subsample = subsample
        self.shrinkage = shrinkage

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
    def _use_kernel_approx(self) -> bool:
        """Return True if we should use kernel approximation."""
        return self.kernel != "linear" and self.kernel_approx is not None

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

        # ----- Kernel handling & approximation -----
        self.kernel_approx_ = None  # will be set if approximation is used
        self.X_fit_ = None
        self.K_train_ = None

        if self.kernel == "linear":
            # Pure linear: no kernel, no approximation
            training_data = X_transformed

        elif self._use_kernel_approx():
            # Use kernel approximation instead of full Gram matrix
            if self.kernel_approx == "rff":
                if self.kernel != "rbf":
                    raise ValueError(
                        "kernel_approx='rff' is only supported with kernel='rbf'. "
                        f"Got kernel='{self.kernel}'."
                    )
                # Ensure gamma is set
                gamma = self.gamma
                if gamma is None:
                    gamma = 1.0 / X_transformed.shape[1]

                self.kernel_approx_ = RBFSampler(
                    gamma=gamma,
                    n_components=self.n_components,
                    # random_state can be None; AdaBoost's randomness is separate
                )
            elif self.kernel_approx == "nystrom":
                self.kernel_approx_ = Nystroem(
                    kernel=self.kernel,
                    gamma=self.gamma,
                    degree=self.degree,
                    coef0=self.coef0,
                    n_components=self.n_components,
                    # random_state can be None
                )
            else:
                raise ValueError(
                    f"Unknown kernel_approx='{self.kernel_approx}'. "
                    "Valid options are 'rff', 'nystrom', or None."
                )

            training_data = self.kernel_approx_.fit_transform(X_transformed)

        else:
            # Exact kernel with full Gram matrix (original behavior)
            self.X_fit_ = X_transformed
            # Precompute kernel matrix ONCE for all estimators
            self.K_train_ = self._get_kernel_matrix(X_transformed)
            training_data = self.K_train_
        # ----- end kernel handling -----

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

        # Handle early stopping with validation split or OOB evaluation
        validation_data = None
        y_val = None
        training_data_val = None
        X_val_transformed = None  # Store original features for validation (needed for exact kernels)
        use_oob = False  # Flag to use OOB evaluation instead of fixed validation split
        
        # Use OOB evaluation if subsampling is enabled and early stopping is requested
        if (self.early_stopping and self.n_iter_no_change is not None and 
            self.subsample < 1.0):
            # Check if we can use OOB (skip for exact kernels)
            is_exact_kernel = (not self._use_kernel_approx() and self.kernel != "linear")
            if not is_exact_kernel:
                use_oob = True
                # Store full data for OOB evaluation
                # For exact kernels, we need to store original features, not kernel matrix
                if not self._use_kernel_approx() and self.kernel != "linear":
                    # This shouldn't happen since we check is_exact_kernel above, but just in case
                    validation_data = (X_transformed, y, sample_weight)
                else:
                    validation_data = (training_data, y, sample_weight)
        
        if self.early_stopping and self.n_iter_no_change is not None and not use_oob:
            # Split BEFORE kernel computation for exact kernels
            # For exact kernels, we need to split X_transformed, not the kernel matrix
            if not self._use_kernel_approx() and self.kernel != "linear":
                # For exact kernels, split the original features
                n_samples = X_transformed.shape[0]
                n_val_samples = max(1, int(self.validation_fraction * n_samples))
                
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(
                    n_splits=1, 
                    test_size=n_val_samples, 
                    random_state=42
                )
                train_idx, val_idx = next(splitter.split(X_transformed, y))
                
                # Split original features
                X_train_transformed = X_transformed[train_idx]
                X_val_transformed = X_transformed[val_idx]
                y_train = y[train_idx]
                y_val = y[val_idx]
                
                # Recompute kernel matrix for training only
                self.X_fit_ = X_train_transformed
                self.K_train_ = self._get_kernel_matrix(X_train_transformed)
                training_data = self.K_train_
                
                # Split sample weights if provided
                if sample_weight is not None:
                    sample_weight_val = sample_weight[val_idx]
                    sample_weight = sample_weight[train_idx]
                else:
                    sample_weight_val = None
                
                # Store validation data (original features for kernel computation)
                validation_data = (X_val_transformed, y_val, sample_weight_val)
                y = y_train
            else:
                # For linear or approximate kernels, split after transformation
                n_samples = training_data.shape[0]
                n_val_samples = max(1, int(self.validation_fraction * n_samples))
                
                from sklearn.model_selection import StratifiedShuffleSplit
                splitter = StratifiedShuffleSplit(
                    n_splits=1, 
                    test_size=n_val_samples, 
                    random_state=42
                )
                train_idx, val_idx = next(splitter.split(training_data, y))
                
                # Split training data
                training_data_val = training_data[val_idx]
                y_val = y[val_idx]
                training_data = training_data[train_idx]
                y_train = y[train_idx]
                
                # Split sample weights if provided
                if sample_weight is not None:
                    sample_weight_val = sample_weight[val_idx]
                    sample_weight = sample_weight[train_idx]
                else:
                    sample_weight_val = None
                
                # Store validation data for checking
                validation_data = (training_data_val, y_val, sample_weight_val)
                y = y_train
        else:
            y_train = y

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
            
            # If early stopping is enabled, use custom boosting loop
            if self.early_stopping and self.n_iter_no_change is not None and validation_data is not None:
                return self._fit_with_early_stopping(training_data, y_train, sample_weight, validation_data, use_oob=use_oob)
            else:
                # Pass the precomputed kernel matrix (or raw features for linear)
                return super().fit(training_data, y_train, sample_weight)

    def _fit_with_early_stopping(self, X, y, sample_weight, validation_data, use_oob=False):
        """Fit with early stopping based on validation error or OOB evaluation.
        
        Parameters
        ----------
        X : array-like
            Training data (features or kernel matrix)
        y : array-like
            Training labels
        sample_weight : array-like
            Sample weights
        validation_data : tuple
            If use_oob=False: (X_val, y_val, sample_weight_val)
            If use_oob=True: (X_full, y_full, sample_weight_full) - full dataset for OOB
        use_oob : bool
            If True, use OOB samples for validation instead of fixed split
        """
        if use_oob:
            # For OOB, validation_data contains the full dataset
            X_full, y_full, sample_weight_full = validation_data
            # We'll track OOB samples per iteration
            oob_indices_history = []
        else:
            # Traditional validation split
            X_val, y_val, sample_weight_val = validation_data
        
        # Initialize from parent class
        from sklearn.utils import check_random_state
        
        # Initialize attributes needed for boosting
        # Ensure estimator_ is set (needed by _make_estimator)
        if not hasattr(self, 'estimator_') or self.estimator_ is None:
            # Reuse the same logic from __init__ to create base estimator
            try:
                if self.kernel_approx is not None or self.kernel == "linear":
                    from .sefr import SEFR
                    self.estimator_ = SEFR(kernel="linear")
                else:
                    from .sefr import SEFR
                    self.estimator_ = SEFR(kernel="precomputed")
            except (ValueError, TypeError):
                from .sefr import SEFR
                self.estimator_ = SEFR(kernel="linear")
        
        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators, dtype=np.float64)
        
        # Initialize sample weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0], dtype=np.float64)
            sample_weight /= sample_weight.sum()
        
        random_state = check_random_state(None)
        
        # Track best validation score and iterations without improvement
        best_val_score = -np.inf
        n_no_improvement = 0
        best_n_estimators = 0
        
        # For OOB, we need to store X_fit_ reference for exact kernels
        if use_oob and hasattr(self, 'X_fit_') and self.X_fit_ is not None:
            # Store reference to training features for kernel computation
            pass  # Already stored
        
        # Early stopping loop
        for iboost in range(self.n_estimators):
            # Perform a single boost
            # For OOB, we need to track which samples were used
            if use_oob:
                boost_result = self._boost(
                    iboost, X, y, sample_weight, random_state, return_oob_indices=True
                )
                if len(boost_result) == 4:
                    sample_weight, estimator_weight, estimator_error, oob_indices = boost_result
                    oob_indices_history.append(oob_indices)
                else:
                    sample_weight, estimator_weight, estimator_error = boost_result
                    oob_indices_history.append(None)
            else:
                sample_weight, estimator_weight, estimator_error = self._boost(
                    iboost, X, y, sample_weight, random_state
                )
            
            if sample_weight is None:
                break
            
            # Store results
            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error
            
            # Evaluate on validation set or OOB samples using F1/ROC-AUC
            if use_oob and len(oob_indices_history) > 0 and oob_indices_history[-1] is not None:
                # Use OOB samples from current iteration
                oob_idx = oob_indices_history[-1]
                if len(oob_idx) > 0:
                    # Get OOB data (X_full is already transformed features/kernel matrix)
                    X_oob = X_full[oob_idx]
                    y_oob = y_full[oob_idx]
                    
                    # Get predictions and probabilities for F1/ROC-AUC
                    val_pred = self._staged_predict_single(X_oob, iboost + 1)
                    val_proba = self._staged_predict_proba_single(X_oob, iboost + 1)
                    
                    # Compute F1 score (primary metric)
                    f1_val = f1_score(y_oob, val_pred, average='weighted', zero_division=0.0)
                    
                    # Compute ROC-AUC if possible (requires probabilities)
                    try:
                        if val_proba is not None and val_proba.shape[1] >= 2:
                            roc_auc_val = roc_auc_score(y_oob, val_proba[:, 1], average='weighted')
                            # Combined metric: 70% F1, 30% ROC-AUC
                            val_score = 0.7 * f1_val + 0.3 * roc_auc_val
                        else:
                            val_score = f1_val
                    except (ValueError, IndexError):
                        # Fallback to F1 only if ROC-AUC fails
                        val_score = f1_val
                else:
                    # No OOB samples (shouldn't happen with subsample < 1.0), skip validation
                    val_score = best_val_score
            else:
                # Traditional validation split
                val_pred = self._staged_predict_single(X_val, iboost + 1)
                val_proba = self._staged_predict_proba_single(X_val, iboost + 1)
                
                # Compute F1 score (primary metric)
                f1_val = f1_score(y_val, val_pred, average='weighted', zero_division=0.0)
                
                # Compute ROC-AUC if possible
                try:
                    if val_proba is not None and val_proba.shape[1] >= 2:
                        roc_auc_val = roc_auc_score(y_val, val_proba[:, 1], average='weighted')
                        # Combined metric: 70% F1, 30% ROC-AUC
                        val_score = 0.7 * f1_val + 0.3 * roc_auc_val
                    else:
                        val_score = f1_val
                except (ValueError, IndexError):
                    # Fallback to F1 only if ROC-AUC fails
                    val_score = f1_val
            
            # Check for improvement
            if val_score > best_val_score + self.tol:
                best_val_score = val_score
                n_no_improvement = 0
                best_n_estimators = iboost + 1
            else:
                n_no_improvement += 1
            
            # Early stopping check
            if n_no_improvement >= self.n_iter_no_change:
                # Trim estimators to best point
                if best_n_estimators > 0:
                    self.estimators_ = self.estimators_[:best_n_estimators]
                    self.estimator_weights_ = self.estimator_weights_[:best_n_estimators]
                    self.estimator_errors_ = self.estimator_errors_[:best_n_estimators]
                break
        
        return self
    
    def _staged_predict_single(self, X, n_estimators):
        """Predict using first n_estimators for validation.
        
        X can be either:
        - Transformed features (for linear/approximate kernels)
        - Kernel matrix (for exact kernels)
        - Original features (for exact kernels - will compute kernel)
        """
        if n_estimators == 0:
            # Return majority class
            return np.full(X.shape[0], self.classes_[0])
        
        # For exact kernels, if X is original features, compute kernel matrix
        if (not self._use_kernel_approx() and self.kernel != "linear" and 
            hasattr(self, 'X_fit_') and self.X_fit_ is not None and
            X.shape[1] == self.X_fit_.shape[1] and X.shape[1] != self.X_fit_.shape[0]):
            # X appears to be original features, compute kernel matrix
            X = self._get_kernel_matrix(X, self.X_fit_)
        
        if self.algorithm == "SAMME.R":
            classes = self.classes_
            n_classes = len(classes)
            
            pred = sum(
                self._samme_proba(estimator, n_classes, X)
                for estimator in self.estimators_[:n_estimators]
            )
            if n_estimators > 0:
                weights_sum = self.estimator_weights_[:n_estimators].sum()
                if weights_sum > 0:
                    pred /= weights_sum
            if n_classes == 2:
                pred[:, 0] *= -1
                decision = pred.sum(axis=1)
            else:
                decision = pred
        else:
            # SAMME algorithm
            classes = self.classes_
            pred = np.zeros((X.shape[0], n_classes))
            
            for i, estimator in enumerate(self.estimators_[:n_estimators]):
                predictions = estimator.predict(X)
                for j, class_label in enumerate(classes):
                    pred[:, j] += (
                        self.estimator_weights_[i] * (predictions == class_label)
                    )
            
            decision = pred
        
        if self.n_classes_ == 2:
            return self.classes_.take((decision > 0).astype(int), axis=0)
        else:
            return self.classes_.take(np.argmax(decision, axis=1), axis=0)
    
    def _staged_predict_proba_single(self, X, n_estimators):
        """Predict probabilities using first n_estimators for validation.
        
        Similar to _staged_predict_single but returns probabilities instead of predictions.
        
        Parameters
        ----------
        X : array-like
            Validation data (features or kernel matrix)
        n_estimators : int
            Number of estimators to use
            
        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities
        """
        if n_estimators == 0:
            # Return uniform probabilities
            return np.ones((X.shape[0], self.n_classes_)) / self.n_classes_
        
        # For exact kernels, if X is original features, compute kernel matrix
        if (not self._use_kernel_approx() and self.kernel != "linear" and 
            hasattr(self, 'X_fit_') and self.X_fit_ is not None and
            X.shape[1] == self.X_fit_.shape[1] and X.shape[1] != self.X_fit_.shape[0]):
            # X appears to be original features, compute kernel matrix
            X = self._get_kernel_matrix(X, self.X_fit_)
        
        if self.algorithm == "SAMME.R":
            # Use decision function and convert to probabilities
            # This matches how predict_proba works in the parent class
            classes = self.classes_
            n_classes = len(classes)
            
            pred = sum(
                self._samme_proba(estimator, n_classes, X)
                for estimator in self.estimators_[:n_estimators]
            )
            if n_estimators > 0:
                weights_sum = self.estimator_weights_[:n_estimators].sum()
                if weights_sum > 0:
                    pred /= weights_sum
                else:
                    # No valid weights, return uniform
                    return np.ones((X.shape[0], n_classes)) / n_classes
            
            # Convert SAMME.R output to probabilities
            # _samme_proba returns log-probability-like values (n_samples, n_classes)
            if n_classes == 2:
                # For binary: pred is 2D (n_samples, 2)
                # Convert to probabilities using softmax
                exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
                proba = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
                # Ensure probabilities are in correct order [class_0, class_1]
                # and sum to 1
                proba = np.clip(proba, 1e-9, 1 - 1e-9)
                proba = proba / np.sum(proba, axis=1, keepdims=True)
            else:
                # Multi-class: use softmax
                exp_pred = np.exp(pred - np.max(pred, axis=1, keepdims=True))
                proba = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)
            
            return proba
        else:
            # SAMME algorithm: use weighted voting
            classes = self.classes_
            n_classes = len(classes)
            proba = np.zeros((X.shape[0], n_classes))
            
            for i, estimator in enumerate(self.estimators_[:n_estimators]):
                if hasattr(estimator, 'predict_proba'):
                    estimator_proba = estimator.predict_proba(X)
                    weight = self.estimator_weights_[i]
                    proba += weight * estimator_proba
                else:
                    # Fallback: use predictions
                    predictions = estimator.predict(X)
                    weight = self.estimator_weights_[i]
                    for j, class_label in enumerate(classes):
                        proba[:, j] += weight * (predictions == class_label)
            
            # Normalize
            proba_sum = np.sum(proba, axis=1, keepdims=True)
            proba_sum[proba_sum == 0] = 1.0  # Avoid division by zero
            proba /= proba_sum
            
            return proba

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

    def _compute_adaptive_learning_rate(self, iboost, estimator_error, base_learning_rate):
        """
        Compute adaptive learning rate based on iteration and estimator error.
        
        Parameters
        ----------
        iboost : int
            Current boosting iteration index (0-based)
        estimator_error : float
            Classification error of the current estimator (0-0.5)
        base_learning_rate : float
            Base learning rate from user parameter
            
        Returns
        -------
        adaptive_lr : float
            Adaptive learning rate adjusted for iteration and error
        """
        # Exponential decay: reduce learning rate as we progress
        # Factor starts at 1.0 and decays to ~0.7 over all iterations
        iteration_decay = 1.0 - (iboost / max(self.n_estimators, 1)) * 0.3
        
        # Error-based adjustment: lower rate for high error estimators
        # High error (0.5) -> factor ~0.57, Low error (0.0) -> factor 1.0
        error_factor = 1.0 / (1.0 + estimator_error * 1.5)
        
        # Combine factors
        adaptive_lr = base_learning_rate * iteration_decay * error_factor
        
        # Clamp to reasonable range: at least 0.01, at most base_learning_rate
        adaptive_lr = np.clip(adaptive_lr, 0.01, base_learning_rate)
        
        return adaptive_lr

    def _boost(self, iboost, X, y, sample_weight, random_state, return_oob_indices=False):
        """
        Implement a single boost using precomputed kernel matrix or raw features.

        Parameters
        ----------
        X : ndarray
            For kernel methods, this is the precomputed kernel matrix.
            For linear methods, this is the raw feature matrix.
        return_oob_indices : bool, default=False
            If True, return OOB indices along with other results.
        """
        estimator = self._make_estimator(random_state=random_state)
        oob_indices = None
        
        # Apply subsampling if enabled
        # Note: For exact kernels (precomputed kernel matrices), subsampling is skipped
        # because it would require tracking subsample indices per estimator for correct prediction
        is_exact_kernel = (X.shape[0] == X.shape[1] and X.shape[0] == y.shape[0] and 
                          not self._use_kernel_approx() and self.kernel != "linear")
        
        if self.subsample < 1.0 and not is_exact_kernel:
            n_samples = X.shape[0]
            n_subsample = max(1, int(self.subsample * n_samples))
            
            # Use stratified sampling to maintain class distribution
            from sklearn.model_selection import StratifiedShuffleSplit
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=n_subsample,
                random_state=random_state.randint(0, 2**31 - 1)
            )
            subsample_idx, _ = next(splitter.split(X, y))
            
            # Track OOB indices if requested
            if return_oob_indices:
                all_indices = np.arange(n_samples)
                oob_indices = np.setdiff1d(all_indices, subsample_idx)
            
            # Subsample data and weights (for feature matrices, subsample rows only)
            X_subsample = X[subsample_idx]
            y_subsample = y[subsample_idx]
            if sample_weight is not None:
                sample_weight_subsample = sample_weight[subsample_idx].copy()
                # Normalize subsampled weights
                sample_weight_subsample /= sample_weight_subsample.sum()
            else:
                sample_weight_subsample = None
            
            # Fit estimator on subsampled data
            estimator.fit(X_subsample, y_subsample, sample_weight=sample_weight_subsample)
        else:
            # No subsampling - use all data
            estimator.fit(X, y, sample_weight=sample_weight)

        # Always evaluate on full dataset for proper error computation
        if self.algorithm == "SAMME.R":
            y_pred = estimator.predict(X)

            incorrect = y_pred != y
            estimator_error = np.mean(
                np.average(incorrect, weights=sample_weight, axis=0)
            )

            if estimator_error <= 0:
                if return_oob_indices:
                    return sample_weight, 1.0, 0.0, oob_indices
                return sample_weight, 1.0, 0.0
            elif estimator_error >= 0.5:
                if len(self.estimators_) > 1:
                    self.estimators_.pop(-1)
                if return_oob_indices:
                    return None, None, None, None
                return None, None, None

            # Compute adaptive learning rate
            adaptive_lr = self._compute_adaptive_learning_rate(
                iboost, estimator_error, self.learning_rate
            )
            
            # Compute F1 score for this estimator to inform weight calculation
            # This aligns estimator weighting with F1 optimization target
            f1 = f1_score(y, y_pred, sample_weight=sample_weight, average='weighted')
            
            # F1 bonus: reward estimators with good F1 performance
            # Scale: 0.5 F1 -> 1.0x multiplier, 1.0 F1 -> 1.2x multiplier
            # This ensures estimators contributing to F1 get higher weights
            f1_bonus = 1.0 + (f1 - 0.5) * 0.6
            
            # Compute base weight from error rate
            base_weight = np.log((1 - estimator_error) / max(estimator_error, 1e-10))
            
            # Apply F1 bonus to estimator weight
            estimator_weight = self.shrinkage * adaptive_lr * base_weight * f1_bonus

            if iboost < self.n_estimators - 1:
                # Compute class frequencies for imbalance handling
                # This gives higher weight boosts to minority class samples when misclassified
                unique_classes, class_counts = np.unique(y, return_counts=True)
                class_freq = class_counts / len(y)
                class_weights = {cls: 1.0 / freq for cls, freq in zip(unique_classes, class_freq)}
                
                # Apply class-aware weight updates (minority class gets higher boost)
                for cls in unique_classes:
                    cls_mask = y == cls
                    cls_weight = class_weights[cls]  # Inverse frequency weighting
                    sample_weight[cls_mask] = np.exp(
                        np.log(sample_weight[cls_mask] + 1e-10)
                        + estimator_weight * incorrect[cls_mask] * cls_weight 
                        * (sample_weight[cls_mask] > 0)
                    )
                
                # Normalize to prevent numerical issues
                sample_weight /= np.sum(sample_weight)

            if return_oob_indices:
                return sample_weight, estimator_weight, estimator_error, oob_indices
            return sample_weight, estimator_weight, estimator_error

        else:  # standard SAMME
            # Always evaluate on full dataset for proper error computation
            y_pred = estimator.predict(X)
            incorrect = y_pred != y
            estimator_error = np.mean(np.average(incorrect, weights=sample_weight))

            if estimator_error <= 0:
                if return_oob_indices:
                    return sample_weight, 1.0, 0.0, oob_indices
                return sample_weight, 1.0, 0.0
            if estimator_error >= 0.5:
                self.estimators_.pop(-1)
                if len(self.estimators_) == 0:
                    raise ValueError(
                        "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble cannot be fit."
                    )
                if return_oob_indices:
                    return None, None, None, None
                return None, None, None

            # Compute adaptive learning rate
            adaptive_lr = self._compute_adaptive_learning_rate(
                iboost, estimator_error, self.learning_rate
            )
            
            # Compute F1 score for this estimator to inform weight calculation
            # This aligns estimator weighting with F1 optimization target
            f1 = f1_score(y, y_pred, sample_weight=sample_weight, average='weighted')
            
            # F1 bonus: reward estimators with good F1 performance
            # Scale: 0.5 F1 -> 1.0x multiplier, 1.0 F1 -> 1.2x multiplier
            # This ensures estimators contributing to F1 get higher weights
            f1_bonus = 1.0 + (f1 - 0.5) * 0.6
            
            # Compute base weight from error rate
            base_weight = np.log((1.0 - estimator_error) / max(estimator_error, 1e-10))
            
            # Apply F1 bonus to estimator weight
            estimator_weight = self.shrinkage * adaptive_lr * base_weight * f1_bonus

            # Compute class frequencies for imbalance handling
            # This gives higher weight boosts to minority class samples when misclassified
            unique_classes, class_counts = np.unique(y, return_counts=True)
            class_freq = class_counts / len(y)
            class_weights = {cls: 1.0 / freq for cls, freq in zip(unique_classes, class_freq)}
            
            # Apply class-aware weight updates (minority class gets higher boost)
            for cls in unique_classes:
                cls_mask = y == cls
                cls_weight = class_weights[cls]  # Inverse frequency weighting
                sample_weight[cls_mask] *= np.exp(
                    estimator_weight * incorrect[cls_mask] * cls_weight
                )

            # Normalize sample weights
            sample_weight /= np.sum(sample_weight)

            if return_oob_indices:
                return sample_weight, estimator_weight, estimator_error, oob_indices
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

        # Decide which representation to use at prediction time:
        if self.kernel == "linear":
            test_data = X_transformed

        elif self._use_kernel_approx():
            # Apply the same feature map as during training
            if self.kernel_approx_ is None:
                raise RuntimeError(
                    "Kernel approximation object is not fitted. "
                    "This should not happen if 'fit' completed successfully."
                )
            test_data = self.kernel_approx_.transform(X_transformed)

        else:
            # Exact kernel: compute kernel matrix between test and training data
            if self.X_fit_ is None:
                raise RuntimeError(
                    "Training data for exact kernel is not stored. "
                    "This should not happen if 'fit' completed successfully."
                )
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