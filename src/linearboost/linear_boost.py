from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
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
from sklearn.utils._param_validation import Hidden, Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted

from ._utils import SKLEARN_V1_6_OR_LATER, check_X_y
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


class LinearBoostClassifier(AdaBoostClassifier):
    """A LinearBoost classifier.

    A LinearBoost classifier is a meta-estimator based on AdaBoost and SEFR.
    It is a fast and accurate classification algorithm built to enhance the 
    performance of the linear classifier SEFR.

    Parameters
    ----------
    n_estimators : int, default=200
        The maximum number of SEFR classifiers at which boosting is terminated.
        In case of perfect fit, the learning procedure is stopped early.
        Values must be in the range `[1, inf)`, preferably in the range `[10, 200]`.

    learning_rate : float, default=1.0
        Weight applied to each SEFR classifier at each boosting iteration. A higher
        learning rate increases the contribution of each SEFR classifier. There is
        a trade-off between the `learning_rate` and `n_estimators` parameters.
        Values must be in the range `(0.0, inf)`, preferably in the range `(0.0, 1.0)`.

    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME'
        If 'SAMME' then use the SAMME discrete boosting algorithm.
        If 'SAMME.R' then use the SAMME.R real boosting algorithm.
        The SAMME.R algorithm typically converges faster than SAMME,
        achieving a lower test error with fewer boosting iterations.

        .. deprecated:: sklearn 1.6
            `algorithm` is deprecated and will be removed in sklearn 1.8. This
            estimator only implements the 'SAMME' algorithm.

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

    class_weight : {"balanced", "balanced_subsample"}, dict or list of dicts, \
            default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. 

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``

        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    loss_function : callable, default=None
        Custom loss function for optimization. Must follow the signature:
        
        ``loss_function(y_true, y_pred, sample_weight) -> float``
        
        where:
        - y_true: Ground truth (correct) target values.
        - y_pred: Estimated target values.
        - sample_weight: Sample weights.

    Attributes
    ----------
    estimator_ : estimator
        The base estimator (SEFR) from which the ensemble is grown.

        .. versionadded:: sklearn 1.2
           `base_estimator_` was renamed to `estimator_`.

    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

        .. deprecated:: sklearn 1.2
            `base_estimator_` is deprecated and will be removed in sklearn 1.4.
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

    Notes
    -----
    The classifier only supports binary classification tasks.

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
        "algorithm": [
            StrOptions({"SAMME", "SAMME.R"}),
            Hidden(StrOptions({"deprecated"})),
        ],
        "scaler": [StrOptions({s for s in _scalers})],
        "class_weight": [
            StrOptions({"balanced_subsample", "balanced"}),
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
        algorithm="SAMME",
        scaler="minmax",
        class_weight=None,
        loss_function=None,
    ):
        super().__init__(
            estimator=SEFR(),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            algorithm="deprecated" if SKLEARN_V1_6_OR_LATER else algorithm,
        )

        self.scaler = scaler
        self.class_weight = class_weight
        self.loss_function = loss_function

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.input_tags.sparse = False
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
                )
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

    def fit(self, X, y, sample_weight=None) -> "LinearBoostClassifier":
        X, y = self._check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        if self.scaler not in _scalers:
            raise ValueError('Invalid scaler provided; got "%s".' % self.scaler)

        if self.scaler == "minmax":
            self.scaler_ = clone(_scalers["minmax"])
        else:
            self.scaler_ = make_pipeline(
                clone(_scalers[self.scaler]), clone(_scalers["minmax"])
            )
        X_transformed = self.scaler_.fit_transform(X)

        if self.class_weight is not None:
            valid_presets = ("balanced", "balanced_subsample")
            if (
                isinstance(self.class_weight, str)
                and self.class_weight not in valid_presets
            ):
                raise ValueError(
                    "Valid presets for class_weight include "
                    '"balanced" and "balanced_subsample".'
                    'Given "%s".' % self.class_weight
                )
            expanded_class_weight = compute_sample_weight(self.class_weight, y)

            if sample_weight is not None:
                sample_weight = sample_weight * expanded_class_weight
            else:
                sample_weight = expanded_class_weight

        return super().fit(X_transformed, y, sample_weight)

    def _boost(self, iboost, X, y, sample_weight, random_state):
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_pred = estimator.predict(X)
        missclassified = y_pred != y

        if self.loss_function:
            estimator_error = self.loss_function(y, y_pred, sample_weight)
        else:
            estimator_error = np.mean(
                np.average(missclassified, weights=sample_weight, axis=0)
            )

        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        if estimator_error >= 0.5:
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."
                )
            return None, None, None

        estimator_weight = (
            self.learning_rate
            * 0.5
            * np.log((1.0 - estimator_error) / max(estimator_error, 1e-10))
        )

        sample_weight *= np.exp(
            estimator_weight
            * missclassified
            * ((sample_weight > 0) | (estimator_weight < 0))
        )

        return sample_weight, estimator_weight, estimator_error

    def decision_function(self, X):
        check_is_fitted(self)
        X_transformed = self.scaler_.transform(X)
        return super().decision_function(X_transformed)
