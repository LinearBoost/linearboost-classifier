"""SEFRBoost: gradient boosting with oblique splits from SEFR at each internal node."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import _check_sample_weight, check_is_fitted

from ._utils import SKLEARN_V1_6_OR_LATER, _fit_context, check_X_y, validate_data
from .sefr import SEFR

__all__ = ["SEFRBoostClassifier", "SEFRBoostRegressor"]


def _per_sample_class_weight(
    y_original: np.ndarray, classes: np.ndarray, class_weight
) -> np.ndarray:
    """Per-sample multipliers from sklearn-style class_weight (None / balanced / dict)."""
    if class_weight is None:
        return np.ones(y_original.shape[0], dtype=np.float64)
    cw = compute_class_weight(class_weight, classes=classes, y=y_original)
    w_by_class = {c: float(w) for c, w in zip(classes, cw)}
    return np.fromiter(
        (w_by_class[y] for y in y_original),
        dtype=np.float64,
        count=len(y_original),
    )


def _effective_fit_weights(
    y_idx: np.ndarray,
    y_original: np.ndarray,
    classes: np.ndarray,
    sample_weight: np.ndarray,
    class_weight,
    scale_pos_weight: Optional[float],
) -> np.ndarray:
    """Combine sample_weight, class_weight, and scale_pos_weight (positive class = classes_[1])."""
    sw = np.asarray(sample_weight, dtype=np.float64)
    cw = _per_sample_class_weight(y_original, classes, class_weight)
    ew = sw * cw
    spw = 1.0 if scale_pos_weight is None else float(scale_pos_weight)
    if spw != 1.0:
        ew = ew * np.where(y_idx == 1, spw, 1.0)
    return ew


def _newton_leaf_value(
    residuals: np.ndarray,
    p: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Weighted log-loss Newton step: sum(w*r) / sum(w * p(1-p)); w is effective weight."""
    h = p * (1.0 - p)
    h = np.maximum(h, 1e-10)
    num = np.sum(sample_weight * residuals)
    den = np.sum(sample_weight * h) + 1e-10
    return float(num / den)


def _mse_leaf_value(residuals: np.ndarray, sample_weight: np.ndarray) -> float:
    """Weighted mean residual (Newton / negative-gradient step for squared loss)."""
    num = np.sum(sample_weight * residuals)
    den = np.sum(sample_weight) + 1e-10
    return float(num / den)


def _sanitize_sefr_hyperplane(
    coef: np.ndarray, intercept: float
) -> tuple[np.ndarray, float]:
    """Stabilize linear split parameters from SEFR (avoid NaN/Inf and matmul overflow)."""
    c = np.asarray(coef, dtype=np.float64).ravel()
    c = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    # With OHE + scaled numerics, keep coefficients in a range that keeps Xm @ c finite.
    c = np.clip(c, -1e4, 1e4)
    b = float(np.nan_to_num(intercept, nan=0.0, posinf=0.0, neginf=0.0))
    b = float(np.clip(b, -1e6, 1e6))
    return c, b


def _affine_hyperplane_scores(
    X: np.ndarray, coef: np.ndarray, intercept: float
) -> np.ndarray:
    """``X @ coef + intercept`` with finite inputs and no spurious matmul warnings."""
    Xb = np.asarray(X, dtype=np.float64, order="C")
    Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.asarray(coef, dtype=np.float64).ravel()
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        s = Xb @ c + float(intercept)
    return np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class _SEFRTreeNode:
    is_leaf: bool
    value: float = 0.0
    coef: Optional[np.ndarray] = None
    intercept: float = 0.0
    left: Optional["_SEFRTreeNode"] = None
    right: Optional["_SEFRTreeNode"] = None


class _SEFRTree:
    """One regression tree: internal nodes use linear SEFR splits on pseudo-residuals."""

    def __init__(self, root: _SEFRTreeNode):
        self.root_ = root

    @staticmethod
    def _grow(
        X: np.ndarray,
        residuals: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray,
        idx: np.ndarray,
        depth: int,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        *,
        regression: bool,
    ) -> _SEFRTreeNode:
        n = idx.shape[0]
        r_n = residuals[idx]
        p_n = p[idx]
        w_n = sample_weight[idx]

        def leaf() -> _SEFRTreeNode:
            if regression:
                v = _mse_leaf_value(r_n, w_n)
            else:
                v = _newton_leaf_value(r_n, p_n, w_n)
            return _SEFRTreeNode(is_leaf=True, value=v)

        if depth >= max_depth or n < min_samples_split:
            return leaf()

        if np.all(r_n > 0) or np.all(r_n < 0):
            return leaf()

        y_bin = (r_n > 0).astype(int)
        if np.unique(y_bin).size < 2:
            return leaf()

        rw = np.abs(r_n) * w_n
        s = rw.sum()
        if s <= 1e-15:
            return leaf()
        rw = rw / s

        X_n = X[idx]
        sefr = SEFR(kernel="linear")
        try:
            sefr.fit(X_n, y_bin, sample_weight=rw)
        except ValueError:
            return leaf()

        coef = np.asarray(sefr.coef_, dtype=np.float64).ravel().copy()
        intercept = float(np.asarray(sefr.intercept_).ravel()[0])
        coef, intercept = _sanitize_sefr_hyperplane(coef, intercept)
        scores = _affine_hyperplane_scores(X_n, coef, intercept)
        left_mask = scores <= 0.0
        right_mask = ~left_mask
        n_left = int(np.sum(left_mask))
        n_right = int(np.sum(right_mask))

        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            return leaf()
        if n_left == 0 or n_right == 0:
            return leaf()

        idx_left = idx[left_mask]
        idx_right = idx[right_mask]

        node = _SEFRTreeNode(
            is_leaf=False,
            coef=coef,
            intercept=intercept,
        )
        node.left = _SEFRTree._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx_left,
            depth + 1,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            regression=regression,
        )
        node.right = _SEFRTree._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx_right,
            depth + 1,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            regression=regression,
        )
        return node

    @classmethod
    def fit(
        cls,
        X: np.ndarray,
        residuals: np.ndarray,
        p: np.ndarray,
        sample_weight: np.ndarray,
        max_depth: int,
        min_samples_leaf: int,
        min_samples_split: int,
        *,
        regression: bool = False,
    ) -> "_SEFRTree":
        n_samples = X.shape[0]
        idx = np.arange(n_samples, dtype=np.intp)
        root = cls._grow(
            X,
            residuals,
            p,
            sample_weight,
            idx,
            depth=0,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            regression=regression,
        )
        return cls(root)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Vectorized prediction: mask all samples through each node (batched matmul per node)."""
        check_is_fitted(self, "root_")
        n = X.shape[0]
        out = np.zeros(n, dtype=np.float64)
        stack: list[tuple[_SEFRTreeNode, np.ndarray]] = [
            (self.root_, np.ones(n, dtype=bool))
        ]
        while stack:
            node, mask = stack.pop()
            if not np.any(mask):
                continue
            if node.is_leaf:
                out[mask] = node.value
                continue
            Xm = X[mask]
            s = _affine_hyperplane_scores(Xm, node.coef, node.intercept)
            idx = np.nonzero(mask)[0]
            left_sub = s <= 0.0
            right_sub = ~left_sub
            m_left = np.zeros(n, dtype=bool)
            m_right = np.zeros(n, dtype=bool)
            m_left[idx[left_sub]] = True
            m_right[idx[right_sub]] = True
            stack.append((node.right, m_right))
            stack.append((node.left, m_left))
        return out


class SEFRBoostClassifier(ClassifierMixin, BaseEstimator):
    """Binary SEFRBoost: gradient boosting with SEFR oblique splits at tree nodes.

    Each boosting stage fits a shallow tree. At every internal node, a linear
    SEFR model is fit to pseudo-residuals (sign as class, magnitude as weight),
    matching :class:`LinearBoostClassifier` gradient boosting. Leaves output a
    Newton step for logistic loss.

    Boosting uses **effective weights** ``ew_i = sw_i * cw_i`` where ``sw`` is
    ``sample_weight`` (or ones), ``cw`` comes from ``class_weight`` (sklearn
    ``\"balanced\"`` or a dict), and ``scale_pos_weight`` optionally multiplies
    weights for samples in the positive class ``classes_[1]``. Trees are fit to
    the usual residual ``y - p``; leaf Newton steps use ``sum(ew * r) /
    sum(ew * p(1-p))``. SEFR node fitting uses sample weights ``|r| * ew``.
    The initial log-odds use the weighted positive rate ``sum(ew * y) / sum(ew)``.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations (trees).

    learning_rate : float, default=0.1
        Shrinkage applied to each tree's prediction.

    max_depth : int, default=3
        Maximum depth of each tree (root depth is 0).

    min_samples_leaf : int, default=10
        Minimum samples per child when splitting (by row count at node). Values
        like 20 are conservative for small training sets (e.g. n≈500); 5–10 is
        often better when folds leave few rows.

    min_samples_split : int, default=2
        Minimum samples required to attempt a split at a node.

    subsample : float, default=1.0
        Fraction of rows used to fit each tree (stochastic boosting).

    class_weight : dict, 'balanced', or None, default=None
        Multipliers per class (sklearn-style), combined with ``sample_weight``.

    scale_pos_weight : float or None, default=None
        If set, multiplies effective weights for the positive class
        ``classes_[1]`` (similar to XGBoost). ``None`` means ``1.0``.

    random_state : int, RandomState instance or None, default=None
        Random seed for subsampling.

    Notes
    -----
    Binary classification only. Use ``sklearn.preprocessing.StandardScaler`` in a
    ``Pipeline`` if features need scaling.

    Standalone :class:`~.sefr.SEFR` applies a heuristic scale to scores inside
    ``predict_proba``; this booster uses raw log-odds from ``decision_function``
    for Newton updates, which is the correct separation for gradient boosting.
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0.0, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "min_samples_split": [Interval(Integral, 2, None, closed="left")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "class_weight": [StrOptions({"balanced"}), dict, None],
        "scale_pos_weight": [Interval(Real, 0.0, None, closed="neither"), None],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        class_weight=None,
        scale_pos_weight=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.class_weight = class_weight
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            tags.classifier_tags.multi_class = False
            return tags

    def _more_tags(self) -> dict:
        return {
            "binary_only": True,
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weight_equivalence_on_dense_data": (
                    "Tree structure can change when zero-weight samples are included vs omitted."
                ),
            },
        }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if SKLEARN_V1_6_OR_LATER:
            X, y = validate_data(
                self,
                X,
                y=y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite=True,
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
                estimator=self,
            )
        self.n_features_in_ = X.shape[1]
        check_classification_targets(y)
        y_type = type_of_target(y)
        if y_type != "binary":
            raise ValueError(
                f"Only binary classification is supported; got target type {y_type!r}."
            )

        self.classes_, y_idx = np.unique(y, return_inverse=True)
        if self.classes_.size != 2:
            raise ValueError(
                "Binary classification requires exactly two classes in y; "
                f"got {self.classes_.size} class(es)."
            )

        y_binary = y_idx.astype(np.float64)
        n_samples = X.shape[0]
        y_original = np.asarray(y)

        if sample_weight is not None:
            sw = _check_sample_weight(sample_weight, X, dtype=np.float64)
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        ew = _effective_fit_weights(
            y_idx,
            y_original,
            self.classes_,
            sw,
            self.class_weight,
            self.scale_pos_weight,
        )
        w_sum = float(ew.sum()) + 1e-15
        pos_rate = float(np.clip(np.dot(ew, y_binary) / w_sum, 1e-10, 1.0 - 1e-10))
        self.init_score_ = np.log(pos_rate / (1.0 - pos_rate))
        F = np.full(n_samples, self.init_score_, dtype=np.float64)

        rng = check_random_state(self.random_state)
        self.trees_: list[_SEFRTree] = []

        for _ in range(self.n_estimators):
            p = 1.0 / (1.0 + np.exp(-F))
            p = np.clip(p, 1e-10, 1.0 - 1e-10)
            residuals = y_binary - p

            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * n_samples))
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
                X_b = X[sub_idx]
                r_b = residuals[sub_idx]
                p_b = p[sub_idx]
                w_b = ew[sub_idx]
            else:
                X_b, r_b, p_b, w_b = X, residuals, p, ew

            tree = _SEFRTree.fit(
                X_b,
                r_b,
                p_b,
                w_b,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                regression=False,
            )
            self.trees_.append(tree)
            F = F + self.learning_rate * tree.predict(X)

        self.F_train_ = F
        return self

    def decision_function(self, X):
        check_is_fitted(self, "trees_")
        if SKLEARN_V1_6_OR_LATER:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                reset=False,
                ensure_all_finite=True,
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
            )
        F = np.full(X.shape[0], self.init_score_, dtype=np.float64)
        for tree in self.trees_:
            F = F + self.learning_rate * tree.predict(X)
        # F is log-odds for classes_[1] vs classes_[0] (y encoded as class index).
        return F

    def predict_proba(self, X):
        df = self.decision_function(X)
        proba_pos = 1.0 / (1.0 + np.exp(-df))
        proba_pos = np.clip(proba_pos, 1e-10, 1.0 - 1e-10)
        return np.column_stack((1.0 - proba_pos, proba_pos))

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[(proba[:, 1] >= 0.5).astype(int)]


class SEFRBoostRegressor(RegressorMixin, BaseEstimator):
    """SEFRBoost regression: gradient boosting with SEFR oblique splits (squared error loss).

    Matches the classification booster structure: each stage fits a shallow tree
    whose internal nodes use a linear SEFR split on pseudo-residuals ``y - F``,
    with sample weights ``|r| * sw`` at nodes. Leaf values are weighted mean
    residuals (the Newton step for squared loss). The initial prediction is the
    weighted mean of ``y``.

    Parameters are the same as :class:`SEFRBoostClassifier` except
    ``class_weight`` and ``scale_pos_weight`` are not used.

    Notes
    -----
    Single-output regression only (``y`` one-dimensional after squeezing).
    """

    _parameter_constraints: dict = {
        "n_estimators": [Interval(Integral, 1, None, closed="left")],
        "learning_rate": [Interval(Real, 0.0, None, closed="left")],
        "max_depth": [Interval(Integral, 1, None, closed="left")],
        "min_samples_leaf": [Interval(Integral, 1, None, closed="left")],
        "min_samples_split": [Interval(Integral, 2, None, closed="left")],
        "subsample": [Interval(Real, 0.0, 1.0, closed="right")],
        "random_state": ["random_state"],
    }

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        min_samples_split: int = 2,
        subsample: float = 1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state

    if SKLEARN_V1_6_OR_LATER:

        def __sklearn_tags__(self):
            tags = super().__sklearn_tags__()
            tags.target_tags.required = True
            return tags

    def _more_tags(self) -> dict:
        return {
            "requires_y": True,
            "_xfail_checks": {
                "check_sample_weight_equivalence_on_dense_data": (
                    "Tree structure can change when zero-weight samples are included vs omitted."
                ),
                "check_regressors_train": (
                    "Default depth/leaf settings are conservative; R² on sklearn checker "
                    "data may not exceed 0.5 without stronger capacity."
                ),
            },
        }

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, sample_weight=None):
        if SKLEARN_V1_6_OR_LATER:
            X, y = validate_data(
                self,
                X,
                y=y,
                accept_sparse=False,
                dtype=np.float64,
                ensure_all_finite=True,
            )
        else:
            X, y = check_X_y(
                X,
                y,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
                estimator=self,
            )
        self.n_features_in_ = X.shape[1]
        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 0:
            y = np.array([float(y)], dtype=np.float64)
        elif y.ndim == 2:
            if y.shape[1] != 1:
                raise ValueError(
                    "SEFRBoostRegressor only supports single-target regression."
                )
            y = y.ravel()
        elif y.ndim != 1:
            raise ValueError(
                "SEFRBoostRegressor only supports single-target regression."
            )
        n_samples = X.shape[0]
        if y.shape[0] != n_samples:
            raise ValueError("X and y must have the same number of samples.")

        if sample_weight is not None:
            sw = _check_sample_weight(sample_weight, X, dtype=np.float64)
        else:
            sw = np.ones(n_samples, dtype=np.float64)

        w_sum = float(sw.sum()) + 1e-15
        self.init_score_ = float(np.dot(sw, y) / w_sum)
        F = np.full(n_samples, self.init_score_, dtype=np.float64)

        rng = check_random_state(self.random_state)
        self.trees_: list[_SEFRTree] = []

        for _ in range(self.n_estimators):
            residuals = y - F

            if self.subsample < 1.0:
                n_sub = max(1, int(self.subsample * n_samples))
                sub_idx = rng.choice(n_samples, size=n_sub, replace=False)
                X_b = X[sub_idx]
                r_b = residuals[sub_idx]
                w_b = sw[sub_idx]
            else:
                X_b, r_b, w_b = X, residuals, sw

            p_dummy = np.ones(X_b.shape[0], dtype=np.float64)
            tree = _SEFRTree.fit(
                X_b,
                r_b,
                p_dummy,
                w_b,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                regression=True,
            )
            self.trees_.append(tree)
            F = F + self.learning_rate * tree.predict(X)

        self.F_train_ = F
        return self

    def predict(self, X):
        check_is_fitted(self, "trees_")
        if SKLEARN_V1_6_OR_LATER:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                reset=False,
                ensure_all_finite=True,
            )
        else:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                dtype=np.float64,
                force_all_finite=True,
            )
        out = np.full(X.shape[0], self.init_score_, dtype=np.float64)
        for tree in self.trees_:
            out = out + self.learning_rate * tree.predict(X)
        return out
