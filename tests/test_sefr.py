from unittest.mock import patch

import numpy as np
import pytest
from sklearn.base import is_classifier
from sklearn.datasets import load_breast_cancer, make_classification
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.utils._testing import (
    assert_allclose,
    assert_array_almost_equal,
    assert_array_equal,
)

from linearboost.sefr import SEFR

from ._utils import check_estimator, get_expected_failed_tests


def test_sefr_estimator():
    """Test whether SEFR classifier adheres to scikit-learn conventions."""
    check_estimator(SEFR(), expected_failed_checks=get_expected_failed_tests(SEFR()))
    assert is_classifier(SEFR)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_init_parameters(fit_intercept):
    """Test SEFR initialization with different parameters."""
    sefr = SEFR(fit_intercept=fit_intercept)
    assert sefr.fit_intercept == fit_intercept


def test_init_default_parameters():
    """Test SEFR initialization with default parameters."""
    sefr = SEFR()
    assert sefr.fit_intercept is True


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_fit_simple_data(fit_intercept):
    """Test fitting on simple 2D data."""
    X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    y = np.array([0, 0, 1, 1])

    sefr = SEFR(fit_intercept=fit_intercept)
    sefr.fit(X, y)

    # Check fitted attributes
    assert hasattr(sefr, "coef_")
    assert hasattr(sefr, "intercept_")
    assert hasattr(sefr, "classes_")
    assert hasattr(sefr, "n_features_in_")

    # Check shapes
    assert sefr.coef_.shape == (1, X.shape[1])
    assert sefr.intercept_.shape == (1,)
    assert sefr.classes_.shape == (2,)
    assert sefr.n_features_in_ == X.shape[1]

    # Check intercept behavior
    if not fit_intercept:
        assert sefr.intercept_[0] == 0.0


def test_fit_with_sample_weight():
    """Test fitting with sample weights."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        n_informative=4,
        random_state=42,
        n_clusters_per_class=1,
    )
    sample_weight = np.random.RandomState(42).rand(len(X))

    sefr = SEFR()
    sefr.fit(X, y, sample_weight=sample_weight)

    assert hasattr(sefr, "coef_")
    assert hasattr(sefr, "intercept_")
    assert sefr.coef_.shape == (1, X.shape[1])


@pytest.mark.parametrize("n_samples,n_features", [(50, 3), (100, 5), (200, 10)])
def test_fit_various_sizes(n_samples, n_features):
    """Test fitting on datasets of various sizes."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr = SEFR()
    sefr.fit(X, y)

    assert sefr.coef_.shape == (1, n_features)
    assert sefr.n_features_in_ == n_features


def test_predict_binary_classification():
    """Test prediction for binary classification."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sefr = SEFR()
    sefr.fit(X_train, y_train)
    predictions = sefr.predict(X_test)

    assert predictions.shape == (X_test.shape[0],)
    assert set(predictions).issubset(set(y_train))


def test_predict_proba():
    """Test probability prediction."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr = SEFR()
    sefr.fit(X, y)
    probas = sefr.predict_proba(X)

    # Check shape
    assert probas.shape == (X.shape[0], 2)

    # Check probabilities are valid
    assert np.all(probas >= 0)
    assert np.all(probas <= 1)
    assert_allclose(probas.sum(axis=1), 1.0, rtol=1e-7)


def test_predict_log_proba():
    """Test log probability prediction."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr = SEFR()
    sefr.fit(X, y)
    log_probas = sefr.predict_log_proba(X)

    # Check shape
    assert log_probas.shape == (X.shape[0], 2)

    # Check all values are <= 0 (log probabilities)
    assert np.all(log_probas <= 0)

    # Compare with predict_proba
    probas = sefr.predict_proba(X)
    expected_log_probas = np.log(probas)
    assert_allclose(log_probas, expected_log_probas, rtol=1e-7)


def test_decision_function():
    """Test decision function."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr = SEFR()
    sefr.fit(X, y)
    decision = sefr.decision_function(X)

    assert decision.shape == (X.shape[0],)

    # Check consistency with predict
    predictions = sefr.predict(X)
    expected_predictions = (decision > 0).astype(int)

    # Map to class labels
    predicted_classes = sefr.classes_[expected_predictions]
    assert_array_equal(predictions, predicted_classes)


def test_score_method():
    """Test the score method."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sefr = SEFR()
    sefr.fit(X_train, y_train)
    score = sefr.score(X_test, y_test)

    assert isinstance(score, float)
    assert 0 <= score <= 1


# Error handling tests
def test_predict_before_fit_raises_error():
    """Test that predicting before fitting raises NotFittedError."""
    X = np.array([[1, 2], [3, 4]])
    sefr = SEFR()

    with pytest.raises(NotFittedError):
        sefr.predict(X)


def test_predict_proba_before_fit_raises_error():
    """Test that predict_proba before fitting raises NotFittedError."""
    X = np.array([[1, 2], [3, 4]])
    sefr = SEFR()

    with pytest.raises(NotFittedError):
        sefr.predict_proba(X)


def test_decision_function_before_fit_raises_error():
    """Test that decision_function before fitting raises NotFittedError."""
    X = np.array([[1, 2], [3, 4]])
    sefr = SEFR()

    with pytest.raises(NotFittedError):
        sefr.decision_function(X)


def test_single_class_error():
    """Test that fitting with single class raises ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 0, 0])  # All same class

    sefr = SEFR()
    with pytest.raises(
        ValueError, match="Classifier can't train when only one class is present"
    ):
        sefr.fit(X, y)


def test_multiclass_error():
    """Test that multiclass data raises ValueError."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])  # Three classes

    sefr = SEFR()
    with pytest.raises(
        ValueError,
        match=r"Only binary classification is supported|Unknown label type: non-binary",
    ):
        sefr.fit(X, y)


def test_multiclass_error_sklearn_pre_16():
    """Test multiclass error with sklearn < 1.6 message."""
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([0, 1, 2])  # Three classes

    sefr = SEFR()

    # Mock SKLEARN_V1_6_OR_LATER to be False to test the older sklearn message
    with patch("linearboost.sefr.SKLEARN_V1_6_OR_LATER", False):
        with pytest.raises(ValueError, match="Unknown label type: non-binary"):
            sefr.fit(X, y)


def test_invalid_sample_weight_shape():
    """Test that invalid sample weight shape raises ValueError."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    sample_weight = np.array([0.5])  # Wrong length

    sefr = SEFR()
    with pytest.raises((ValueError, IndexError)):
        sefr.fit(X, y, sample_weight=sample_weight)


def test_zero_sample_weights_error():
    """Test that zero sample weights for all samples of a class raises error."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 0, 1, 1])
    sample_weight = np.array(
        [0.0, 0.0, 1.0, 1.0]
    )  # All class 0 samples have zero weight

    sefr = SEFR()
    with pytest.raises(ValueError, match="SEFR requires 2 classes"):
        sefr.fit(X, y, sample_weight=sample_weight)


def test_inconsistent_feature_numbers():
    """Test that inconsistent feature numbers raise ValueError."""
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([0, 1])
    X_test = np.array([[1, 2, 3]])  # Different number of features

    sefr = SEFR()
    sefr.fit(X_train, y_train)

    with pytest.raises(ValueError):
        sefr.predict(X_test)


def test_empty_data_error():
    """Test that empty data raises appropriate error."""
    X = np.array([]).reshape(0, 2)
    y = np.array([])

    sefr = SEFR()
    with pytest.raises(ValueError):
        sefr.fit(X, y)


# Mathematical properties tests
def test_linear_separable_perfect_classification():
    """Test that SEFR works well on linearly separable data."""
    # Create perfectly linearly separable data
    np.random.seed(42)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2], [2, 3], [3, 2], [3, 3]])
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    sefr = SEFR()
    sefr.fit(X, y)
    predictions = sefr.predict(X)

    # Should achieve perfect or near-perfect accuracy on this simple dataset
    accuracy = (predictions == y).mean()
    assert accuracy >= 0.75  # At least 75% accuracy


def test_reproducibility():
    """Test that results are reproducible with same data."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr1 = SEFR()
    sefr1.fit(X, y)
    pred1 = sefr1.predict(X)

    sefr2 = SEFR()
    sefr2.fit(X, y)
    pred2 = sefr2.predict(X)

    assert_array_equal(pred1, pred2)
    assert_array_almost_equal(sefr1.coef_, sefr2.coef_)


@pytest.mark.parametrize("fit_intercept", [True, False])
def test_coefficient_properties(fit_intercept):
    """Test mathematical properties of fitted coefficients."""
    X, y = make_classification(
        n_samples=100,
        n_features=4,
        n_redundant=0,
        random_state=42,
        n_clusters_per_class=1,
    )

    sefr = SEFR(fit_intercept=fit_intercept)
    sefr.fit(X, y)

    # Coefficients should be finite
    assert np.all(np.isfinite(sefr.coef_))

    # Intercept behavior
    if fit_intercept:
        assert np.isfinite(sefr.intercept_[0])
    else:
        assert sefr.intercept_[0] == 0.0


def test_feature_names_in():
    """Test feature_names_in_ attribute when using pandas DataFrame."""
    try:
        import pandas as pd

        # Create DataFrame with named features
        X, y = make_classification(
            n_samples=50,
            n_features=3,
            n_redundant=0,
            random_state=42,
            n_clusters_per_class=1,
        )
        feature_names = ["feature1", "feature2", "feature3"]
        X_df = pd.DataFrame(X, columns=feature_names)

        sefr = SEFR()
        sefr.fit(X_df, y)

        assert hasattr(sefr, "feature_names_in_")
        assert_array_equal(sefr.feature_names_in_, feature_names)

    except ImportError:
        pytest.skip("pandas not available")


def test_classes_attribute():
    """Test the classes_ attribute after fitting."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])

    sefr = SEFR()
    sefr.fit(X, y)

    assert hasattr(sefr, "classes_")
    assert_array_equal(sefr.classes_, [0, 1])


def test_classes_attribute_string_labels():
    """Test the classes_ attribute with string labels."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array(["cat", "dog", "cat", "dog"])

    sefr = SEFR()
    sefr.fit(X, y)

    assert hasattr(sefr, "classes_")
    assert_array_equal(sefr.classes_, ["cat", "dog"])


def test_breast_cancer_dataset():
    """Test on real-world breast cancer dataset."""
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    sefr = SEFR()
    sefr.fit(X_train, y_train)

    # Check basic functionality
    predictions = sefr.predict(X_test)
    probabilities = sefr.predict_proba(X_test)
    score = sefr.score(X_test, y_test)

    assert len(predictions) == len(X_test)
    assert probabilities.shape == (len(X_test), 2)
    assert 0 <= score <= 1
    assert score > 0.5  # Should be better than random guessing


def test_different_class_labels():
    """Test with different types of class labels."""
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])

    # Test with different label encodings
    label_sets = [
        [0, 1, 0, 1],  # Standard 0/1
        [1, 2, 1, 2],  # 1/2 encoding
        [-1, 1, -1, 1],  # -1/1 encoding
        ["A", "B", "A", "B"],  # String labels
    ]

    for y in label_sets:
        y = np.array(y)
        sefr = SEFR()
        sefr.fit(X, y)

        predictions = sefr.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset(set(y))


def test_numerical_stability():
    """Test numerical stability with extreme values."""
    # Test with very small values
    X_small = np.array([[1e-10, 2e-10], [3e-10, 4e-10], [5e-10, 6e-10], [7e-10, 8e-10]])
    y = np.array([0, 0, 1, 1])

    sefr = SEFR()
    sefr.fit(X_small, y)
    predictions = sefr.predict(X_small)

    assert len(predictions) == len(X_small)
    assert np.all(np.isfinite(sefr.coef_))

    # Test with large values
    X_large = np.array([[1e6, 2e6], [3e6, 4e6], [5e6, 6e6], [7e6, 8e6]])

    sefr_large = SEFR()
    sefr_large.fit(X_large, y)
    predictions_large = sefr_large.predict(X_large)

    assert len(predictions_large) == len(X_large)
    assert np.all(np.isfinite(sefr_large.coef_))


def test_check_X_method():
    """Test the _check_X method directly."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    sefr = SEFR()
    sefr.fit(X, y)

    # Test successful validation
    X_valid = np.array([[5, 6], [7, 8]])
    result = sefr._check_X(X_valid)
    assert result.shape == X_valid.shape
    assert np.array_equal(result, X_valid)


def test_check_X_wrong_features():
    """Test _check_X method with wrong number of features."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    sefr = SEFR()
    sefr.fit(X, y)

    # Mock validate_data to return X but skip sklearn's feature check, so we can test our custom feature count check
    X_wrong = np.array([[1, 2, 3]])  # 3 features instead of 2
    with patch("linearboost.sefr.validate_data", return_value=X_wrong):
        with pytest.raises(
            ValueError, match="Expected input with 2 features, got 3 instead"
        ):
            sefr._check_X(X_wrong)
