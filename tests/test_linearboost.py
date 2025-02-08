from sklearn.base import is_classifier
from sklearn.utils.estimator_checks import check_estimator

from src.linearboost.linear_boost import LinearBoostClassifier


def test_linear_boost_estimator():
    """
    Test whether `LinearBoostClassifier` adheres to scikit-learn conventions.
    """
    check_estimator(LinearBoostClassifier())
    assert is_classifier(LinearBoostClassifier)
