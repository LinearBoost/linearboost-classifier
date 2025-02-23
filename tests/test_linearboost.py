from sklearn.base import is_classifier

from linearboost.linear_boost import LinearBoostClassifier

from ._utils import check_estimator, get_expected_failed_tests


def test_linear_boost_estimator():
    """
    Test whether `LinearBoostClassifier` adheres to scikit-learn conventions.
    """
    check_estimator(
        LinearBoostClassifier(),
        expected_failed_checks=get_expected_failed_tests(LinearBoostClassifier()),
    )
    assert is_classifier(LinearBoostClassifier)
