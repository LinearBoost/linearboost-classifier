from sklearn.base import is_classifier

from linearboost.linear_boost import LinearBoostClassifier

from ._utils import check_estimator, get_expected_failed_tests, _yield_all_checks


def get_expected_failed_tests(estimator):
    failed = []
    for check in _yield_all_checks(estimator):
        if check.__name__ == "check_sample_weights_invariance":
            failed.append(check)
    return failed


def test_linear_boost_estimator():
    """
    Test whether `LinearBoostClassifier` adheres to scikit-learn conventions.
    """
    check_estimator(
        LinearBoostClassifier(),
        expected_failed_checks=get_expected_failed_tests(LinearBoostClassifier()),
    )
    assert is_classifier(LinearBoostClassifier)
