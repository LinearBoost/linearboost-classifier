import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks
from linearboost.linear_boost import LinearBoostClassifier

from ._utils import check_estimator, get_expected_failed_tests

@parametrize_with_checks([LinearBoostClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)