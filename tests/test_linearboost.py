from sklearn.utils.estimator_checks import parametrize_with_checks
from linearboost.linear_boost import LinearBoostClassifier


@parametrize_with_checks([LinearBoostClassifier()])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)
