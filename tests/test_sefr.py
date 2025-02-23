from sklearn.base import is_classifier

from linearboost.sefr import SEFR

from ._utils import check_estimator, get_expected_failed_tests


def test_sefr_estimator():
    """
    Test whether `SEFR` classifier adheres to scikit-learn conventions.
    """
    check_estimator(SEFR(), expected_failed_checks=get_expected_failed_tests(SEFR()))
    assert is_classifier(SEFR)
