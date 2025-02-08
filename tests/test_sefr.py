from sklearn.base import is_classifier
from sklearn.utils.estimator_checks import check_estimator

from src.linearboost.sefr import SEFR


def test_sefr_estimator():
    """
    Test whether `SEFR` classifier adheres to scikit-learn conventions.
    """
    check_estimator(SEFR())
    assert is_classifier(SEFR)
