from __future__ import annotations

from sklearn.utils.estimator_checks import check_estimator as sklearn_check_estimator

from linearboost._utils import SKLEARN_V1_6_OR_LATER

__all__ = ["check_estimator", "get_expected_failed_tests"]


if SKLEARN_V1_6_OR_LATER:
    check_estimator = sklearn_check_estimator
else:

    def check_estimator(estimator, *args, **kwargs):
        return sklearn_check_estimator(estimator)


def get_expected_failed_tests(estimator) -> dict[str, str]:
    return estimator._more_tags().get("_xfail_checks", {})
