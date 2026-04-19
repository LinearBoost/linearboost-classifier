__version__ = "0.2.0"

from .linear_boost import LinearBoostClassifier
from .sefr import SEFR
from .sefr_boost import SEFRBoostClassifier, SEFRBoostRegressor

__all__ = [
    "LinearBoostClassifier",
    "SEFR",
    "SEFRBoostClassifier",
    "SEFRBoostRegressor",
]
