"""RPS-GMM: time series classification of supraglacial lake evolution.

Reference implementation for:

    E. Hossain, M. O. Gani, D. Dunmire, A. C. Subramanian and H. Younas,
    "Time Series Classification of Supraglacial Lakes Evolution over Greenland
    Ice Sheet", 2024 International Conference on Machine Learning and
    Applications (ICMLA), 2024, pp. 490-497. doi:10.1109/ICMLA61862.2024.00072
"""

from .data import load_backscatter, load_backscatter_water, load_split
from .model import CLASSES, GridSearchResult, RPSGMMClassifier
from .rps import embed, phase_space_length

__version__ = "1.0.0"

__all__ = [
    "CLASSES",
    "GridSearchResult",
    "RPSGMMClassifier",
    "embed",
    "phase_space_length",
    "load_backscatter",
    "load_backscatter_water",
    "load_split",
]
