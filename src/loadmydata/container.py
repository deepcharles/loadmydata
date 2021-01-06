from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.ma.core import MaskedArray
from yarl import URL


@dataclass
class DataSet:
    """Class for holding a data set (from the UEA/UCR repository)."""

    X_train: MaskedArray
    y_train: np.ndarray
    X_test: MaskedArray
    y_test: np.ndarray
    description: str
    url: URL
    location: Path
