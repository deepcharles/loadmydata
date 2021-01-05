from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from yarl import URL


@dataclass
class DataSet:
    """Class for holding a data set (from the UEA/UCR repository)."""

    X_train: pd.DataFrame
    y_train: np.ndarray
    X_test: pd.DataFrame
    y_test: np.ndarray
    description: str
    url: URL
    location: Path
