from pathlib import Path

import numpy as np
import numpy.ma as ma
import pandas as pd
from numpy.ma.core import MaskedArray
from scipy.io.arff import loadarff

from loadmydata.container import DataSet
from loadmydata.padding import pad_at_the_end
from loadmydata.utils import (
    download_from_remote_uea_ucr,
    get_local_data_path,
    get_uea_ucr_download_link,
)


def load_Xy_from_arff(data_path: Path) -> (MaskedArray, np.ndarray):
    """Load (X, y) from a .arff file.

    The shape of X is (n_series, n_samples, n_dims). The shape of y is
    (n_series,).
    """

    # load from downloaded (or cached) files
    data, meta = loadarff(data_path)
    names = meta.names()
    # are the data multivariate or univariate?
    is_multivariate = len(names) == 2
    is_univariate = not is_multivariate
    # load y, the target variable
    y = data["target"].view().astype(str)
    # load X, the attributes
    if is_univariate:
        keep_col = [col for col in names if col != "target"]
        X_raw = data[keep_col].view()
    elif is_multivariate:
        X_raw = data[names[0]].view()

    X = list()
    for signal in X_raw:
        if is_multivariate:
            X.append(np.array(signal.tolist()).T)
        elif is_univariate:
            X.append(np.array(signal.tolist()).T.reshape(-1, 1))

    max_size = max(signal.shape[0] for signal in X)

    X = ma.stack(
        [pad_at_the_end(signal, max_size - signal.shape[0]) for signal in X]
    )

    return X, y


def load_uea_ucr_data(name: str) -> DataSet:
    """Return data for the given data set.

    The data are contained in a `DataSet` instance.
    Time series are available in the `.X_train` and `.X_test` atttributes, which
    are pandas dataframe where each **row** contains a distinct time series.
    The labels of each time series are in the `.y_train` and `.y_test`
    atttributes.

    Args:
        name (str): data set's name, e.g. `ArrowHead` (case-sensitive).

    Returns:
        [loadmydata.container.DataSet]: X_train, X_test, y_train, y_test, url
            and description of the data set.
    """

    # download data
    download_from_remote_uea_ucr(name)
    # get data path
    data_path = get_local_data_path(name)
    data_path_train = data_path / Path(f"{name}_TRAIN.arff")
    data_path_test = data_path / Path(f"{name}_TEST.arff")
    data_path_description = data_path / Path(f"{name}.txt")
    # load X, y for train and test
    X_train, y_train = load_Xy_from_arff(data_path_train)
    X_test, y_test = load_Xy_from_arff(data_path_test)
    # load description
    with (open(data_path_description, encoding="ISO-8859-1")) as f:
        description = f.read()

    return DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        description=description,
        url=(get_uea_ucr_download_link() / (name + ".zip")),
        location=data_path.absolute().resolve(),
    )
