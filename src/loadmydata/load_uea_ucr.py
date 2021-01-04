from pathlib import Path
import pandas as pd

from container import DataSet
from scipy.io.arff import loadarff
from loadmydata.utils import (download_from_remote_uea_ucr,
                              get_local_data_path, get_uea_ucr_download_link)


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

    # load from downloaded (or cached) files
    data, meta = loadarff(data_path_train)
    X_train = pd.DataFrame(data[column for column in meta.names() if column!="target"])
    y_train = data["target"].view().astype(str)

    data, meta = loadarff(data_path_test)
    X_test = pd.DataFrame(data[column for column in meta.names() if column!="target"])
    y_test = data["target"].view().astype(str)
    
    with (open(data_path_description, encoding="ISO-8859-1")) as f:
        description = f.read()

    return DataSet(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        description=description,
        url=(get_uea_ucr_download_link() / (name + ".zip")),
        location=str(data_path.absolute().resolve()),
    )
