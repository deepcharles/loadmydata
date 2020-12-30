from pathlib import Path

from sklearn.utils import Bunch
from sktime.utils.data_io import load_from_tsfile_to_dataframe

from loadmydata.utils import (download_from_remote_uea_ucr,
                              get_local_data_path, get_uea_ucr_download_link)


def load_uea_ucr_data(name: str) -> Bunch:
    """Return data for the given data set.

    The data are contained in a `Bunch` instance which is like a dictionary
    whose keys are accessible as attributes.
    (See [here](https://scikit-learn.org/stable/modules/generated/sklearn.utils.Bunch.html)
    for more information.)

    Args:
        name (str): data set's name, e.g. `ArrowHead` (case-sensitive).

    Returns:
        [sklearn.utils.Bunch]: X_train, X_test, y_train, y_test, url and
            description of the data set.
    """

    # download data
    download_from_remote_uea_ucr(name)
    # get data path
    data_path = get_local_data_path(name)
    data_path_train = data_path / Path(f"{name}_TRAIN.ts")
    data_path_test = data_path / Path(f"{name}_TEST.ts")
    data_path_description = data_path / Path(f"{name}.txt")

    # load from downloaded (or cached) files
    X_train, y_train = load_from_tsfile_to_dataframe(data_path_train)
    X_test, y_test = load_from_tsfile_to_dataframe(data_path_test)
    with (open(data_path_description, encoding="ISO-8859-1")) as f:
        description = f.read()

    return Bunch(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        description=description,
        url=(get_uea_ucr_download_link() / (name + ".zip")),
        location=str(data_path.absolute().resolve()),
    )
