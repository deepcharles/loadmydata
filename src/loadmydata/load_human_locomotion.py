import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import requests
from sklearn.utils import Bunch
from tqdm import tqdm
from yarl import URL

from loadmydata.config import CONFIG, HUMAN_LOCOMOTION_CODE_LIST
from loadmydata.utils import get_local_data_path, is_directory_empty

DATASET_NAME = "HumanLocomotion"
DATAFILE_NAME = "GaitData.zip"

DESCRIPTION = """This data set consists of 1020 multivariate gait signals collected with two inertial measurement units, from 230 subjects undergoing a fixed protocol:
        - standing still,
        - walking 10 m,
        - turning around,
        - walking back,
        - stopping.

In total, there are 8.5 h of gait time series. The measured population was composed of healthy subjects as well as patients with neurological or orthopedic disorders.
The start and end time stamps of more than 40,000 footsteps are available, as well as a number of contextual information about each trial. This exact data set was used in [1] to design and evaluate a step detection procedure.

The data are thoroughly described in [2].

[1] Oudre, L., Barrois-Müller, R., Moreau, T., Truong, C., Vienne-Jumeau, A., Ricard, D., Vayatis, N., & Vidal, P.-P. (2018). Template-based step detection with inertial measurement units. Sensors, 18(11).

[2] Truong, C., Barrois-Müller, R., Moreau, T., Provost, C., Vienne-Jumeau, A., Moreau, A., Vidal, P.-P., Vayatis, N., Buffat, S., Yelnik, A., Ricard, D., & Oudre, L. (2019). A data set for the study of human locomotion with inertial measurements units. Image Processing On Line (IPOL), 9.
"""


def get_code_list():
    return HUMAN_LOCOMOTION_CODE_LIST


def get_trial_filename(code: str) -> Path:
    """Returns the filename of the signal file and the metadata file.

    Code must be "{subject number}-{trial number}", e.g. "14-3".

    Args:
        code (str): code of the trial ("Patient-Trial").

    Returns:
        Path: path to the associated files.
    """
    local_cache_data = get_local_data_path(DATASET_NAME)
    filename = local_cache_data / code
    assert filename.with_suffix(
        ".csv"
    ).exists(), f"The code {code} cannot be found in the data set."
    return filename


def get_human_locomotion_download_link() -> URL:
    """Return the download link to the UEA/UCR repository.

    The download link is read from `config.ini`.
    """
    return CONFIG["human_locomotion_download_link"] / DATAFILE_NAME


def download_from_remote_human_locomotion() -> None:
    """Download and uncompress the human locomotion data set."""
    local_cache_data = get_local_data_path(DATASET_NAME)
    local_archive_path = local_cache_data / DATAFILE_NAME

    if not local_cache_data.exists():
        local_cache_data.mkdir(exist_ok=True, parents=True)

        # get archive's url
        remote_archive_path = get_human_locomotion_download_link()
        response = requests.get(remote_archive_path, stream=True)
        # handle the download progress bar
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        # actual download
        with open(local_archive_path, "wb") as handle:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                handle.write(data)
        progress_bar.close()
        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print(f"ERROR: the download of {DATAFILE_NAME} went wrong.")
        # uncompress the data in the data folder
        with ZipFile(local_archive_path, "r") as zf:
            zf.extractall(local_cache_data)
        # remove zip file
        os.remove(local_archive_path)
        # Check if the extracted directory contains a single sub-directory and
        # no other file.
        directory_list = [x for x in local_cache_data.iterdir() if x.is_dir()]
        non_directory_list = [
            x for x in local_cache_data.iterdir() if not x.is_dir()
        ]
        if len(directory_list) == 1 and len(non_directory_list) == 0:
            sub_dir = directory_list[0]
            for element in sub_dir.iterdir():
                shutil.move(str(element), str(sub_dir.parent))
            os.rmdir(str(sub_dir))


def load_trial(code: str) -> pd.DataFrame:
    """Returns the signal of the trial.

    Args:
        code (str): code of the trial ("Patient-Trial").

    Returns:
        pd.DataFrame: Signal of the the trial, shape (n_sample, n_dimension).
    """
    fname = get_trial_filename(code)
    df = pd.read_csv(fname.with_suffix(".csv"), sep=",")
    return df


def load_metadata(code):
    """Returns the metadata of the trial.
    Parameters
    ----------
    code : str
        Code of the trial ("Patient-Trial").
    Returns
    -------
    dict
        Metadata dictionary.
    """
    fname = get_trial_filename(code)
    with open(fname.with_suffix(".json"), "r") as f:
        metadata = json.load(f)
    return metadata


def load_human_locomotion_dataset(code: str) -> Bunch:
    """Load the human locomotion data set.

    Returns:
        sklearn.utils.Bunch: (dict-like) the acceleration and angular velocity,
            the step indexes, the metadata and the description
    """
    # check if in cache, othewise download data
    download_from_remote_human_locomotion()
    # get data
    signal = load_trial(code)
    metadata = load_metadata(code)
    left_steps = np.array(metadata.pop("LeftFootActivity"))
    right_steps = np.array(metadata.pop("RightFootActivity"))

    # load from the downloaded (or cached) files
    return Bunch(
        signal=signal,
        left_steps=left_steps,
        right_steps=right_steps,
        metadata=metadata,
        description=DESCRIPTION,
    )
