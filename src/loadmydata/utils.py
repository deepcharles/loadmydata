import os
import shutil
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import requests
from tqdm import tqdm
from yarl import URL

from loadmydata.config import CONFIG


def get_cache_home() -> str:
    """Return the path of the cached data directory.

    The data dir is read from the `CONFIG` variable.
    """
    return CONFIG["cache_home"]


def get_local_data_path(name: str):
    """Return the path to the local data folder.

    Args:
        name (str): data set's name, e.g. `ArrowHead` (case-sensitive).
    """
    return get_cache_home() / name


def get_uea_ucr_download_link() -> URL:
    """Return the download link to the UEA/UCR repository.

    The download link is read from `config.ini`.
    """
    return CONFIG["uea_ucr_download_link"]


def download_from_remote_uea_ucr(name: str) -> None:
    """Download and uncompress data from UEA/UCR repository.

    Args:
        name (str): data set's name, e.g. `ArrowHead` (case-sensitive).
    """
    local_cache_data = get_local_data_path(name)

    if not local_cache_data.exists():
        local_cache_data.mkdir(exist_ok=True, parents=True)

        # get archive's url
        archive_name = name + ".zip"
        local_archive_path = local_cache_data / archive_name
        remote_archive_path = get_uea_ucr_download_link() / archive_name
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
            print(f"ERROR: the download of {archive_name} went wrong.")
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
