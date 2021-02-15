import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from yarl import URL

from loadmydata.config import CONFIG
from loadmydata.utils import get_local_data_path

DATASET_NAME = "NYCTaxi"
DATAFILE_NAME = "nyc_taxi.csv"

DESCRIPTION = (
    "This data set contains the number of New York taxi passengers "
    "aggregated in 30 minutes buckets for the period between July 2014 and "
    "January 2015. "
    "There are five anomalies occur during the NYC marathon, Thanksgiving, "
    "Christmas, New Years day, and a snow storm.\n\n"
    "The raw data is from the NYC Taxi and Limousine Commission [1] and has "
    "been curated by [2].\n\n"
    "[1]: https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page \n"
    "[2]: Ahmad, S., Lavin, A., Purdy, S., & Agha, Z. (2017). Unsupervised "
    "real-time anomaly detection for streaming data. Neurocomputing."
)
# Anomaly labels are taken from the followink link:
# https://github.com/numenta/NAB/blob/master/labels/combined_labels.json
ANOMALY_LABELS = [
    "2014-11-01 19:00:00",
    "2014-11-27 15:30:00",
    "2014-12-25 15:00:00",
    "2015-01-01 01:00:00",
    "2015-01-27 00:00:00",
]


def read_timestamps_str(timestamp_str: str) -> dt.datetime:
    return dt.datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")


def get_nyc_taxi_download_link() -> URL:
    """Return the download link to the UEA/UCR repository.

    The download link is read from `config.ini`.
    """
    return CONFIG["nyc_taxi_download_link"] / DATAFILE_NAME


def load_nyc_taxi_dataset() -> (pd.DataFrame, np.ndarray, str):
    """Load (X, y) from a .csv file.

    The shape of X is (n_samples, 2): [timestamp, value]. The shape of y
    is (n_anomalies,).
    """
    local_cache_data = get_local_data_path(DATASET_NAME)
    local_archive_path = local_cache_data / DATAFILE_NAME
    if not local_cache_data.exists():
        local_cache_data.mkdir(exist_ok=True, parents=True)
        # get archive's url
        remote_archive_path = get_nyc_taxi_download_link()
        response = requests.get(remote_archive_path, stream=True)
        with open(local_archive_path, "w") as handle:
            print(response.text, file=handle)

    # load from downloaded (or cached) files
    X = pd.read_csv(local_archive_path, parse_dates=["timestamp"]).rename(
        {"value": "taxi_count"}, axis=1
    )
    y = np.array(
        [
            read_timestamps_str(timestamp_str)
            for timestamp_str in ANOMALY_LABELS
        ]
    )
    return X, y, DESCRIPTION
