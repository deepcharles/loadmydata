from pathlib import Path

from yarl import URL

LOADMYDATA_FOLDER_STR = ".loadmydata_datasets"
CACHE_HOME = Path.home() / LOADMYDATA_FOLDER_STR

CONFIG = {
    "cache_home": CACHE_HOME,
    "uea_ucr_download_link": URL(
        "http://www.timeseriesclassification.com/Downloads/"
    ),
    "nyc_taxi_download_link": URL(
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause"
    ),
}
