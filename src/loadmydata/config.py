from pathlib import Path

from yarl import URL

HERE = Path(__file__).parent.absolute()

CONFIG = {
    "cache_home": HERE / "datasets",
    "uea_ucr_download_link": URL(
        "http://www.timeseriesclassification.com/Downloads/"
    ),
    "nyc_taxi_download_link": URL(
        "https://raw.githubusercontent.com/numenta/NAB/master/data/realKnownCause"
    ),
}
