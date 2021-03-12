import json
import os
import shutil
from pathlib import Path
import tarfile

import numpy as np
import pandas as pd
import requests
from sklearn.utils import Bunch
from tqdm import tqdm
from yarl import URL

from loadmydata.config import CONFIG, HUMAN_LOCOMOTION_CODE_LIST
from loadmydata.utils import (
    get_local_data_path,
    is_directory_empty,
    get_cache_home,
)

DATASET_NAME = "MoleneMeteo"
DATAFILE_NAME = "RADOMEH.tar.gz"
README_DOWNLOAD_LINK = URL(
    "https://www.data.gouv.fr/fr/datasets/r/80fb22dc-e155-4d5d-a02e-d263fa789fda"
)
README_FILENAME = "readme_radomeh.csv"

DESCRIPTION = """The French national meteorological service made publicly available [1] a data set of hourly observations from a number of weather ground stations. Those stations are located in Brittany, France, and the data were collected during the month of January 2014. The stations recorded several meteorological variables, such as temperature, humidity, wind speed and direction, etc. Missing data (denoted by 'mq' in the original data) are replaced by NaNs.

In addition, the exact positions of the ground stations are provided.

Here is an excerpt of the README file that comes with the data.

    Descriptif  Mnémonique  type    unité
    Paramètres standard
    Indicatif INSEE station numer_sta   car
    Indicatif OMM station   id_omm  int
    Date    date    car
    Point de rosée  td  réel    K
    Température t    réel   K
    Température maximale de l'air   tx  réel    K
    Température minimale de l'air   tn  réel    K
    Humidité    u   int %
    Humidité maximale   ux  int %
    Humidité minimale   un  int %
    Direction du vent moyen 10 mn   dd    int   degré
    Vitesse du vent moyen 10 mn ff   réel   m/s
    Direction du vent moyen maximal dxy   int   degré
    Vitesse maximale du vent tmoyen fxy  réel   m/s
    Direction du vent instantané maximal    dxi   int   degré
    Vitesse maximale du vent instantané fxi  réel   m/s
    Précipitations dans  l'heure    rr1 réel    kg/m²
    Paramètres selon instrumentation spécifique
    Température à -10 cm    t_10    réel    K
    Température à -20 cm    t_20    réel    K
    Température à -50 cm    t_50    réel    K
    Température à -100 cm   t_100       K
    Visibilité horizontale  vv  réel    m
    Etat du sol etat_sol    int code
    Hauteur totale de la couche de neige    sss réel    m
    Nebulosité totale   n   réel    %
    Durée insolation    insolh  int mn
    Rayonnement global  ray_glo01   réel    J/m²
    Pression station    pres    int Pa
    Pression au niveau mer  pmer    int Pa

[1] https://www.data.gouv.fr/fr/datasets/projections-climatiques-sur-la-zone-large-molene-sur-un-mois/
"""


def get_molene_meteo_download_link() -> URL:
    """Return the download link to the Molene meteo data set."""
    return CONFIG["molene_meteo_download_link"]


def download_from_remote_molene_meteo() -> None:
    """Download and uncompress the human locomotion data set."""
    local_cache_data = get_local_data_path(DATASET_NAME)
    local_archive_path = local_cache_data / DATAFILE_NAME

    if not local_cache_data.exists():
        local_cache_data.mkdir(exist_ok=True, parents=True)

        # get archive's url
        remote_archive_path = get_molene_meteo_download_link()
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
        # if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        #     print(f"ERROR: the download of {DATAFILE_NAME} went wrong.")
        # TODO: Add an error management procedure.
        # uncompress the data in the data folder
        with tarfile.open(local_archive_path) as tar:
            tar.extractall(local_cache_data)
        # remove archive file
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

        # Download README file
        response = requests.get(README_DOWNLOAD_LINK, stream=True)
        # handle the download progress bar
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        # actual download
        with open(local_cache_data / README_FILENAME, "wb") as handle:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                handle.write(data)
        progress_bar.close()


def load_molene_meteo_dataset() -> (pd.DataFrame, pd.DataFrame, str):
    """Load the Molene meteo data set.

    Returns:
        (pd.DataFrame, pd.DataFrame, str): the collected data, the weather
            stations' positions, and the description string.
    """
    # check if in cache, othewise download data
    download_from_remote_molene_meteo()

    # read the station information
    local_cache_data = get_local_data_path(DATASET_NAME)
    stations_df = pd.read_csv(
        local_cache_data / README_FILENAME,
        skiprows=43,
        sep=";",
        encoding="latin1",
    )

    # read the sensors' data
    list_of_df = list()
    for fname in local_cache_data.iterdir():
        if fname.suffix == ".txt":
            list_of_df.append(
                pd.read_csv(
                    fname,
                    converters={
                        "date": pd.to_datetime,
                        "date_insert": pd.to_datetime,
                        "numer_sta": pd.to_numeric,
                    },
                    skipfooter=1,
                    engine="python",
                    na_values="mq",
                )
            )
    data_df = pd.concat(list_of_df).drop("Unnamed: 29", axis=1)

    # add the station name in the data
    station_name_converter_dict = dict(
        stations_df[["Numéro", "Nom"]].to_numpy().tolist()
    )
    data_df["station_name"] = data_df.numer_sta.replace(
        station_name_converter_dict
    )

    return data_df, stations_df, DESCRIPTION
