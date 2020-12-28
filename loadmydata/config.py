from configparser import ConfigParser
from pathlib import Path

CONFIG_FILE_LOCATION = Path(__file__).parents[1] / Path("config.ini")

CONFIG = ConfigParser()
CONFIG.read(CONFIG_FILE_LOCATION)
