from configparser import ConfigParser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE_LOCATION = BASE_DIR / Path("config.ini")

CONFIG = ConfigParser()
CONFIG.read(CONFIG_FILE_LOCATION)

