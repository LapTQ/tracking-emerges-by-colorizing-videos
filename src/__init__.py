from .utils.config import load_config
from .utils.logger import load_logger
from pathlib import Path

HERE = Path(__file__).parent

CONFIG = load_config(
    config_path=str(HERE / 'config.yml')
)

LOGGER = load_logger(
    **CONFIG['logging']
)
     

    