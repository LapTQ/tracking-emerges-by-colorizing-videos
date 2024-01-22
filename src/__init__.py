from .utils.config import load_config
from pathlib import Path
import datetime
import os


HERE = Path(__file__).parent


CONFIG = load_config(
    config_path=str(HERE / 'config.yml')
)


RUN_DIR = str(Path(CONFIG['run_dir']) / datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
assert not os.path.exists(RUN_DIR), 'Run directory already exists: {}'.format(RUN_DIR)
CONFIG['run_dir'] = RUN_DIR
CONFIG['logging']['directory'] = RUN_DIR