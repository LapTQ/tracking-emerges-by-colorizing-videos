from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

import src as GLOBAL

LOGGER = GLOBAL.LOGGER
LOGGER.info('Hello, config!\n{}'.format(GLOBAL.CONFIG))