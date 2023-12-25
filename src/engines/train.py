
from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.utils.mics import get_device

# ==================================================================================================

LOGGER = GLOBAL.LOGGER
import torch



    

