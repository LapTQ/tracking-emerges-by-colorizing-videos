from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL

# ==================================================================================================

LOGGER = GLOBAL.LOGGER
import numpy as np
import torch


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_device(device=None):
    if device is None or device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        LOGGER.info('Device was not specified, while cuda is {}. Using {}.'.format(
            'available' if torch.cuda.is_available() else 'not available',
            device
        ))
    else:
        device = torch.device(device)
        LOGGER.info('Using {}.'.format(device))
    return device
        