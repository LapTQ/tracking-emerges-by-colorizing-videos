import logging
import numpy as np
import torch


logger = logging.getLogger(__name__)


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def get_device(device=None):
    if device is None or device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info('Device was not specified, while cuda is {}. Using {}.'.format(
            'available' if torch.cuda.is_available() else 'not available',
            device
        ))
    else:
        device = torch.device(device)
        logger.info('Using {}.'.format(device))
    return device
        