from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL

from src.models import model_factory
from src.utils.dataset import setup_dataset_and_transform

# ==================================================================================================

from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
from copy import deepcopy
import pytest
import numpy as np


CONFIG_DATASET = GLOBAL.CONFIG['dataset']
CONFIG_TRANSFORM = GLOBAL.CONFIG['transform']


@pytest.fixture
def config_dataset_template():
    config_dataset = deepcopy(CONFIG_DATASET)

    config_dataset['module_name'] = 'fake'
    config_dataset['kwargs']['n_references'] = 3
    config_dataset['kwargs']['n_samples'] = 20
    config_dataset['kwargs']['batch_size'] = 32
    config_dataset['kwargs']['shuffle'] = True

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    return config_dataset


@pytest.fixture
def config_transform_template():
    config_transform = deepcopy(CONFIG_TRANSFORM)

    config_transform['input'] = [
        {
            'module_name': 'v2ToImage',
        },
        {
            'module_name': 'v2ToDtype',
            'kwargs': {
                'dtype': 'torch.float32',
                'scale': True
            }
        },
        {
            'module_name': 'v2Resize',
            'kwargs': {
                'size': (360, 640),
                'antialias': True
            }
        },
        {
            'module_name': 'v2Grayscale',
        },
    ]

    config_transform['label'] = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 64),
            }
        },
        {
            'module_name': 'cv2cvtColor',
            'kwargs': {
                'code': 'cv2.COLOR_BGR2LAB'
            }
        },
        {
            'module_name': 'ExtractChannel',
            'kwargs': {
                'channels': [1, 2]
            }
        },
        {
            'module_name': 'Quantize',
            'kwargs': {
                'model': {
                    'module_name': 'KMeans',
                    'kwargs': {
                        'n_clusters': 16,
                    },
                },
                'encoder': 'OneHotEncoder',
                'checkpoint_path': None
            }
        }
    ]

    # assume this order for convenience
    assert config_transform['input'][2]['module_name'] == 'v2Resize'
    assert config_transform['label'][0]['module_name'] == 'cv2Resize'
    assert config_transform['label'][2]['module_name'] == 'ExtractChannel'
    assert config_transform['label'][-1]['module_name'] == 'Quantize'

    return config_transform




