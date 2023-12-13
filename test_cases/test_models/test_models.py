from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL

from src.models import model_factory
from src.utils.dataset import setup_dataset_and_transform
from src.utils.mics import set_seed

# ==================================================================================================

from torch import nn
from copy import deepcopy
import pytest
import numpy as np
from copy import deepcopy


CONFIG_DATASET = GLOBAL.CONFIG['dataset']
CONFIG_TRANSFORM = GLOBAL.CONFIG['transform']


@pytest.fixture
def config_dataset_template():
    config_dataset = deepcopy(CONFIG_DATASET['train'])

    config_dataset['module_name'] = 'fake'
    config_dataset['kwargs']['n_references'] = 3
    config_dataset['kwargs']['n_samples'] = 10
    config_dataset['kwargs']['batch_size'] = 32
    config_dataset['kwargs']['shuffle'] = True

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    return config_dataset


@pytest.fixture
def config_transform_template():
    config_transform = deepcopy(CONFIG_TRANSFORM['train'])

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
                'size': (256, 256),
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
                'size': (32, 32),
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


def test_model_shape(
        config_dataset_template,
        config_transform_template
):
    config_model = {
        'backbone': {
            'module_name': 'resnet18',
            'kwargs': {
                'mid_channels': [64, 256, 256, 256],
                'mid_strides': [1, 2, 1, 1]
            }
        },
        'head': {
            'module_name': 'convnet3d'
        }
    }

    config_dataset = deepcopy(config_dataset_template)
    config_transform = deepcopy(config_transform_template)

    batch_size = config_dataset['kwargs']['batch_size']
    target_label_size = config_transform['label'][0]['kwargs']['size']

    set_seed()
    _ = setup_dataset_and_transform(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    dataloader = _['dataloader']
    n_references = config_dataset['kwargs']['n_references']

    config_model['module_name'] = {
        'backbone': config_model['backbone']['module_name'],
        'head': config_model['head']['module_name']
    }
    config_model['kwargs'] = {
        'backbone': config_model['backbone']['kwargs'],
        'head': {
            'n_references': n_references,
            'in_channels': config_model['backbone']['kwargs']['mid_channels'][-1],
            'mid_channels': 256,
            'out_channels': 64,
            'dilations': [1, 2, 4, 8, 16]
        }
    }

    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {})
    )

    backbone = model.backbone
    head = model.head
    assert head.n_references == n_references

    X, Y = next(iter(dataloader))
    backbone_output = backbone(X)
    head_output = head(backbone_output)

    assert backbone_output.shape == (
        batch_size,
        256,
        target_label_size[0],
        target_label_size[1]
    )
    assert head_output.shape == (
        batch_size,
        config_model['kwargs']['head']['out_channels'],
        target_label_size[0],
        target_label_size[1]
    )


if __name__ == '__main__':
    pytest.main([__file__])

    






