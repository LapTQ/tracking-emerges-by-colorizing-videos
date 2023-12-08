from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.datasets.utils import custom_collate_fn
from src.utils.dataset import setup_dataset, setup_dataset_and_transform
from src.utils.mics import set_seed

# ==================================================================================================

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import pytest
from copy import deepcopy
import cv2
import imgviz
import numpy as np


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


def test_dataset_output(
        config_dataset_template,
        config_transform_template
):
    config_dataset = deepcopy(config_dataset_template)
    config_transform = deepcopy(config_transform_template)

    # get value for convenience
    n_references = config_dataset['kwargs']['n_references']

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label'][:-1]   # exclude Quantize
    )
    dataset = _['dataset']

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    extracted_channels = config_transform['label'][2]['kwargs']['channels']
    extracted_channels = [extracted_channels] if isinstance(extracted_channels, int) else extracted_channels

    assert x.shape == (
        n_references + 1,
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert y.shape == (
        n_references + 1,
        target_label_size[0],
        target_label_size[1],
        len(extracted_channels)
    )


def test_custom_collate_fn(
        config_dataset_template,
        config_transform_template
):
    config_dataset = deepcopy(config_dataset_template)
    config_transform = deepcopy(config_transform_template)

    # get value for convenience
    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label'][:-1]   # exclude Quantize
    )
    dataset = _['dataset']

    batch = [dataset[i] for i in range(batch_size // (n_references + 1))]
    batch_X_collated, batch_Y_collated = custom_collate_fn(batch)

    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    extracted_channels = config_transform['label'][2]['kwargs']['channels']
    extracted_channels = [extracted_channels] if isinstance(extracted_channels, int) else extracted_channels
    
    # check shapes
    assert batch_X_collated.shape == (
        batch_size, 
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert batch_Y_collated.shape == (
        batch_size, 
        target_label_size[0], 
        target_label_size[1],
        len(extracted_channels)
    )


def test_dataloader_output(
        config_dataset_template,
        config_transform_template
):
    config_dataset = deepcopy(config_dataset_template)
    config_transform = deepcopy(config_transform_template)

    # get value for convenience
    batch_size = config_dataset['kwargs']['batch_size']

    # create dummy dataset to fit Quantize
    set_seed()
    _ = setup_dataset_and_transform(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    dataloader = _['dataloader']

    set_seed()
    batch_X, batch_Y = next(iter(dataloader))


    # check shapes
    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    encoder_name = config_transform['label'][-1]['kwargs']['encoder']
    n_clusters = config_transform['label'][-1]['kwargs']['model']['kwargs']['n_clusters']
    
    assert batch_X.shape == (
        batch_size, 
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert batch_Y.shape == (
        batch_size,  
        target_label_size[0], 
        target_label_size[1],
        1 if encoder_name == 'LabelEncoder' else n_clusters
    )
    

    # check values
    assert batch_X.dtype == torch.float32
    assert 0 <= batch_X.min() <= batch_X.max() <= 1
    assert batch_Y.dtype == (torch.float64 if encoder_name == 'OneHotEncoder' else torch.int64)
    assert (0 <= batch_Y.min() <= batch_Y.max() <= 1) if encoder_name == 'OneHotEncoder' else torch.all(batch_Y.max() < n_clusters)


    # visual check
    tile = []
    batch_X = (batch_X.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for input_, label in zip(batch_X, batch_Y):
        tile.append(cv2.cvtColor(input_, cv2.COLOR_GRAY2BGR))
        
        background = 255 + np.zeros((200, 200, 3), dtype='uint8')
        cv2.putText(background, 'shape={}'.format(tuple(label.shape)), (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(background, 'dtype={}'.format(label.dtype), (5, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(background, 'uniques={}'.format(np.unique(label)), (5, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        tile.append(background)
    tile = imgviz.tile(
        tile, 
        border=(255, 255, 255),
        border_width=5
    )
    
    window_title = 'Sample transformed data in a batch (left to right, top to bottom). True [t] or False [f]?'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, tile)
    key = cv2.waitKey(0)
    assert chr(key).lower().strip() == 't'


if __name__ == '__main__':
    pytest.main([__file__])
    



