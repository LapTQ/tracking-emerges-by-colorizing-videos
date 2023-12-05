from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.dataset import dataset_factory
from src.dataset.utils import custom_collate_fn
from src.transform import transform_factory

# ==================================================================================================

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import pytest
from copy import deepcopy
import cv2
import imgviz
import numpy as np


DATASET_CONFIG = GLOBAL.CONFIG['dataset']
TRANSFORM_CONFIG = GLOBAL.CONFIG['transform']


@pytest.fixture
def train_transforms():
    config_transform = deepcopy(TRANSFORM_CONFIG['train'])

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
                'n_colors': 16
            }
        },
        {
            'module_name': 'LabelEncoder',
        }
    ]

    input_transform = transforms.Compose(
        [
            transform_factory(
                module_name=module['module_name']
            )(
                **module.get('kwargs', {})
            ) 
            for module in config_transform['input']
        ]
    )

    label_transform = transforms.Compose(
        [
            transform_factory(
                module_name=module['module_name']
            )(
                **module.get('kwargs', {})
            ) 
            for module in config_transform['label']
        ]
    )

    return config_transform, input_transform, label_transform



@pytest.fixture
def fake_train_dataset(train_transforms):
    config_dataset = deepcopy(DATASET_CONFIG['train'])

    config_dataset['module_name'] = 'fake'
    config_dataset['kwargs']['n_references'] = 3
    config_dataset['kwargs']['n_samples'] = 1024
    config_dataset['kwargs']['batch_size'] = 32
    config_dataset['kwargs']['shuffle'] = True

    config_transform, input_transform, label_transform = train_transforms
    config_dataset['kwargs']['input_transform'] = input_transform
    config_dataset['kwargs']['label_transform'] = label_transform

    dataset = dataset_factory(
        module_name=config_dataset['module_name']
    )(
        **config_dataset.get('kwargs', {})
    )

    sample = dataset[0]
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)

    return config_dataset, config_transform, dataset


def test_custom_collate_fn(fake_train_dataset):
    config_dataset, config_transform, dataset = fake_train_dataset

    batch = [dataset[i] for i in range(config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1))]
    batch_X_collated, batch_Y_collated = custom_collate_fn(batch)

    assert config_transform['input'][2]['module_name'] == 'v2Resize'
    assert config_transform['label'][0]['module_name'] == 'cv2Resize'

    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][2]['kwargs']['size']
    
    # check shapes
    assert batch_X_collated.shape == (config_dataset['kwargs']['batch_size'], 1, target_input_size[0], target_input_size[1])
    assert batch_Y_collated.shape == (config_dataset['kwargs']['batch_size'], 1, target_label_size[0], target_label_size[1])


@pytest.fixture
def fake_train_dataloader(fake_train_dataset):
    config_dataset, config_transform, dataset = fake_train_dataset

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1),
        shuffle=config_dataset['kwargs']['shuffle'],
        collate_fn=custom_collate_fn
    )

    return config_dataset, config_transform, dataset, dataloader


def test_fake_dataloader(fake_train_dataloader):
    config_dataset, config_transform, _, dataloader = fake_train_dataloader

    batch_X, batch_Y = next(iter(dataloader))

    # check values
    assert batch_X.dtype == torch.float32
    assert 0 <= batch_X.min() <= batch_X.max() <= 1
    assert batch_Y.dtype == torch.float32
    assert 0 <= batch_Y.min() <= batch_Y.max() <= 1

    # visual check
    tile = []
    batch_X = (batch_X.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    batch_Y = (batch_Y.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for input_, label in zip(batch_X, batch_Y):
        tile.append(cv2.cvtColor(input_, cv2.COLOR_GRAY2BGR))
        tile.append(cv2.cvtColor(label, cv2.COLOR_GRAY2BGR))
    tile = imgviz.tile(
        tile, 
        border=(255, 255, 255),
        border_width=5
    )
    
    window_title = 'Sample images in a batch (left to right, top to bottom). True [t] or False [f]?'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, tile)
    key = cv2.waitKey(0)
    assert chr(key).lower().strip() == 't'


if __name__ == '__main__':
    pytest.main([__file__])
    



