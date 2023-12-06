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
SEED = 42


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


@pytest.fixture
def config_dataset_template():
    config_dataset = deepcopy(DATASET_CONFIG['train'])

    config_dataset['module_name'] = 'fake'
    config_dataset['kwargs']['n_references'] = 3
    config_dataset['kwargs']['n_samples'] = 20
    config_dataset['kwargs']['batch_size'] = 32
    config_dataset['kwargs']['shuffle'] = True

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    return config_dataset


@pytest.fixture
def config_transform_template():
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


def setup_data(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_input_transform = kwargs['config_input_transform']
    config_label_transform = kwargs['config_label_transform']

    input_transform = transforms.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_.get('kwargs', {})
            )
            for _ in config_input_transform
        ]
    )

    label_transform = transforms.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_.get('kwargs', {})
            )
            for _ in config_label_transform
        ]
    )

    config_dataset['kwargs']['input_transform'] = input_transform
    config_dataset['kwargs']['label_transform'] = label_transform

    dataset = dataset_factory(
        module_name=config_dataset['module_name'],
    )(
        **config_dataset.get('kwargs', {})
    )

    return dataset, input_transform, label_transform


def test_dataset_output(
        config_dataset_template,
        config_transform_template
):
    config_dataset = deepcopy(config_dataset_template)
    config_transform = deepcopy(config_transform_template)

    dataset, _, __ = setup_data(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label'][:-1]   # ignore Quantize
    )

    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    extracted_channels = config_transform['label'][2]['kwargs']['channels']
    extracted_channels = [extracted_channels] if isinstance(extracted_channels, int) else extracted_channels

    assert x.shape == (
        config_dataset['kwargs']['n_references'] + 1,
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert y.shape == (
        config_dataset['kwargs']['n_references'] + 1,
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

    dataset, _, __ = setup_data(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label'][:-1]   # ignore Quantize
    )

    batch = [dataset[i] for i in range(config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1))]
    batch_X_collated, batch_Y_collated = custom_collate_fn(batch)

    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    extracted_channels = config_transform['label'][2]['kwargs']['channels']
    extracted_channels = [extracted_channels] if isinstance(extracted_channels, int) else extracted_channels
    
    # check shapes
    assert batch_X_collated.shape == (
        config_dataset['kwargs']['batch_size'], 
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert batch_Y_collated.shape == (
        config_dataset['kwargs']['batch_size'], 
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

    # create dummy dataset to fit Quantize
    dummny_dataset, _, __ = setup_data(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label'][:-1]   # ignore Quantize
    )
    dummny_dataloader = DataLoader(
        dummny_dataset,
        batch_size=config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1),
        shuffle=config_dataset['kwargs']['shuffle'],
        collate_fn=custom_collate_fn
    )
    Y = []
    for _, batch_Y in dummny_dataloader:
        Y.append(batch_Y)
    dummy_Y = np.concatenate(Y, axis=0)

    dataset, _, label_transform = setup_data(
        config_dataset=config_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']    # include Quantize
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1),
        shuffle=config_dataset['kwargs']['shuffle'],
        collate_fn=custom_collate_fn
    )

    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(dummy_Y)

    set_seed()
    batch_X, batch_Y = next(iter(dataloader))


    # check shapes
    target_input_size = config_transform['input'][2]['kwargs']['size']
    target_label_size = config_transform['label'][0]['kwargs']['size']
    
    assert batch_X.shape == (
        config_dataset['kwargs']['batch_size'], 
        1, 
        target_input_size[0], 
        target_input_size[1]
    )
    assert batch_Y.shape == (
        config_dataset['kwargs']['batch_size'],  
        target_label_size[0], 
        target_label_size[1],
        1 if config_transform['label'][-1]['kwargs']['encoder'] == 'LabelEncoder' else config_transform['label'][-1]['kwargs']['model']['kwargs']['n_clusters']
    )
    

    # check values
    assert batch_X.dtype == torch.float32
    assert 0 <= batch_X.min() <= batch_X.max() <= 1
    assert batch_Y.dtype == (torch.float64 if config_transform['label'][-1]['kwargs']['encoder'] == 'OneHotEncoder' else torch.int64)
    assert (0 <= batch_Y.min() <= batch_Y.max() <= 1) if config_transform['label'][-1]['kwargs']['encoder'] == 'OneHotEncoder' else torch.all(batch_Y.max() < config_transform['label'][-1]['kwargs']['model']['kwargs']['n_clusters'])


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
    



