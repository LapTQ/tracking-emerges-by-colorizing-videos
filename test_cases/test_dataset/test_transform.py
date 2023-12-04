from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

from src.transform import transform_factory
from src.dataset import dataset_factory
from src.dataset.utils import custom_collate_fn
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import cv2
import numpy as np
import imgviz
import pytest
from copy import deepcopy


CONFIG_FAKE_DATASET = {
    'module_name': 'fake',
    'kwargs': {
        'n_references': 4,
        'n_samples': 1024,
        'batch_size': 32,
        'shuffle': True,
    }
}
SEED = 42


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


def get_dataloader(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_label_transform = kwargs['config_transform']

    label_transform = transforms.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_['kwargs']
            )
            for _ in config_label_transform
        ]
    )
    
    

    config_dataset['kwargs']['label_transform'] = label_transform

    dataset = dataset_factory(
        module_name=config_dataset['module_name'],
    )(
        **config_dataset.get('kwargs', {})
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config_dataset['kwargs']['batch_size'] // (config_dataset['kwargs']['n_references'] + 1),
        shuffle=config_dataset['kwargs']['shuffle'],
        collate_fn=custom_collate_fn,
    )

    return dataloader


def visual_check(
        **kwargs
):
    # parse kwargs
    batch_Y = kwargs['batch_Y']
    window_title = kwargs.get('window_title', 'Sample labels in a batch (left to right, top to bottom). True [t] or False [f]?')

    tile = []
    batch_Y = batch_Y.cpu().numpy().astype(np.uint8)
    
    for label in batch_Y:
        tile.append(label)

    tile = imgviz.tile(
        tile, 
        border=(255, 255, 255),
        border_width=5
    )
    
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, tile)
    key = cv2.waitKey(0)

    return key


def test_cv2Resize():
    config_transform = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 64),
            }
        }
    ]

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)

    dataloader = get_dataloader(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    set_seed()
    _, batch_Y = next(iter(dataloader))
    
    assert batch_Y.shape[1:3] == config_transform[0]['kwargs']['size']
    
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check cv2Resize transform with (H, W)={}. True [t] or False [f]?'.format(config_transform[0]['kwargs']['size']),
    )
    assert chr(key).lower().strip() == 't'


def test_cvtColor():
    config_transform = [
        {
            'module_name': 'cv2cvtColor',
            'kwargs': {
                'code': 'cv2.COLOR_BGR2LAB',
            }
        }
    ]

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)

    dataloader = get_dataloader(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    set_seed()
    _, batch_Y = next(iter(dataloader))
        
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check cv2cvtColor transform with code={}. True [t] or False [f]?'.format(config_transform[0]['kwargs']['code']),
    )
    assert chr(key).lower().strip() == 't'


def test_ExtractChannel():
    config_transform = [
        {
            'module_name': 'cv2cvtColor',
            'kwargs': {
                'code': 'cv2.COLOR_BGR2LAB',
            }
        },
        {
            'module_name': 'ExtractChannel',
            'kwargs': {
                'channels': [1, 2],
            }
        }
    ]

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)

    dataloader = get_dataloader(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    set_seed()
    _, batch_Y = next(iter(dataloader))

    # add a dummy L channel and convert back to BGR for visual comparison
    fixed_L = 200
    batch_Y = torch.cat(
        [
            fixed_L + torch.zeros_like(batch_Y[:, :, :, :1]), 
            batch_Y
        ], 
        dim=3
    )
    batch_Y = torch.stack(
        [
            torch.from_numpy(
                cv2.cvtColor(y.numpy(), cv2.COLOR_LAB2BGR)
            ) for y in batch_Y
        ], 
        dim=0
    )
        
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check ExtractChannel transform with channels={}. True [t] or False [f]?'.format(config_transform[1]['kwargs']['channels']),
    )
    assert chr(key).lower().strip() == 't'


if __name__ == '__main__':
    pytest.main([__file__])


    

