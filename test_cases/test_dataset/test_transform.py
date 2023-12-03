from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

from src.transform import transform_factory
from src.dataset import dataset_factory
from src.dataset.utils import custom_collate_fn
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


def get_dataloader(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_label_transform = kwargs['config_transform']

    label_transform = transform_factory(
        module_name=config_label_transform['module_name'],
    )(
        **config_label_transform['kwargs']
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

    tile = []
    batch_Y = (batch_Y.numpy()).astype(np.uint8)
    
    for label in batch_Y:
        tile.append(label)

    tile = imgviz.tile(
        tile, 
        border=(255, 255, 255),
        border_width=5
    )
    
    window_title = 'Sample labels in a batch (left to right, top to bottom). True [t] or False [f]?'
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, tile)
    key = cv2.waitKey(0)

    return key


def test_cv2Resize():
    config_transform = {
        'module_name': 'cv2Resize',
        'kwargs': {
            'size': (32, 64),
        }
    }

    

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)

    dataloader = get_dataloader(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    _, batch_Y = next(iter(dataloader))
    
    assert batch_Y.shape[1:3] == config_transform['kwargs']['size']
    
    key = visual_check(
        batch_Y=batch_Y,
    )
    assert chr(key).lower().strip() == 't'


if __name__ == '__main__':
    pytest.main([__file__])


    

