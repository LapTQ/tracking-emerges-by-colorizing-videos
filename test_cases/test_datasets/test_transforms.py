from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.utils.dataset import setup_dataset, setup_dataset_and_transform
from src.utils.mics import set_seed

# ==================================================================================================

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import cv2
import numpy as np
import imgviz
import pytest
from copy import deepcopy
import os


CONFIG_DATASET = {
    'module_name': 'fake',
    'kwargs': {
        'n_references': 3,
        'n_samples': 1024,
        'batch_size': 32,
        'shuffle': True,
    }
}


def visual_check(
        **kwargs
):
    # parse kwargs
    batch_Y = kwargs['batch_Y']
    window_title = kwargs.get('window_title', 
                'Sample labels in a batch (left to right, top to bottom). True [t] or False [f]?'
    )

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

    config_dataset = deepcopy(CONFIG_DATASET)

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    dataloader = _['dataloader']

    set_seed()
    _, batch_Y = next(iter(dataloader))
    
    assert batch_Y.shape[1:3] == config_transform[0]['kwargs']['size']
    
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check cv2Resize transform with (H, W)={}. True [t] or False [f]?'\
            .format(config_transform[0]['kwargs']['size']),
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

    config_dataset = deepcopy(CONFIG_DATASET)

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    dataloader = _['dataloader']

    set_seed()
    _, batch_Y = next(iter(dataloader))
        
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check cv2cvtColor transform with code={}. True [t] or False [f]?'\
            .format(config_transform[0]['kwargs']['code']),
    )
    assert chr(key).lower().strip() == 't'


@pytest.mark.parametrize(
        'channels,expected_dim,show', 
        [
            ([1, 2], 2, True), 
            ([0], 1, False), 
            (0, 1, False)
        ]
)
def test_ExtractChannel(channels, expected_dim, show):
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
                'channels': channels,
            }
        }
    ]

    config_dataset = deepcopy(CONFIG_DATASET)

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    dataloader = _['dataloader']

    set_seed()
    _, batch_Y = next(iter(dataloader))

    assert batch_Y.shape[-1] == expected_dim

    # add a dummy L channel and convert back to BGR for visual comparison
    if show:  
        fixed_L = 200
        batch_Y = torch.cat(
            [
                fixed_L + torch.zeros_like(batch_Y[:, :, :, :3-expected_dim]), 
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
            window_title='Check ExtractChannel transform with channels={}. True [t] or False [f]?'\
                .format(config_transform[1]['kwargs']['channels']),
        )
        assert chr(key).lower().strip() == 't'


@pytest.mark.parametrize(
        'encoder_name,n_clusters,expected_dim,expected_dtype,expected_max_value,close_cv2', 
        [
            ('LabelEncoder', 16, 1, torch.int64, 15, False), 
            ('OneHotEncoder', 16, 16, torch.float64, 1, False),
            ('OneHotEncoder', 2, 2, torch.float64, 1, True)
        ]
)
def test_Quantize_semantic(encoder_name, n_clusters, expected_dim, expected_dtype, expected_max_value, close_cv2):
    config_dataset = deepcopy(CONFIG_DATASET)
    config_transform = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 32),
            }
        },
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
        },
        {
            'module_name': 'Quantize',
            'kwargs': {
                'require_fit': True,
                'n_fit': 96,
                'model': {
                    'module_name': 'KMeans',
                    'kwargs': {
                        'n_clusters': n_clusters,
                    }
                },
                'encoder': encoder_name,
                'checkpoint_path': None,
            }
        }
    ]
    
    # assuming that Quantize is the last transform for convenience
    assert config_transform[-1]['module_name'] == 'Quantize'
    
    # check shape, dtype, and max value
    set_seed()
    _ = setup_dataset_and_transform(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    dataloader = _['dataloader']
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]

    set_seed()
    _, batch_Y = next(iter(dataloader))

    assert batch_Y.shape[-1] == expected_dim
    assert batch_Y.dtype == expected_dtype
    assert torch.all(batch_Y <= expected_max_value)


    # visual check
    batch_Y = quantize_transform.invert_transform_batch(batch_Y.cpu().numpy())
    
    # add a dummy L channel and convert back to BGR for visual comparison
    fixed_L = 200
    batch_Y = np.concatenate(
        [
            fixed_L + np.zeros_like(batch_Y[:, :, :, :1]), 
            batch_Y
        ], 
        axis=3
    ).astype(np.uint8)
    batch_Y = torch.stack(
        [
            torch.from_numpy(
                cv2.cvtColor(y, cv2.COLOR_LAB2BGR)
            ) for y in batch_Y
        ], 
        dim=0
    )
        
    key = visual_check(
        batch_Y=batch_Y,
        window_title='Check Quantize transform with n_clusters={} and {}. True [t] or False [f]?'\
            .format(
                n_clusters,
                encoder_name
            ),
    )
    assert chr(key).lower().strip() == 't'

    if close_cv2:
        cv2.destroyAllWindows()


def get_fit_data(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_label_transform = kwargs['config_label_transform']
    n_fit = kwargs['n_fit']

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_label_transform,
    )
    dummy_dataloader = _['dataloader']

    Y_to_fit = []
    batch_iter = iter(dummy_dataloader)
    for _ in range(n_fit):
        _, batch_Y = next(batch_iter)
        Y_to_fit.append(batch_Y)
    Y_to_fit = np.concatenate(Y_to_fit, axis=0)

    return Y_to_fit
    

def test_Quantize_checkpoint():
    config_transform = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 32),
            }
        },
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
        },
        {
            'module_name': 'Quantize',
            'kwargs': {
                'require_fit': True,
                'n_fit': 3,
                'model': {
                    'module_name': 'KMeans',
                    'kwargs': {
                        'n_clusters': 2,
                    }
                },
                'encoder': 'LabelEncoder',
                'checkpoint_path': None,
            }
        }
    ]

    # assuming that Quantize is the last transform
    assert config_transform[-1]['module_name'] == 'Quantize'

    config_dataset = deepcopy(CONFIG_DATASET)
    Y_to_fit = get_fit_data(
        config_dataset=config_dataset,
        config_label_transform=config_transform[:-1],   # exclude Quantize
        n_fit=config_transform[-1]['kwargs']['n_fit']        
    )

    # 1. checkpoint path is file
    checkpoint_path = 'checkpoints/transform/Quantize/test_case/checkpoint.pkl'
    parent, filename = os.path.split(checkpoint_path)
    os.system('rm -rf {}'.format(parent))

    config_transform[-1]['kwargs']['checkpoint_path'] = checkpoint_path

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]
    
    # 1.1. if parent directory does not exist, then both parent and file should only be created after fitting
    assert not os.path.exists(parent)
    quantize_transform.fit(Y_to_fit)
    assert os.path.exists(checkpoint_path)

    # 1.2. if parent directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}'.format(checkpoint_path))
    quantize_transform.fit(Y_to_fit)
    assert os.path.exists(checkpoint_path)

    # 1.3. if checkpoint file exists, it should be overwritten
    mtime = os.path.getmtime(checkpoint_path)
    quantize_transform.fit(Y_to_fit)
    assert mtime < os.path.getmtime(checkpoint_path)

    # 1.4. if checkpoint file exists, the Quantize object should be fitted after initialization
    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]
    assert quantize_transform.is_fitted

    # 2. checkpoint path is directory
    checkpoint_path = 'checkpoints/transform/Quantize/test_case/'
    os.system('rm -rf {}'.format(checkpoint_path))

    config_transform[-1]['kwargs']['checkpoint_path'] = checkpoint_path

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]

    # 2.1. if directory does not exist, then it and a file should only be created after fitting
    assert not os.path.exists(checkpoint_path)
    quantize_transform.fit(Y_to_fit)
    assert os.path.exists(checkpoint_path)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.2. if directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}/*'.format(checkpoint_path))
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.3. if a file exists, but the Quantize instance stays the same, then the file should be overwritten
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.4. if a file exists, but the Quantize instance changes, then a new file should be created
    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(checkpoint_path)) == 2

    # 3. checkpoint path is None
    os.system('rm -rf {}'.format(checkpoint_path))
    config_transform[-1]['kwargs']['checkpoint_path'] = None

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]

    quantize_transform.fit(Y_to_fit)
    assert not os.path.exists(checkpoint_path)


    os.system('rm -rf {}'.format(parent))


if __name__ == '__main__':
    pytest.main([__file__])


    

