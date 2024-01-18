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


@pytest.fixture
def config_dataset_fake():
    config_dataset = {
        'module_name': 'fake',
        'kwargs': {
            'n_references': 3,
            'n_samples': 1024,
            'batch_size': 32,
        }
    }

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    return config_dataset


@pytest.fixture
def config_dataset_kinetics700():
    config_dataset = {
        'module_name': 'kinetics700',
        'kwargs': {
            'dataset_dir': 'data/k700-2020/train',
            'n_references': 3,
            'n_samples': 1024,
            'batch_size': 16,
            'shuffle': True,
            'frame_rate': 6,
        }
    }

    assert config_dataset['kwargs']['batch_size'] % (config_dataset['kwargs']['n_references'] + 1) == 0

    return config_dataset


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


@pytest.mark.parametrize(
        'config_dataset_template',
        [
            'config_dataset_fake',
            'config_dataset_kinetics700',
        ]
)
def test_cv2Resize(config_dataset_template, request):
    config_transform = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 64),
            }
        }
    ]

    config_dataset = request.getfixturevalue(config_dataset_template)

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
        window_title='Check cv2Resize transform with (H, W)={}, dataset={}. True [t] or False [f]?'\
            .format(config_transform[0]['kwargs']['size'], config_dataset_template),
    )
    assert chr(key).lower().strip() == 't'


@pytest.mark.parametrize(
        'config_dataset_template',
        [
            'config_dataset_fake',
            'config_dataset_kinetics700',
        ]
)
def test_cvtColor(config_dataset_template, request):
    config_transform = [
        {
            'module_name': 'cv2cvtColor',
            'kwargs': {
                'code': 'cv2.COLOR_BGR2LAB',
            }
        }
    ]

    config_dataset = request.getfixturevalue(config_dataset_template)

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
        window_title='Check cv2cvtColor transform with code={}, dataset={}. True [t] or False [f]?'\
            .format(config_transform[0]['kwargs']['code'], config_dataset_template),
    )
    assert chr(key).lower().strip() == 't'


@pytest.mark.parametrize(
        'config_dataset_template,channels,expected_dim,show', 
        [
            ('config_dataset_fake', [1, 2], 2, True), 
            ('config_dataset_fake', [0], 1, False), 
            ('config_dataset_fake', 0, 1, False),
            ('config_dataset_kinetics700', [1, 2], 2, True), 
            ('config_dataset_kinetics700', [0], 1, False), 
            ('config_dataset_kinetics700', 0, 1, False),
        ]
)
def test_ExtractChannel(config_dataset_template, channels, expected_dim, show, request):
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

    config_dataset = request.getfixturevalue(config_dataset_template)

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
            window_title='Check ExtractChannel transform with channels={}, dataset={}. True [t] or False [f]?'\
                .format(config_transform[1]['kwargs']['channels'], config_dataset_template),
        )
        assert chr(key).lower().strip() == 't'


@pytest.mark.parametrize(
        'config_dataset_template, encoder_name,n_clusters,expected_dim,expected_dtype,expected_max_value,close_cv2', 
        [
            ('config_dataset_fake', 'LabelEncoder', 16, 1, torch.int64, 15, False), 
            ('config_dataset_fake', 'OneHotEncoder', 16, 16, torch.float64, 1, False),
            ('config_dataset_fake', 'OneHotEncoder', 2, 2, torch.float64, 1, False),
            ('config_dataset_kinetics700', 'LabelEncoder', 16, 1, torch.int64, 15, False), 
            ('config_dataset_kinetics700', 'OneHotEncoder', 16, 16, torch.float64, 1, False),
            ('config_dataset_kinetics700', 'OneHotEncoder', 2, 2, torch.float64, 1, True)
        ]
)
def test_Quantize_semantic(config_dataset_template, encoder_name, n_clusters, expected_dim, expected_dtype, expected_max_value, close_cv2, request):
    config_dataset = request.getfixturevalue(config_dataset_template)
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
                'load_checkpoint_path': None,
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
        window_title='Check Quantize transform with n_clusters={} and {}, dataset={}. True [t] or False [f]?'\
            .format(
                n_clusters,
                encoder_name,
                config_dataset_template
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
    

def test_Quantize_checkpoint(config_dataset_fake):
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
                'load_checkpoint_path': None,
            }
        }
    ]

    # assuming that Quantize is the last transform
    assert config_transform[-1]['module_name'] == 'Quantize'

    config_dataset = config_dataset_fake
    Y_to_fit = get_fit_data(
        config_dataset=config_dataset,
        config_label_transform=config_transform[:-1],   # exclude Quantize
        n_fit=config_transform[-1]['kwargs']['n_fit']        
    )

    # 1. checkpoint path is file
    load_checkpoint_path = 'temp/test_cases/checkpoints/transform/Quantize/load/checkpoint.pkl'
    save_checkpoint_path = 'temp/test_cases/checkpoints/transform/Quantize/save/checkpoint.pkl'
    os.system('rm -rf {}'.format(os.path.split(load_checkpoint_path)[0]))
    parent, filename = os.path.split(save_checkpoint_path)
    os.system('rm -rf {}'.format(parent))

    config_transform[-1]['kwargs']['load_checkpoint_path'] = load_checkpoint_path
    config_transform[-1]['kwargs']['save_checkpoint_path'] = save_checkpoint_path

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
    assert os.path.exists(save_checkpoint_path)

    # 1.2. if parent directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}'.format(save_checkpoint_path))
    quantize_transform.fit(Y_to_fit)
    assert os.path.exists(save_checkpoint_path)

    # 1.3. if checkpoint file exists, it should be overwritten
    mtime = os.path.getmtime(save_checkpoint_path)
    quantize_transform.fit(Y_to_fit)
    assert mtime < os.path.getmtime(save_checkpoint_path)

    # 1.4. if checkpoint file exists, the Quantize object should be fitted after initialization
    config_transform[-1]['kwargs']['load_checkpoint_path'] = save_checkpoint_path
    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]
    assert quantize_transform.is_fitted
    config_transform[-1]['kwargs']['load_checkpoint_path'] = load_checkpoint_path

    # 2. checkpoint path is directory
    load_checkpoint_path = 'temp/test_cases/checkpoints/transform/Quantize/load'
    save_checkpoint_path = 'temp/test_cases/checkpoints/transform/Quantize/save'
    os.system('rm -rf {}'.format(load_checkpoint_path))
    os.system('rm -rf {}'.format(save_checkpoint_path))

    config_transform[-1]['kwargs']['load_checkpoint_path'] = load_checkpoint_path
    config_transform[-1]['kwargs']['save_checkpoint_path'] = save_checkpoint_path

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]

    # 2.1. if directory does not exist, then it and a file should only be created after fitting
    assert not os.path.exists(save_checkpoint_path)
    quantize_transform.fit(Y_to_fit)
    assert os.path.exists(save_checkpoint_path)
    assert len(os.listdir(save_checkpoint_path)) == 1

    # 2.2. if directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}/*'.format(save_checkpoint_path))
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(save_checkpoint_path)) == 1

    # 2.3. if a file exists, but the Quantize instance stays the same, then the file should be overwritten
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(save_checkpoint_path)) == 1

    # 2.4. if a file exists, but the Quantize instance changes, then a new file should be created
    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(Y_to_fit)
    assert len(os.listdir(save_checkpoint_path)) == 2

    # 3. checkpoint path is None
    os.system('rm -rf {}'.format(save_checkpoint_path))
    config_transform[-1]['kwargs']['load_checkpoint_path'] = None
    config_transform[-1]['kwargs']['save_checkpoint_path'] = None

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=None,
        config_label_transform=config_transform,
    )
    label_transform = _['label_transform']
    quantize_transform = label_transform.transforms[-1]

    quantize_transform.fit(Y_to_fit)
    assert not os.path.exists(load_checkpoint_path)
    assert not os.path.exists(save_checkpoint_path)


    os.system('rm -rf {}'.format(parent))


if __name__ == '__main__':
    pytest.main([__file__])


    

