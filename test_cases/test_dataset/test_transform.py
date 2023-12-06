from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.transform import transform_factory
from src.dataset import dataset_factory
from src.dataset.utils import custom_collate_fn

# ==================================================================================================

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms
import cv2
import numpy as np
import imgviz
import pytest
from copy import deepcopy
import os


CONFIG_FAKE_DATASET = {
    'module_name': 'fake',
    'kwargs': {
        'n_references': 3,
        'n_samples': 20,
        'batch_size': 32,
        'shuffle': True,
    }
}
SEED = 42


def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)


def setup_data(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_label_transform = kwargs['config_transform']

    # get value for convenience
    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']

    assert batch_size % (n_references + 1) == 0

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
        batch_size=batch_size // (n_references + 1),
        shuffle=config_dataset['kwargs']['shuffle'],
        collate_fn=custom_collate_fn,
    )

    return label_transform, dataloader


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

    _, dataloader = setup_data(
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

    _, dataloader = setup_data(
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

    _, dataloader = setup_data(
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


@pytest.fixture
def Quantize_config_template():
    config_transform = [
        {
            'module_name': 'cv2Resize',
            'kwargs': {
                'size': (32, 64),
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
                'model': {
                    'module_name': 'KMeans',
                    'kwargs': {
                        'n_clusters': 16,
                    }
                },
                'encoder': None,
                'checkpoint_path': None,
            }
        }
    ]

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)
    _, dataloader = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform[:-1],
    )

    Y = []
    set_seed()
    for _, batch_Y in dataloader:
        Y.append(batch_Y)
    Y = np.concatenate(Y, axis=0)

    return config_transform, Y


def test_Quantize_semantic(Quantize_config_template):
    config_transform, Y = Quantize_config_template
    
    # assuming that Quantize is the last transform for convenience
    assert config_transform[-1]['module_name'] == 'Quantize'

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)

    # get value for convenience
    batch_size = config_dataset['kwargs']['batch_size']
    target_size = config_transform[0]['kwargs']['size']

    
    # check LabelEncoder
    config_transform[-1]['kwargs']['encoder'] = 'LabelEncoder'

    label_transform, dataloader = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(Y)

    set_seed()
    _, batch_Y = next(iter(dataloader))

    assert batch_Y.shape == (
        batch_size, 
        target_size[0], 
        target_size[1], 
        1
    )
    assert batch_Y.dtype == torch.int64
    assert torch.all(batch_Y < quantize_transform.n_clusters)

    
    # check OneHotEncoder
    config_transform[-1]['kwargs']['encoder'] = 'OneHotEncoder'

    label_transform, dataloader = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )

    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(Y)

    set_seed()
    _, batch_Y = next(iter(dataloader))
    assert batch_Y.shape == (
        batch_size, 
        target_size[0], 
        target_size[1], 
        quantize_transform.n_clusters
    )
    assert batch_Y.dtype == torch.float64
    assert torch.all(batch_Y < 2)


    # visual check, given OneHotEncoder
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
        window_title='Check Quantize transform with n_clusters={}. True [t] or False [f]?'.format(config_transform[-1]['kwargs']['model']['kwargs']['n_clusters']),
    )
    assert chr(key).lower().strip() == 't'

    cv2.destroyAllWindows()
    

def test_Quantize_checkpoint(Quantize_config_template):
    config_transform, Y = Quantize_config_template
    # assuming that Quantize is the last transform
    assert config_transform[-1]['module_name'] == 'Quantize'

    config_dataset = deepcopy(CONFIG_FAKE_DATASET)
    config_transform[-1]['kwargs']['encoder'] = 'LabelEncoder'

    # 1. check point path is file
    checkpoint_path = 'checkpoints/transform/Quantize/test_case/checkpoint.pkl'
    parent, filename = os.path.split(checkpoint_path)
    os.system('rm -rf {}'.format(parent))

    config_transform[-1]['kwargs']['checkpoint_path'] = checkpoint_path

    label_transform, _ = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )
    quantize_transform = label_transform.transforms[-1]
    
    # 1.1. if parent directory does not exist, then both parent and file should only be created after fitting
    assert not os.path.exists(parent)
    quantize_transform.fit(Y)
    assert os.path.exists(checkpoint_path)

    # 1.2. if parent directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}'.format(checkpoint_path))
    quantize_transform.fit(Y)
    assert os.path.exists(checkpoint_path)

    # 1.3. if checkpoint file exists, it should be overwritten
    mtime = os.path.getmtime(checkpoint_path)
    quantize_transform.fit(Y)
    assert mtime < os.path.getmtime(checkpoint_path)

    # 2. check point path is directory
    checkpoint_path = 'checkpoints/transform/Quantize/test_case/'
    os.system('rm -rf {}'.format(checkpoint_path))

    config_transform[-1]['kwargs']['checkpoint_path'] = checkpoint_path

    label_transform, _ = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )
    quantize_transform = label_transform.transforms[-1]

    # 2.1. if directory does not exist, then it and a file should only be created after fitting
    assert not os.path.exists(checkpoint_path)
    quantize_transform.fit(Y)
    assert os.path.exists(checkpoint_path)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.2. if directory exists, but checkpoint file does not exist, then file should only be created after fitting
    os.system('rm -rf {}/*'.format(checkpoint_path))
    quantize_transform.fit(Y)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.3. if a file exists, but the Quantize instance stays the same, then the file should be overwritten
    quantize_transform.fit(Y)
    assert len(os.listdir(checkpoint_path)) == 1

    # 2.4. if a file exists, but the Quantize instance changes, then a new file should be created
    label_transform, _ = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )
    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(Y)
    assert len(os.listdir(checkpoint_path)) == 2

    # 3. check point path is None
    os.system('rm -rf {}'.format(checkpoint_path))
    config_transform[-1]['kwargs']['checkpoint_path'] = None

    label_transform, _ = setup_data(
        config_dataset=config_dataset,
        config_transform=config_transform,
    )
    quantize_transform = label_transform.transforms[-1]

    quantize_transform.fit(Y)
    assert not os.path.exists(checkpoint_path)


    os.system('rm -rf {}'.format(checkpoint_path))


if __name__ == '__main__':
    pytest.main([__file__])


    

