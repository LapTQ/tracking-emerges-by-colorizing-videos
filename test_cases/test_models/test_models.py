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

import torch
from torch import nn
from copy import deepcopy
import pytest
import numpy as np
from copy import deepcopy
import os


CONFIG_DATASET = GLOBAL.CONFIG['dataset']
CONFIG_TRANSFORM = GLOBAL.CONFIG['transform']


@pytest.fixture
def config_dataset_template():
    config_dataset = deepcopy(CONFIG_DATASET['train'])

    config_dataset['module_name'] = 'fake'
    config_dataset['kwargs']['n_references'] = 3
    config_dataset['kwargs']['n_samples'] = 1024
    config_dataset['kwargs']['batch_size'] = 16
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
                'require_fit': True,
                'n_fit': 32,
                'model': {
                    'module_name': 'KMeans',
                    'kwargs': {
                        'n_clusters': 8,
                    },
                },
                'encoder': 'OneHotEncoder',
                'checkpoint_path': None
            }
        },
        {
            'module_name': 'v2ToImage',
        }
    ]

    # assume this order for convenience
    assert config_transform['input'][2]['module_name'] == 'v2Resize'
    assert config_transform['label'][0]['module_name'] == 'cv2Resize'
    assert config_transform['label'][2]['module_name'] == 'ExtractChannel'
    assert config_transform['label'][-2]['module_name'] == 'Quantize'
    assert config_transform['label'][-1]['module_name'] == 'v2ToImage'

    return config_transform


def test_model_shape(
        config_dataset_template,
        config_transform_template
):
    # use smaller models for quick test
    config_model = {
        'backbone': {
            'module_name': 'resnet18',
            'kwargs': {
                'mid_channels': [32, 32, 32, 32],
                'mid_strides': [1, 2, 1, 2]
            }
        },
        'head': {
            'module_name': 'convnet3d',
            'kwargs': {
                'mid_channels': 32,
                'out_channels': 16,
                'dilations': [1, 2, 4, 8, 16]
            }
        },
        'kwargs': {
            'use_softmax': True
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

    # reformat config_model
    config_model['module_name'] = {
        'backbone': config_model['backbone']['module_name'],
        'head': config_model['head']['module_name']
    }
    config_model['kwargs'] = {
        'backbone': config_model['backbone']['kwargs'],
        'head': config_model['head']['kwargs'],
        **config_model['kwargs']
    }

    # set model parameters to match the input
    config_model['kwargs']['backbone']['in_channels'] = 1
    config_model['kwargs']['head']['n_references'] = n_references
    config_model['kwargs']['head']['in_channels'] = config_model['backbone']['kwargs']['mid_channels'][-1]

    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {})
    )

    backbone = model.backbone
    head = model.head
    assert head.n_references == n_references

    X, Y = next(iter(dataloader))
    true_color = Y[[i for i in range(batch_size) if i % (n_references + 1) == n_references]]
    ref_colors = Y[[i for i in range(batch_size) if i % (n_references + 1) != n_references]]
    backbone_output = backbone(X)
    head_output = head(backbone_output)
    ref_colors = ref_colors.float()
    model_output = model(X, ref_colors)

    # check shape
    assert backbone_output.shape == (
        batch_size,
        config_model['backbone']['kwargs']['mid_channels'][-1],
        target_label_size[0],
        target_label_size[1]
    )
    assert head_output.shape == (
        batch_size,
        config_model['kwargs']['head']['out_channels'],
        target_label_size[0],
        target_label_size[1]
    )
    assert model_output.shape == true_color.shape    

    # check gradient
    loss = nn.CrossEntropyLoss()
    l = loss(model_output, true_color)
    l.backward()

    for name, param in model.named_parameters():
        assert (torch.abs(param.grad) > 1e-4).any(), 'No element of {} has gradient greater than 1e-4'.format(name)

    
def test_model_checkpoint(
        config_dataset_template,
):
    # use smaller models for quick test
    config_model = {
        'backbone': {
            'module_name': 'resnet18',
            'kwargs': {
                'mid_channels': [32, 32, 32, 32],
                'mid_strides': [1, 2, 1, 2]
            }
        },
        'head': {
            'module_name': 'convnet3d',
            'kwargs': {
                'mid_channels': 32,
                'out_channels': 16,
                'dilations': [1, 2, 4, 8, 16]
            }
        },
        'kwargs': {
            'use_softmax': True
        }
    }

    n_references = config_dataset_template['kwargs']['n_references']

    # reformat config_model
    config_model['module_name'] = {
        'backbone': config_model['backbone']['module_name'],
        'head': config_model['head']['module_name']
    }
    config_model['kwargs'] = {
        'backbone': config_model['backbone']['kwargs'],
        'head': config_model['head']['kwargs'],
        **config_model['kwargs']
    }

    # set model parameters to match the input
    config_model['kwargs']['backbone']['in_channels'] = 1
    config_model['kwargs']['head']['n_references'] = n_references
    config_model['kwargs']['head']['in_channels'] = config_model['backbone']['kwargs']['mid_channels'][-1]

       
    # 1. checkpoint path is file
    checkpoint_path  = 'checkpoints/model/test_case/last.pth'
    parent, filename = os.path.split(checkpoint_path)
    os.system('rm -rf {}'.format(parent))

    # 1.1. if parent directory does not exist, then file should not be created until being explicitly evoked.
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    assert not os.path.exists(parent)
    model.save_checkpoint()
    assert os.path.exists(checkpoint_path)

    # 1.2. if parent directory exists, but checkpoint file does not exists, then file should not be created until being explicitly evoked.
    os.system('rm -rf {}'.format(parent))
    os.makedirs(parent)
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    assert not os.path.exists(checkpoint_path)
    model.save_checkpoint()
    assert os.path.exists(checkpoint_path)

    # 1.3. if checkpoint file exists, it should be overwritten.
    mtime = os.path.getmtime(checkpoint_path)
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    assert mtime == os.path.getmtime(checkpoint_path)
    model.save_checkpoint()
    new_mtime = os.path.getmtime(checkpoint_path)
    assert mtime < new_mtime
    model.save_checkpoint()
    assert new_mtime < os.path.getmtime(checkpoint_path)
    assert len(os.listdir(parent)) == 1

    # 2. checkpoint path is directory
    checkpoint_path  = 'checkpoints/model/test_case/'
    os.system('rm -rf {}'.format(checkpoint_path))

    # 2.1. if directory does not exist, then file should not be created until being explicitly evoked.
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    assert not os.path.exists(checkpoint_path)
    model.save_checkpoint()
    assert os.path.exists(checkpoint_path)
    assert len(os.listdir(checkpoint_path)) == 1
    assert model.checkpoint_path == os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])

    # 2.2. if directory exists, but checkpoint file does not exists, then file should not be created until being explicitly evoked.
    os.system('rm -rf {}'.format(checkpoint_path))
    os.makedirs(checkpoint_path)
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    assert len(os.listdir(checkpoint_path)) == 0
    model.save_checkpoint()
    assert len(os.listdir(checkpoint_path)) == 1
    assert model.checkpoint_path == os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])

    # 2.3. if a file exists, but the model instance stays the same, then the file should be overwritten.
    mtime = os.path.getmtime(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    model.save_checkpoint()
    assert len(os.listdir(checkpoint_path)) == 1
    assert mtime < os.path.getmtime(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
    assert model.checkpoint_path == os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0])

    # 2.4. if a file exists, but the model instance changes, then a new file should be created.
    model = model_factory(
        **config_model['module_name']
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=checkpoint_path
    )
    model.save_checkpoint()
    assert len(os.listdir(checkpoint_path)) == 2
    assert model.checkpoint_path == os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1])

    os.system('rm -rf {}'.format(parent))


if __name__ == '__main__':
    pytest.main([__file__])

    






