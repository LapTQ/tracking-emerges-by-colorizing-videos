from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

import src as GLOBAL
from src.dataset import load_dataset
import torch
from torch.utils.data import DataLoader
import pytest
from copy import deepcopy

DATASET_CONFIG = GLOBAL.CONFIG['dataset']


@pytest.fixture
def fake_dataset():
    config = deepcopy(DATASET_CONFIG)

    config['module_name'] = 'fake'
    config['kwargs']['n_references'] = 3
    config['kwargs']['image_size'] = (256, 256)
    config['kwargs']['n_samples'] = 1024
    config['kwargs']['batch_size'] = 32
    config['kwargs']['shuffle'] = True

    dataset = load_dataset(
        module_name=config['module_name']
    )(
        **config['kwargs']
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['kwargs']['batch_size'],
        shuffle=config['kwargs']['shuffle']
    )

    return config, dataloader


def test_fake_dataset(fake_dataset):
    config, dataloader = fake_dataset

    input_, label = next(iter(dataloader))

    assert len(input_) == config['kwargs']['n_references'] + 1
    assert len(label) == config['kwargs']['n_references'] + 1
    assert input_[0].shape == (config['kwargs']['batch_size'], 1, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])
    assert label[0].shape == (config['kwargs']['batch_size'], 3, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])
    assert input_[0].dtype == torch.float32
    assert 0 <= input_[0].min() <= input_[0].max() <= 1
    assert label[0].dtype == torch.float32
    assert 0 <= label[0].min() <= label[0].max() <= 1


if __name__ == '__main__':
    pytest.main([__file__])
    



