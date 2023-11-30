from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

# ==================================================================================================

import src as GLOBAL
from src.dataset import dataset_factory
from src.dataset.utils import custom_collate_fn
import torch
from torch.utils.data import DataLoader
import pytest
from copy import deepcopy

DATASET_CONFIG = GLOBAL.CONFIG['dataset']


@pytest.fixture
def fake_train_dataset():
    config = deepcopy(DATASET_CONFIG['train'])

    config['module_name'] = 'fake'
    config['kwargs']['n_references'] = 3
    config['kwargs']['image_size'] = (640, 360)
    config['kwargs']['n_samples'] = 1024
    config['kwargs']['batch_size'] = 32
    config['kwargs']['shuffle'] = True

    dataset = dataset_factory(
        module_name=config['module_name']
    )(
        **config['kwargs']
    )

    sample = dataset[0]
    assert isinstance(sample[0], torch.Tensor)
    assert isinstance(sample[1], torch.Tensor)

    return config, dataset


def test_custom_collate_fn(fake_train_dataset):
    config, dataset = fake_train_dataset

    batch = [dataset[i] for i in range(config['kwargs']['batch_size'] // (config['kwargs']['n_references'] + 1))]
    batch_X_collated, batch_Y_collated = custom_collate_fn(batch)
    
    assert batch_X_collated.shape == (config['kwargs']['batch_size'], 1, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])
    assert batch_Y_collated.shape == (config['kwargs']['batch_size'], 3, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])


@pytest.fixture
def fake_train_dataloader(fake_train_dataset):
    config, dataset = fake_train_dataset

    assert config['kwargs']['batch_size'] % (config['kwargs']['n_references'] + 1) == 0

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config['kwargs']['batch_size'] // (config['kwargs']['n_references'] + 1),
        shuffle=config['kwargs']['shuffle'],
        collate_fn=custom_collate_fn
    )

    return config, dataset, dataloader


def test_fake_dataloader(fake_train_dataloader):
    config, _, dataloader = fake_train_dataloader

    batch_X, batch_Y = next(iter(dataloader))

    assert batch_X.shape == (config['kwargs']['batch_size'], 1, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])
    assert batch_Y.shape == (config['kwargs']['batch_size'], 3, config['kwargs']['image_size'][1], config['kwargs']['image_size'][0])
    assert batch_X.dtype == torch.float32
    assert 0 <= batch_X.min() <= batch_X.max() <= 1
    assert batch_Y.dtype == torch.float32
    assert 0 <= batch_Y.min() <= batch_Y.max() <= 1


if __name__ == '__main__':
    pytest.main([__file__])
    



