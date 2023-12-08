from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL

from src.models import model_factory
from src.datasets import dataset_factory
from src.datasets.utils import custom_collate_fn
from src.transforms import transform_factory

# ==================================================================================================

import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader
import numpy as np


def setup_dataset(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_input_transform = kwargs['config_input_transform']
    config_label_transform = kwargs['config_label_transform']

    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']
    shuffle = config_dataset['kwargs']['shuffle']

    input_transform = T.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_.get('kwargs', {})
            )
            for _ in config_input_transform
        ]
    ) if config_input_transform is not None else None

    label_transform = T.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_.get('kwargs', {})
            )
            for _ in config_label_transform
        ]
    ) if config_label_transform is not None else None

    config_dataset['kwargs']['input_transform'] = input_transform
    config_dataset['kwargs']['label_transform'] = label_transform

    dataset = dataset_factory(
        module_name=config_dataset['module_name'],
    )(
        **config_dataset.get('kwargs', {})
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size // (n_references + 1),
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )

    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'input_transform': input_transform, 
        'label_transform': label_transform
    }


def setup_dataset_and_transform(
        **kwargs
):
    # parse kwargs
    config_dataset = kwargs['config_dataset']
    config_input_transform = kwargs['config_input_transform']
    config_label_transform = kwargs['config_label_transform']

    assert config_label_transform[-1]['module_name'] == 'Quantize'

    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']
    shuffle = config_dataset['kwargs']['shuffle']

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=config_input_transform,
        config_label_transform=config_label_transform[:-1] \
            if config_label_transform is not None else None    # exclude Quantize
    )
    dummy_dataloader = _['dataloader']
    input_transform = _['input_transform']  
    

    Y = []
    for _, batch_Y in dummy_dataloader:
        Y.append(batch_Y)
    dummy_Y = np.concatenate(Y, axis=0)


    label_transform = T.Compose(
        [
            transform_factory(
                module_name=_['module_name'],
            )(
                **_.get('kwargs', {})
            )
            for _ in config_label_transform   # include Quantize
        ]
    ) if config_label_transform is not None else None
    quantize_transform = label_transform.transforms[-1]
    quantize_transform.fit(dummy_Y)

    config_dataset['kwargs']['label_transform'] = label_transform

    dataset = dataset_factory(
        module_name=config_dataset['module_name'],
    )(
        **config_dataset.get('kwargs', {})
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size // (n_references + 1),
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )

    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'input_transform': input_transform, 
        'label_transform': label_transform
    }