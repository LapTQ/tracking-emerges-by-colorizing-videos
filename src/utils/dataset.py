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
import math


def setup_dataset(
        **kwargs
):
    """Helper function to setup dataset.
    Possible kwargs:
        config_dataset: dict
        config_input_transform: list of dict, or None. If input_transform is specified, this parameter must not be set.
        config_label_transform: list of dict, or None. If label_transform is specified, this parameter must not be set.
        input_transform: T.Compose, or list of Torchvision transforms, or None. If config_input_transform is specified, this parameter must not be set.
        label_transform: T.Compose, or list of Torchvision transforms, or None. If config_label_transform is specified, this parameter must not be set.
    """
    ########### parse kwargs
    config_dataset = kwargs['config_dataset']
    
    assert not ('config_input_transform' in kwargs and 'input_transform' in kwargs)  # only one of them can be specified
    assert 'config_input_transform' in kwargs or 'input_transform' in kwargs         # one of them must be specified
    assert not ('config_label_transform' in kwargs and 'label_transform' in kwargs)  # only one of them can be specified
    assert 'config_label_transform' in kwargs or 'label_transform' in kwargs         # one of them must be specified
    
    if 'config_input_transform' in kwargs:
        config_input_transform = kwargs['config_input_transform']
        assert isinstance(config_input_transform, list) or config_input_transform is None 
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
    else:
        input_transform = kwargs['input_transform']
        assert isinstance(input_transform, [T.Compose, list]) or input_transform is None
        if isinstance(input_transform, list):
            input_transform = T.Compose(input_transform)
    
    if 'config_label_transform' in kwargs:
        config_label_transform = kwargs['config_label_transform']
        assert isinstance(config_label_transform, list) or config_label_transform is None
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
    else:
        label_transform = kwargs['label_transform']
        assert isinstance(label_transform, (T.Compose, list)) or label_transform is None
        if isinstance(label_transform, list):
            label_transform = T.Compose(label_transform)
    ##########

    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']
    shuffle = config_dataset['kwargs']['shuffle']

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
    assert isinstance(config_input_transform, list) or config_input_transform is None
    assert isinstance(config_label_transform, list) or config_label_transform is None

    batch_size = config_dataset['kwargs']['batch_size']
    n_references = config_dataset['kwargs']['n_references']

    label_require_fits = [cfg.get('kwargs', {}).get('require_fit', False) for cfg in (config_label_transform or [])]
    label_n_sample_fits = [cfg.get('kwargs', {}).get('n_fit', 0) for cfg in (config_label_transform or [])]
    label_n_batch_fits = [math.ceil(n_sample_fit * (n_references + 1) / batch_size) for n_sample_fit in label_n_sample_fits]
    ls_label_transform = [
        transform_factory(
            module_name=cfg['module_name'],
        )(
            **cfg.get('kwargs', {})
        )
        for cfg in (config_label_transform or [])
    ]

    for i, (require_fit, n_sample_fit, n_batch_fit, transform)  \
        in enumerate(zip(label_require_fits, label_n_batch_fits, label_n_batch_fits, ls_label_transform)):
        if not require_fit or transform.is_fitted:
            continue
        
        assert n_sample_fit is not None
        _ = setup_dataset(
            config_dataset=config_dataset,
            config_input_transform=config_input_transform,
            label_transform=ls_label_transform[:i]     # exclude this transform
        )
        dummy_dataloader = _['dataloader']

        Y_to_fit = []
        batch_iter = iter(dummy_dataloader)
        for _ in range(n_batch_fit):
            _, batch_Y = next(batch_iter)
            Y_to_fit.append(batch_Y)
        Y_to_fit = np.concatenate(Y_to_fit, axis=0)
        transform.fit(Y_to_fit[:n_sample_fit])
    
    label_transform = ls_label_transform if config_label_transform is not None else None

    _ = setup_dataset(
        config_dataset=config_dataset,
        config_input_transform=config_input_transform,
        label_transform=label_transform
    )
    dataset = _['dataset']
    dataloader = _['dataloader']
    input_transform = _['input_transform']
    label_transform = _['label_transform']

    return {
        'dataset': dataset,
        'dataloader': dataloader,
        'input_transform': input_transform, 
        'label_transform': label_transform
    }