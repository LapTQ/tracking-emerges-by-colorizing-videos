from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.utils.dataset import setup_dataset_and_transform
from src.models import model_factory
from src.callbacks import callback_factory
from src.utils.mics import set_seed, get_device
from src.engines.train import Trainer

# ==================================================================================================

import logging
from copy import deepcopy
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb


logger = logging.getLogger(__name__)


def train():

    config = deepcopy(GLOBAL.CONFIG)
    config_train_dataset = config['dataset']['train']
    config_val_dataset = config['dataset']['val']
    config_transform = config['transform']['train']
    config_model = config['model']
    config_training = config['training']
    config_wandb = config_training['wandb']

    n_references = config_train_dataset['kwargs']['n_references']
    assert config_val_dataset['kwargs']['n_references'] == n_references

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

    set_seed()
    _ = setup_dataset_and_transform(
        config_dataset=config_train_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    train_dataloader = _['dataloader']
    label_transform = _['label_transform']

    _ = setup_dataset_and_transform(
        config_dataset=config_val_dataset,
        config_input_transform=config_transform['input'],
        config_label_transform=config_transform['label']
    )
    val_dataloader = _['dataloader']

    model = model_factory(
        **config_model['module_name'],
    )(
        **config_model.get('kwargs', {}),
        checkpoint_path=config_training['checkpoint_path'],
    )
    logger.info('Model:\n{}'.format(model))

    device = get_device(config_training['device'])
    model = model.to(device)
    criterion = eval(config_training['loss'])()
    optimizer = eval(config_training['optimizer']['module_name'])(
        model.parameters(),
        **config_training['optimizer']['kwargs']
    )
    scheduler = eval(config_training['scheduler']['module_name'])(
        optimizer,
        **config_training['scheduler']['kwargs']
    )
    callbacks = [
        callback_factory(
            module_name=callback['module_name'],
        )(
            model=model,
            checkpoint_path=config_training['checkpoint_path'],
            **callback['kwargs']
        ) for callback in config_training.get('callbacks', [])
    ]
    epochs = config_training['epochs']
    verbose_step = config_training['verbose_step']

    wandb.login(key=config_wandb['api_key'])
    wandb.init(
        project='Tracking emerges by colorizing videos',
        config=config,
    )

    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        n_references=n_references,
        optimizer=optimizer,
        criterion=criterion,
    )

    assert config_transform['label'][-2]['module_name'] == 'Quantize', \
            'Assuming the second last label transform to be Quantize.'
    
    trainer.train(
        epochs=epochs,
        scheduler=scheduler,
        callbacks=callbacks,
        callback_targets=[cfg['kwargs']['target'] for cfg in config_training['callbacks']],
        quantize_transform=label_transform.transforms[-2],
        verbose_step=verbose_step,
    )


if __name__ == '__main__':
    train()