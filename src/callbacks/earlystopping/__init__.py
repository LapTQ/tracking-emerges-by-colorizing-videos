from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.utils.logger import parse_save_checkpoint_path

# ==================================================================================================

import logging
import os
import torch


logger = logging.getLogger(__name__)


class CustomCallback:

    def __init__(
            self,
            **kwargs
    ):
        # parse kwargs
        model = kwargs.get('model')
        patience = kwargs.get('patience')
        mode = kwargs.get('mode')
        min_delta = kwargs.get('min_delta', 0)
        checkpoint_path = kwargs.get('checkpoint_path', None)

        self.model = model
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best = None
        self.counter = 0
        self._first_time = True

        assert self.mode in ['min', 'max']

    
    def step(
            self,
            **kwargs
    ):
        value = kwargs.get('value')

        if self.best is None:
            self.best = value
            self.save_checkpoint()
            return True
        
        is_better = value > self.best + self.min_delta if self.mode == 'max' \
            else value < self.best - self.min_delta
        
        if is_better:
            self.best = value
            self.counter = 0
            self.save_checkpoint()
            return True
        else:
            self.counter += 1
            logger.warning(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter > self.patience:
                logger.warning('EarlyStopping: Excceeding patience.')
                return False
            return True
        
    
    def save_checkpoint(
            self,
    ):
        if self.checkpoint_path is None:
            return
        
        file_path = parse_save_checkpoint_path(
            input_path=self.checkpoint_path,
            ext='pth'
        )

        if self._first_time:
            parent, name = os.path.split(file_path)
            basename, ext = os.path.splitext(name)
            self.checkpoint_path = os.path.join(
                parent,
                f'{basename}_best{ext}'
            )
            self._first_time = False
        else:
            self.checkpoint_path = file_path

        torch.save(self.model.state_dict(), self.checkpoint_path)
        logger.info('EarlyStopping: Save checkpoint at {}'.format(self.checkpoint_path))


        

        