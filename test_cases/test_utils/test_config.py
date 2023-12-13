from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL

# ==================================================================================================

import json
import pytest

LOGGER = GLOBAL.LOGGER


# LOGGER.info('This is the loaded config:\n{}'.format(json.dumps(GLOBAL.CONFIG, indent=4)))
# key = input('Is the config correct? True [t] or False [f]: ')
# assert key.lower().strip() == 't'

def test_key():

    if 'dataset' in GLOBAL.CONFIG:
        assert 'train' in GLOBAL.CONFIG['dataset']

        config_train = GLOBAL.CONFIG['dataset']['train']
        assert 'kwargs' in config_train
        assert 'n_references' in config_train['kwargs']
        
        assert config_train['kwargs']['batch_size'] % (config_train['kwargs']['n_references'] + 1) == 0, \
            'Batch size must be divisible by the number of references + 1'
        

    if 'transform' in GLOBAL.CONFIG:
        assert 'train' in GLOBAL.CONFIG['transform']

        train_transform = GLOBAL.CONFIG['transform']['train']
        assert 'input' in train_transform
        assert 'label' in train_transform

        assert train_transform['input'][0]['module_name'] == 'v2ToImage', \
            'Specific to the training input, ToImage and ToDtype must be the first, \
                because other transformations expect a Tensor/TvTensor and with (C, H, W)'
        assert train_transform['input'][1]['module_name'] == 'v2ToDtype', \
            'Specific to the training input, ToImage and ToDtype must be the first, \
                because other transformations expect a Tensor/TvTensor and with (C, H, W)'
        
    if 'model' in GLOBAL.CONFIG:
        assert 'head' in GLOBAL.CONFIG['model']


if __name__ == '__main__':
    pytest.main([__file__])
