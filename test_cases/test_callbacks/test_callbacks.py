from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.models import model_factory
from src.callbacks import callback_factory

# ==================================================================================================

import pytest
import os


@pytest.fixture
def model():
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

    n_references = 3

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
        **config_model.get('kwargs', {}),
    )

    return model


@pytest.mark.parametrize(
        'patient,mode,min_delta,step_values,counters,returns,checkpoint_modified',
        [
            (5, 'min', 0, [10, 9, 8, 8, 7, 7.5, 7.2, 7, 6, 5], [0, 0, 0, 1, 0, 1, 2, 3, 0, 0], [True, True, True, True, True, True, True, True, True, True], [None, True, True, False, True, False, False, False, True, True]),
            (5, 'min', 0, [10, 9, 8, 8, 8, 8.5, 8.2, 8, 8.4, 7], [0, 0, 0, 1, 2, 3, 4, 5, 6, 0], [True, True, True, True, True, True, True, True, False, True], [None, True, True, False, False, False, False, False, False, True]),
            (5, 'min', 0.5, [10, 9, 8, 8, 7, 6.5, 7.2, 7, 6, 5], [0, 0, 0, 1, 0, 1, 2, 3, 0, 0], [True, True, True, True, True, True, True, True, True, True], [None, True, True, False, True, False, False, False, True, True]),
            (5, 'max', 0, [-10, -9, -8, -8, -7, -7.5, -7.2, -7, -6, -5], [0, 0, 0, 1, 0, 1, 2, 3, 0, 0], [True, True, True, True, True, True, True, True, True, True], [None, True, True, False, True, False, False, False, True, True])
        ]
)
def test_earlystopping(
        model,
        patient,
        mode, 
        min_delta,
        step_values,
        counters,
        returns,
        checkpoint_modified
):
    config_callback = {
        'module_name': 'earlystopping',
        'kwargs': {
            'patience': patient,
            'mode': mode,
            'min_delta': min_delta
        }
    }

    checkpoint_path = 'checkpoints/callbacks/earlystopping/test_case/'

    callback = callback_factory(
        module_name=config_callback['module_name']
    )(
        model=model,
        checkpoint_path=checkpoint_path,
        **config_callback.get('kwargs', {})
    )

    os.system(f'rm -rf {checkpoint_path}')

    for i, (value, count, returned_value, modified) in enumerate(zip(step_values, counters, returns, checkpoint_modified)):
        assert returned_value == callback.step(
            value=value
        )
        print(i, mode, min_delta, value)
        assert callback.counter == count
        
        assert len(os.listdir(checkpoint_path)) == 1
        if i == 0:
            mtime = os.path.getmtime(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
            continue

        previous_mtime = mtime
        mtime = os.path.getmtime(os.path.join(checkpoint_path, os.listdir(checkpoint_path)[0]))
        assert (mtime > previous_mtime) == modified
    
    os.system(f'rm -rf {checkpoint_path}')

    # in context with real training
    # 1. if checkpoint_path is directory
    model.checkpoint_path = checkpoint_path
    callback = callback_factory(
        module_name=config_callback['module_name']
    )(
        model=model,
        checkpoint_path=checkpoint_path,
        **config_callback.get('kwargs', {})
    )

    for i, (value, count, returned_value, modified) in enumerate(zip(step_values, counters, returns, checkpoint_modified)):
        model.save_checkpoint()
        assert returned_value == callback.step(
            value=value
        )
        print(i, mode, min_delta, value)
        assert callback.counter == count
        
        assert len(os.listdir(checkpoint_path)) == 2
        if i == 0:
            mtime = os.path.getmtime(os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1]))
            continue

        previous_mtime = mtime
        mtime = os.path.getmtime(os.path.join(checkpoint_path, sorted(os.listdir(checkpoint_path))[-1]))
        assert (mtime > previous_mtime) == modified
    
    os.system(f'rm -rf {checkpoint_path}')

    # 2. if checkpoint_path is file
    os.makedirs(checkpoint_path, exist_ok=True)
    model.save_checkpoint()
    checkpoint_path = model.checkpoint_path
    parent, name = os.path.split(checkpoint_path)
    callback = callback_factory(
        module_name=config_callback['module_name']
    )(
        model=model,
        checkpoint_path=checkpoint_path,
        **config_callback.get('kwargs', {})
    )

    for i, (value, count, returned_value, modified) in enumerate(zip(step_values, counters, returns, checkpoint_modified)):
        model.save_checkpoint()
        assert returned_value == callback.step(
            value=value
        )
        print(i, mode, min_delta, value)
        assert callback.counter == count
        
        assert len(os.listdir(parent)) == 2
        if i == 0:
            mtime = os.path.getmtime(os.path.join(parent, sorted(os.listdir(parent))[-1]))
            continue

        previous_mtime = mtime
        mtime = os.path.getmtime(os.path.join(parent, sorted(os.listdir(parent))[-1]))
        assert (mtime > previous_mtime) == modified
    
    os.system(f'rm -rf {parent}')
        







if __name__ == '__main__':
    pytest.main([__file__])


