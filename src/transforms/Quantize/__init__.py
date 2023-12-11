
from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
LOGGER = GLOBAL.LOGGER

# ==================================================================================================

from torch import nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import os
from datetime import datetime
import numpy as np


EXPECTED_INPUT_SHAPE = {
    OneHotEncoder: (-1, 1),
    LabelEncoder: (-1,),
}


class CustomTransform(nn.Module):
    """Custom transform to quantize data."""


    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.model = kwargs['model']
        self.encoder = kwargs['encoder']
        self.checkpoint_path = kwargs.get('checkpoint_path', None)

        self.n_clusters = kwargs['model']['kwargs']['n_clusters']
        self.model_cls = eval(self.model['module_name'])
        self.model = self.model_cls(
            **self.model.get('kwargs', {})
        )

        self.encoder_cls = eval(self.encoder)
        self.encoder = self.encoder_cls()
        self._expected_input_shape = EXPECTED_INPUT_SHAPE[self.encoder_cls]

        self.is_fitted = False

        if self.checkpoint_path is not None:
            self.checkpoint_path = self.checkpoint_path.strip()
            self._load_checkpoint()
    

    def fit(self, X):
        assert len(X.shape) == 4, 'Input shape must be (N, H, W, C).'
        X = X.reshape(-1, X.shape[-1])
        self.model.fit(X)
        self.encoder.fit(self.model.labels_.reshape(*self._expected_input_shape))
        self.is_fitted = True
        LOGGER.info('Model {} and encoder {} fitted.'.format(self.model, self.encoder))

        if self.checkpoint_path is not None:
            self._save_checkpoint()
    

    def get_params(self):
        return {
            'model': self.model,
            'encoder': self.encoder,
        }

    
    def _load_checkpoint(self):
        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so loading checkpoint is not allowed.')
        
        if not os.path.exists(self.checkpoint_path):
            LOGGER.warning('Checkpoint path was set but {} does not exist. Starting from scratch'.format(self.checkpoint_path))
            return

        if os.path.isdir(self.checkpoint_path):
            filenames = [f for f in os.listdir(self.checkpoint_path) if f.endswith('.pkl')]
            if len(filenames) == 0:
                LOGGER.warning('No checkpoint .pkl file exists in {}. Starting from scratch.'.format(self.checkpoint_path))
                return
            last = sorted(filenames)[-1]
            LOGGER.warning('{} is a directory. So new checkpoint will be created after each fit.'.format(self.checkpoint_path))
            
            file_path = os.path.join(self.checkpoint_path, last)
            LOGGER.info('Loading the last checkpoint for {} at {}'.format(str(self), file_path))
        else:
            assert self.checkpoint_path.endswith('.pkl'), 'Checkpoint path must be a .pkl file.'
            LOGGER.warning('{} is a file. So it will be overwritten after each fit.'.format(self.checkpoint_path))
            file_path = self.checkpoint_path
        
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)
        
        loaded_model = loaded_file['model']
        loaded_encoder = loaded_file['encoder']
        
        assert isinstance(loaded_model, self.model_cls), 'Loaded model is not an instance of {}'.format(self.model_cls)
        assert isinstance(loaded_encoder, self.encoder_cls), 'Loaded encoder is not an instance of {}'.format(self.encoder_cls)
        self.model = loaded_model
        self.encoder = loaded_encoder
        self.is_fitted = True

    
    def _save_checkpoint(self):
        if not self.is_fitted:
            raise ValueError('Model is not fitted yet.')

        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so saving checkpoint is not allowed.')
        
        # check if the path is a directory or file
        parent, filename = os.path.split(self.checkpoint_path)
        is_dir = '.' not in filename

        os.makedirs(parent, exist_ok=True)
        if is_dir:
            filename = 'checkpoint_{}.pkl'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.checkpoint_path = os.path.join(parent, filename)
        
        with open(self.checkpoint_path, 'wb') as f:
            pickle.dump(
                {
                    'model': self.model,
                    'encoder': self.encoder,
                },
                f
            )
        
        LOGGER.info('New checkpoint for {} is saved to {}'.format(str(self), self.checkpoint_path))

    
    def forward(
            self,
            x
    ):  
        if not self.is_fitted:
            raise ValueError('Model is not fitted yet.')

        H, W, C = x.shape
        x = x.reshape(-1, C)
        x = self.model.predict(x)
        x = self.encoder.transform(x.reshape(*self._expected_input_shape))
        if not isinstance(x, np.ndarray):
            x = x.toarray()
        x = x.reshape(H, W, -1)
        return x
    

    def invert_transform_batch(
            self,
            x
    ):
        """Return the quantized value of the transformed data."""

        if not self.is_fitted:
            raise ValueError('Model is not fitted yet.')
        
        assert len(x.shape) == 4, 'Input shape must be (N, H, W, C).'
        
        N, H, W, C = x.shape
        x = x.reshape(-1, C)
        x = self.encoder.inverse_transform(x)
        x = self.model.cluster_centers_[x]
        x = x.reshape(N, H, W, -1)
        return x


    def __str__(self):
        return 'Quantize'
        
