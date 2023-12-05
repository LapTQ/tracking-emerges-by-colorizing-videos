
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
        self.checkpoint = kwargs.get('checkpoint', None)

        self.n_clusters = kwargs['model']['kwargs']['n_clusters']
        self.model_cls = eval(self.model['module_name'])
        self.model = self.model_cls(
            **self.model.get('kwargs', {})
        )

        self.encoder_cls = eval(self.encoder)
        self.encoder = self.encoder_cls()
        self._expected_input_shape = (-1, 1) if self.encoder_cls == OneHotEncoder else (-1,)

        self._is_fitted = False

        if self.checkpoint is not None:
            self.load_checkpoint()
    

    def fit(self, X):
        assert len(X.shape) == 4, 'Input shape must be (N, H, W, C).'
        X = X.reshape(-1, X.shape[-1])
        self.model.fit(X)
        self.encoder.fit(self.model.labels_.reshape(*self._expected_input_shape))
        self._is_fitted = True
        LOGGER.info('Model {} fitted.'.format(self.model_cls))
    

    def get_params(self):
        return {
            'model': self.model,
            'encoder': self.encoder,
        }

    
    def load_checkpoint(self):
        if self.checkpoint is None:
            raise ValueError('Checkpoint argument is set to None, so loading checkpoint is not allowed.')
        
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError('Path does not exist.')

        if os.path.isdir(self.checkpoint):
            filenames = [f for f in os.listdir(self.checkpoint) if f.endswith('.pkl')]
            if len(filenames) == 0:
                raise FileNotFoundError('No checkpoint file exists in the directory.')
            last = sorted(filenames)[-1]
            self.checkpoint = os.path.join(self.checkpoint, last)
            LOGGER.info('Loading the last checkpoint for {} at {}'.format(self.model_cls, self.checkpoint))
        
        with open(self.checkpoint, 'rb') as f:
            loaded_file = pickle.loads(f)
        
        loaded_model = loaded_file['model']
        loaded_encoder = loaded_file['encoder']
        
        assert isinstance(loaded_model, self.model_cls), 'Loaded model is not an instance of {}'.format(self.model_cls)
        assert isinstance(loaded_encoder, self.encoder_cls), 'Loaded encoder is not an instance of {}'.format(self.encoder_cls)
        self.model = loaded_model
        self.encoder = loaded_encoder

    
    def save_checkpoint(self):
        if not self._is_fitted:
            raise ValueError('Model is not fitted yet.')

        if self.checkpoint is None:
            raise ValueError('Checkpoint argument is set to None, so saving checkpoint is not allowed.')
        
        # check if the path is a directory or file
        parent, filename = os.path.split(self.checkpoint)
        is_dir = '.' not in filename

        os.path.makedirs(parent, exist_ok=True)
        if is_dir:
            filename = 'checkpoint_{}.pkl'.format(datetime.now().strftime('%Y%m%d_%H%M%S'))
            self.checkpoint = os.path.join(parent, filename)
        
        with open(self.checkpoint, 'wb') as f:
            pickle.dumps(
                {
                    'model': self.model,
                    'encoder': self.encoder,
                },
                f
            )
        
        LOGGER.info('Checkpoint for {} saved to {}'.format(self.model_cls, self.checkpoint))

    
    def forward(
            self,
            x
    ):  
        if not self._is_fitted:
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
        if not self._is_fitted:
            raise ValueError('Model is not fitted yet.')
        
        assert len(x.shape) == 4, 'Input shape must be (N, H, W, C).'
        
        N, H, W, C = x.shape
        x = x.reshape(-1, C)
        x = self.encoder.inverse_transform(x)
        x = self.model.cluster_centers_[x]
        x = x.reshape(N, H, W, -1)
        return x
        
