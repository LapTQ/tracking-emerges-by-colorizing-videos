
from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent.parent

sys.path.append(str(ROOT_DIR))

import src as GLOBAL
from src.utils.logger import parse_save_checkpoint_path, parse_load_checkpoint_path
LOGGER = GLOBAL.LOGGER

# ==================================================================================================

from torch import nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pickle
import os
import numpy as np


EXPECTED_INPUT_SHAPE = {
    OneHotEncoder: (-1, 1),
    LabelEncoder: (-1,),
}
EXPECTED_CAT_ATTR = {
    OneHotEncoder: 'categories_',
    LabelEncoder: 'classes_',
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
            self.load_checkpoint()
    

    def fit(self, X):
        assert len(X.shape) == 4, 'Input shape must be (B, H, W, C).'
        X = X.reshape(-1, X.shape[-1])
        self.model.fit(X)
        self.encoder.fit(self.model.labels_.reshape(*self._expected_input_shape))
        self.is_fitted = True
        LOGGER.info('Model {} and encoder {} fitted.'.format(self.model, self.encoder))

        if self.checkpoint_path is not None:
            self.save_checkpoint()
    

    def get_params(self):
        return {
            'model': self.model,
            'encoder': self.encoder,
        }

    
    def load_checkpoint(self):
        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so loading checkpoint is not allowed.')
        
        file_path = parse_load_checkpoint_path(
            input_path=self.checkpoint_path,
            ext='pkl',
        )

        if file_path == -1:
            return
        
        with open(file_path, 'rb') as f:
            loaded_file = pickle.load(f)
        
        loaded_model = loaded_file['model']
        loaded_encoder = loaded_file['encoder']
        
        assert isinstance(loaded_model, self.model_cls), 'Loaded model is not an instance of {}'.format(self.model_cls)
        assert isinstance(loaded_encoder, self.encoder_cls), 'Loaded encoder is not an instance of {}'.format(self.encoder_cls)
        self.model = loaded_model
        self.encoder = loaded_encoder
        self.is_fitted = True
        LOGGER.info('Quantize model loaded successfully:\n{}'.format(str(self)))

    
    def save_checkpoint(self):
        if not self.is_fitted:
            raise ValueError('Model is not fitted yet.')

        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so saving checkpoint is not allowed.')
        
        self.checkpoint_path = parse_save_checkpoint_path(
            input_path=self.checkpoint_path,
            ext='pkl',
        )
        
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
        return 'Quantize(\n\
    model: {}(\n\
        cluster_centers_: {}\n\
    ),\n\
    encoder: {}(\n\
        categories_: {}\n\
    ),\n\
)'.format(
    self.model_cls,
    self.model.cluster_centers_,
    self.encoder_cls,
    eval('self.encoder.{}'.format(EXPECTED_CAT_ATTR[self.encoder_cls])),
)
        
