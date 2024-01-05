from pathlib import Path
import sys

HERE = Path(__file__).parent
ROOT_DIR = HERE.parent.parent

sys.path.append(str(ROOT_DIR))

from src.utils.logger import parse_save_checkpoint_path, parse_load_checkpoint_path

# ==================================================================================================

import logging
import torch
from torch import nn
import torch.nn.functional as F
from .backbone import backbone_factory
from .head import head_factory


logger = logging.getLogger(__name__)


def model_factory(
        **kwargs
):
    # parse kwargs
    module_name = kwargs

    model_builder = Colorizer.Builder(
        **module_name
    )

    return model_builder


class Colorizer(nn.Module):

    class Builder:

        def __init__(
                self,
                **kwargs
        ):
            self.reset()

            self.backbone_name = kwargs['backbone']
            self.head_name = kwargs['head']


        def reset(
                self,
        ):
            self._product = Colorizer()
        

        def __call__(
                self,
                **kwargs
        ):
            # parse kwargs
            backbone_kwargs = kwargs['backbone']
            head_kwargs = kwargs['head']
            use_softmax = kwargs['use_softmax']
            checkpoint_path = kwargs.get('checkpoint_path', None)

            self._product.backbone = backbone_factory(
                module_name=self.backbone_name
            )(
                **backbone_kwargs
            )

            self._product.head = head_factory(
                module_name=self.head_name
            )(
                **head_kwargs
            )

            assert 'n_references' in head_kwargs
            self._product.n_references = head_kwargs['n_references']
            self._product.use_softmax = use_softmax
            self._product.checkpoint_path = checkpoint_path

            if checkpoint_path is not None:
                self._product.load_checkpoint()

            product = self._product
            self.reset()
            return product
        
    
    def forward(
            self,
            x,
            y
    ):
        """Forward pass.
        Args:
            x (torch.Tensor): Input of shape (B1, 1, H, W) where B includes both reference and target images.
            y (torch.Tensor): Reference colors of shape (B2, C, H, W) which includes only reference images."""
        B = x.shape[0]
        assert B / (self.n_references + 1) * self.n_references == y.shape[0]
        x = self.backbone(x)
        x = self.head(x)
        sim = self.simmat(x)
        out = self.predict(sim, y)

        return out
    

    def simmat(
            self,
            x
    ):
        """Compute similarity matrix from backbone output.
        Args:
            x (torch.Tensor): Backbone output of shape (B, C, H, W), where B % (n_references + 1) == 0.
        Returns:
            sim (torch.Tensor): Similarity matrix of shape (B / (n_references + 1), n_references*H*W, H*W).
        """
        B, C, H, W = x.shape
        assert B % (self.n_references + 1) == 0
        ref = x[[i for i in range(B) if i % (self.n_references + 1) != self.n_references]]
        tar = x[[i for i in range(B) if i % (self.n_references + 1) == self.n_references]]
        B = B // (self.n_references + 1)

        ref = ref.reshape(B, self.n_references, C, H, W)
        ref = ref.permute(0, 2, 1, 3, 4)
        ref = ref.reshape(B, C, self.n_references * H * W)
        ref = ref.transpose(1, 2)

        tar = tar.view(B, C, H * W)
        sim = torch.matmul(ref, tar)
        if self.use_softmax:
            sim = F.softmax(sim, dim=1)

        return sim
    

    def predict(
            self,
            sim,
            y
    ):
        """Predict color distribution from similarity matrix and reference colors.
        Args:
            sim (torch.Tensor): Similarity matrix of shape (B1, n_references*H*W, H*W).
            y (torch.Tensor): Reference colors of shape (B2, C, H, W), where B2 = B1 * n_references.
        Returns:
            out (torch.Tensor): Color distribution of shape (B1, C, H, W).
        """
        B, C, H, W = y.shape
        
        assert B % self.n_references == 0
        B = B // self.n_references
        
        y = y.reshape(B, self.n_references, C, H, W)
        y = y.permute(0, 2, 1, 3, 4)    # must match the order of permutation in sim
        y = y.reshape(B, C, self.n_references * H * W)
        out = torch.matmul(y, sim)
        out = out.view(B, C, H, W)
        
        return out


    def load_checkpoint(self):
        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so loading checkpoint is not allowed.')
        
        file_path = parse_load_checkpoint_path(
            input_path=self.checkpoint_path,
            ext='pth',
        )

        if file_path == -1:
            return
        
        weight = torch.load(file_path)
        self.load_state_dict(weight)

        logger.info('Model loaded from {}.'.format(file_path))
    

    def save_checkpoint(self):
        if self.checkpoint_path is None:
            raise ValueError('Checkpoint argument is set to None, so saving checkpoint is not allowed.')
        
        self.checkpoint_path = parse_save_checkpoint_path(
            input_path=self.checkpoint_path,
            ext='pth',
        )
        
        torch.save(self.state_dict(), self.checkpoint_path)
        logger.info('Model saved at {}.'.format(self.checkpoint_path))


    

        


    


