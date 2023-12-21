import torch
from torch import nn
from .backbone import backbone_factory
from .head import head_factory



def model_factory(
        **kwargs
):
    # parse kwargs
    module_name = kwargs

    model_builder = ColorizationModel.Builder(
        **module_name
    )

    return model_builder


class ColorizationModel(nn.Module):

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
            self._product = ColorizationModel()
        

        def __call__(
                self,
                **kwargs
        ):
            # parse kwargs
            backbone_kwargs = kwargs['backbone']
            head_kwargs = kwargs['head']

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
            sim (torch.Tensor): Similarity matrix of shape (B, H*W*3, H*W).
        """
        B, C, H, W = x.shape
        assert B % (self.n_references + 1) == 0
        ref = x[[i for i in range(B) if i % (self.n_references + 1) != self.n_references]]
        tar = x[[i for i in range(B) if i % (self.n_references + 1) == self.n_references]]
        B = B // (self.n_references + 1)

        ref = ref.view(B, self.n_references, C, H, W)
        ref = ref.permute(0, 2, 1, 3, 4).contiguous()
        ref = ref.view(B, C, self.n_references * H * W)
        ref = ref.transpose(1, 2)

        tar = tar.view(B, C, H * W)
        sim = torch.matmul(ref, tar)

        return sim
    

    def predict(
            self,
            sim,
            y
    ):
        """Predict color distribution from similarity matrix and reference colors.
        Args:
            sim (torch.Tensor): Similarity matrix of shape (B1, H*W*3, H*W).
            y (torch.Tensor): Reference colors of shape (B2, C, H, W), where B2 = B1 * n_references.
        Returns:
            out (torch.Tensor): Color distribution of shape (B1, C, H, W).
        """
        B, C, H, W = y.shape
        
        assert B % self.n_references == 0
        B = B // self.n_references
        
        y = y.view(B, self.n_references, C, H, W)
        y = y.permute(0, 2, 1, 3, 4).contiguous()   # must match the order of permutation in sim
        y = y.view(B, C, self.n_references * H * W)
        out = torch.matmul(y, sim)
        out = out.view(B, C, H, W)
        
        return out
    

        


    


