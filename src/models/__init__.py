from typing import Any
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

            return 
        

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
                **{
                    'module_name': self.backbone_name
                }
            )(
                **backbone_kwargs
            )

            self._product.head = head_factory(
                **{
                    'module_name': self.head_name
                }
            )(
                **head_kwargs
            )

            product = self._product
            self.reset()
            return product

    

        


    


