from torch import nn
from torchvision.models import resnet18


class CustomBackbone(nn.Module):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()
        base_model = resnet18()
        self.layers = list(base_model.children())[:-2]
        for i, layer in enumerate(self.layers):
            self.add_module(f'layer_{i}', layer)

        return

    def forward(
            self,
            x
    ):
        for layer in self.layers:
            x = layer(x)
        return x