from torch import nn


class CustomTransform(nn.Module):
    """Custom transform to extract channels from an image"""

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.channels = kwargs['channels']

        if isinstance(self.channels, int):
            self.channels = [self.channels]

    
    def forward(
            self,
            x
    ):
        x = x[:, :, self.channels]
        return x