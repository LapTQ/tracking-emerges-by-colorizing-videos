from torch import nn
import cv2


class CustomTransform(nn.Module):
    """Custom Resize transform using OpenCV."""

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.size = kwargs['size']

        self.imH, self.imW = self.size

    
    def forward(
            self,
            x
    ):
        x = cv2.resize(x, (self.imW, self.imH))
        return x

    