from torch import nn
import cv2


class CustomTransform(nn.Module):
    """Custom Color Space Converter transform using OpenCV."""

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.code = kwargs['code']

        self.code = eval(self.code)

    
    def forward(
            self,
            x
    ):
        x = cv2.cvtColor(x, self.code)
        return x