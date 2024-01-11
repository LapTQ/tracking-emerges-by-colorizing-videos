from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class BasicBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            **kwargs
    ):
        super().__init__()

        activation = kwargs['activation']

        self.activation = getattr(F, activation)

        # Biases are in the BN layers that follow.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(
            self,
            x
    ):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample(identity)
        
        x = x + identity
        x = self.activation(x)

        return x


class CustomBackbone(nn.Module):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        in_channels = kwargs['in_channels']
        mid_channels = kwargs.get('mid_channels', [128, 256, 256, 512])
        mid_strides = kwargs.get('mid_strides', [1, 2, 1, 2])
        activation = kwargs['activation']

        self.activation = getattr(F, activation)

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(128)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.layer1 = self._make_layer(128, mid_channels[0], mid_strides[0], activation)
        self.layer2 = self._make_layer(mid_channels[0], mid_channels[1], mid_strides[1], activation)
        self.layer3 = self._make_layer(mid_channels[1], mid_channels[2], mid_strides[2], activation)
        self.layer4 = self._make_layer(mid_channels[2], mid_channels[3], mid_strides[3], activation)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity=activation if activation in ['relu', 'leaky_relu'] else 'leaky_relu'
                )
            elif isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    
    def _make_layer(
            self,
            in_channels,
            out_channels,
            stride,
            activation,
    ):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride, activation=activation),
            BasicBlock(out_channels, out_channels, 1, activation=activation)
        )
        

    def forward(
            self,
            x
    ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    

if __name__ == '__main__':
    model = CustomBackbone(
        in_channels=1,
        mid_channels=[128, 256, 256, 512],
        mid_strides=[1, 2, 1, 2],
        activation='leaky_relu'
    )
    summary(model, (1, 256, 256))