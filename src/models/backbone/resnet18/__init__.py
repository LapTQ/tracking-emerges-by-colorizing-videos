from torch import nn


class BasicBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            stride=1,
            **kwargs
    ):
        super().__init__()

        # Biases are in the BN layers that follow.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.downsample:
            identity = self.downsample(identity)
        
        x = x + identity
        x = self.relu(x)

        return x


class CustomBackbone(nn.Module):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        stage_channels = kwargs.get('stage_channels', [64, 128, 256, 512])
        stage_strides = kwargs.get('stage_strides', [1, 2, 2, 2])

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, stage_channels[0], stage_strides[0])
        self.layer2 = self._make_layer(stage_channels[0], stage_channels[1], stage_strides[1])
        self.layer3 = self._make_layer(stage_channels[1], stage_channels[2], stage_strides[2])
        self.layer4 = self._make_layer(stage_channels[2], stage_channels[3], stage_strides[3])

    
    def _make_layer(
            self,
            in_channels,
            out_channels,
            stride,
    ):
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride),
            BasicBlock(out_channels, out_channels, 1)
        )
        

    def forward(
            self,
            x
    ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    

if __name__ == '__main__':
    from torchsummary import summary
    model = CustomBackbone()
    summary(model, (3, 256, 256))