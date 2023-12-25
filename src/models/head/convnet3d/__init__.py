from torch import nn
import torch.nn.functional as F


class BasicBlock(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dilation=1,
            **kwargs
    ):
        super().__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 3, 3), padding=(0, dilation, dilation), dilation=(1, dilation, dilation), bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
    

    def forward(
            self,
            x
    ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x)

        return x


class CustomHead(nn.Module):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__()

        # parse kwargs
        self.n_references = kwargs['n_references']
        in_channels = kwargs.get('in_channels', 256)
        mid_channels = kwargs.get('mid_channels', 256)
        out_channels = kwargs.get('out_channels', 64)
        dilations = kwargs.get('dilations', [1, 2, 4, 8, 16])

        # # add 2 channels for spatial information
        # in_channels += 2

        self.layer1 = BasicBlock(in_channels, mid_channels, dilations[0])
        self.layer2 = BasicBlock(mid_channels, mid_channels, dilations[1])
        self.layer3 = BasicBlock(mid_channels, mid_channels, dilations[2])
        self.layer4 = BasicBlock(mid_channels, mid_channels, dilations[3])
        self.layer5 = BasicBlock(mid_channels, mid_channels, dilations[4])
        self.conv = nn.Conv3d(mid_channels, out_channels, kernel_size=1)


    def forward(
            self,
            x
    ):
        B, C, H, W = x.shape
        x = x.reshape(-1, self.n_references + 1, C, H, W)
        # spatial_info = torch.stack(
        #     torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        # )
        # spatial_info = 2 * spatial_info / torch.tensor([[[H - 1]], [[W - 1]]]) - 1
        x = x.permute(0, 2, 1, 3, 4)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.conv(x)

        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(B, -1, H, W)

        return x