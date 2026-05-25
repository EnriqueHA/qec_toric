import torch
import torch.nn as nn
import torch.nn.init as init

class PeriodicConv2d(nn.Module):
    """
    2D convulutional layer with periodic padding -> BatchNorm -> GELU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, padding_mode='circular')
        self.bn = nn.BatchNorm2d(out_channels)
        self.gelu = nn.GELU()

        # To initialize the weights
        init.kaiming_normal_(self.conv.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.gelu(x)
        return x

class ResNetBlock(nn.Module):
    """
    ResNet block of 3 periodic convulutional layers and the residual connection
    """

    def __init__(self, channels):
        super().__init__()
        self.conv1 = PeriodicConv2d(channels, channels)
        self.conv2 = PeriodicConv2d(channels, channels)
        self.conv3 = PeriodicConv2d(channels, channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)

        # + x is the residual connection
        return x3 + x

class ToricWideResNet(nn.Module):
    """
    Implementation of modified Wide ResNet architecture
    In: (Batch, Ch_in = 2, L, L)
    Out: (Batch, Ch_out = 16, L, L)
    """

    def __init__(self, in_channels = 2, init_channels = 128, middle_channels = 64, num_classes = 16):
        super().__init__()
        self.init_channels = init_channels
        self.middle_channels = middle_channels

        # Upsampling: channels: 2 -> init_channels
        self.upsample= PeriodicConv2d(in_channels, init_channels)

        # ResNet blocks
        self.block1 = ResNetBlock(init_channels)
        # Projection from init_channels to middle_channels, either upsampling or downsampling
        self.init_proj = PeriodicConv2d(init_channels, middle_channels)

        self.block2 = ResNetBlock(middle_channels)
        self.block3 = ResNetBlock(middle_channels)

        # Downsampling: channels: middle_channels -> num_classes
        self.downsample = PeriodicConv2d(middle_channels, num_classes)

    def forward(self, x):
        x = self.upsample(x)     # (B, 2, L, L)) -> (B, init_channels, L, L)
        x = self.block1(x)       # (B, init_channels, L, L)
        x = self.init_proj(x)    # (B, init_channels, L, L) -> (B, middle_channels, L, L)
        x = self.block2(x)       # (B, middle_channels, L, L)
        x = self.block3(x)       # (B, middle_channels, L, L)
        x = self.downsample(x)   # (B, middle_channels, L, L) -> (B, num_classes, L, L)

        return x 