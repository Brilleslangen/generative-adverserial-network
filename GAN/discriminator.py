import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, fm_size, channels, num_channels):
        super().__init__()
        modules = OrderedDict()

        # Layer 1: Convolution with leaky ReLU
        modules['L1-conv'] = nn.Conv2d(
            in_channels=channels, 
            out_channels=fm_size, 
            kernel_size=4,
            stride=2, 
            padding=1, 
            bias=False)
        modules['L1-norm'] = nn.BatchNorm2d(fm_size)
        modules['L1-leaky-relu'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        for i in range(2, num_channels+2):
            modules[f'L{i}-conv'] = nn.Conv2d(
                in_channels=fm_size * 2 ** (i-2),
                out_channels=fm_size * 2 ** (i-1),
                kernel_size=4,
                stride=2,
                padding=1
            )
            modules[f'L{i}-norm'] = nn.BatchNorm2d(fm_size * 2 ** (i-1))
            modules[f'L{i}-leaky-relu'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Final convolution layer with sigmoid
        modules[f'L{num_channels+2}-conv'] = nn.Conv2d(
            in_channels=fm_size * 2 ** num_channels,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        modules[f'L{num_channels+2}-sigmoid'] = nn.Sigmoid()

        self.model = nn.Sequential(modules)

    def forward(self, input):
        return self.model(input)
