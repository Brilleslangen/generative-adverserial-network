import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, fm_size, channels, num_channels):
        super().__init__()
        modules = OrderedDict()

        # Layer 1: Convolution with leaky ReLU
        modules['L0-conv'] = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=fm_size,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(fm_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        
        for i in range(2, num_channels+2):
            modules[f'L{i}-conv'] = nn.Sequential(
                nn.Conv2d(in_channels=fm_size * 2 ** (i-2),
                          out_channels=fm_size * 2 ** (i-1),
                          kernel_size=4,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(fm_size * 2 ** (i-1)),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Final convolution layer with sigmoid
        modules[f'L{num_channels+2}-conv'] = nn.Sequential(
            nn.Conv2d(in_channels=fm_size * 2 ** num_channels,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid())

        self.model = nn.Sequential(modules)
        print(self)

    def forward(self, input):
        return self.model(input)
