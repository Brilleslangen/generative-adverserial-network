from collections import OrderedDict

import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, conv_scalar, channels, num_conv_layers):
        super().__init__()
        layers = OrderedDict()

        # Layer 1: Convolution with leaky ReLU
        layers['L0-conv'] = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=conv_scalar,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(conv_scalar),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(1, num_conv_layers + 1):
            layers[f'L{i}-conv'] = nn.Sequential(
                nn.Conv2d(in_channels=conv_scalar * 2 ** (i - 1),
                          out_channels=conv_scalar * 2 ** i,
                          kernel_size=4,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(conv_scalar * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

        # Final convolution layer with sigmoid
        layers[f'evaluate'] = nn.Sequential(
            nn.Conv2d(in_channels=conv_scalar * 2 ** num_conv_layers,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid())

        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)
