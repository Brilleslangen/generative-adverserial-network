import torch
import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, conv_scalar, channels, num_conv_layers):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.num_conv_layers = num_conv_layers

        # Layer 1: Convolution with leaky ReLU
        self.layers['L0-conv'] = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=conv_scalar,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(conv_scalar),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(1, num_conv_layers + 1):
            self.layers[f'L{i}-conv'] = nn.Sequential(
                nn.Conv2d(in_channels=conv_scalar * 2 ** (i - 1),
                          out_channels=conv_scalar * 2 ** i,
                          kernel_size=4,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(conv_scalar * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.layers[f'L{i}-upsample'] = nn.Conv2d(channels, conv_scalar * 2 ** (i - 1), kernel_size=1)
            self.layers[f'L{i}-downsample'] = nn.Conv2d(conv_scalar * 2 ** i, conv_scalar * 2 ** (i - 1), kernel_size=1)

        # Final convolution layer with sigmoid
        self.layers['evaluate'] = nn.Sequential(
            nn.Conv2d(in_channels=conv_scalar * 2 ** num_conv_layers,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid())

    def forward(self, inputs):
        x = self.layers[f'L0-conv'](inputs[0])
        for i in range(1, self.num_conv_layers + 1):
            inputs[i] = self.layers[f'L{i}-upsample'](inputs[i])
            x = torch.cat((inputs[i], x), dim=1)
            x = self.layers[f'L{i}-downsample'](x)
            x = self.layers[f'L{i}-conv'](x)
        x = self.layers['evaluate'](x)

        return x
