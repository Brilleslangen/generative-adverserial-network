import torch
import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, fm_size, channels, num_conv_layers):
        super().__init__()
        self.modules = OrderedDict()
        self.num_conv_layers = num_conv_layers

        # Layer 1: Convolution with leaky ReLU
        self.modules['L0-conv'] = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=fm_size,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(fm_size),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))

        for i in range(1, num_conv_layers + 1):
            self.modules[f'L{i}-conv'] = nn.Sequential(
                nn.Conv2d(in_channels=fm_size * 2 ** (i - 1),
                          out_channels=fm_size * 2 ** i,
                          kernel_size=4,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(fm_size * 2 ** i),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.modules[f'L{i}-upsample'] = nn.Conv2d(1, fm_size * 2 ** (i - 1), kernel_size=1)
            self.modules[f'L{i}-downsample'] = nn.Conv2d(fm_size * 2 ** i, fm_size * 2 ** (i - 1), kernel_size=1)

        # Final convolution layer with sigmoid
        self.modules['evaluate'] = nn.Sequential(
            nn.Conv2d(in_channels=fm_size * 2 ** num_conv_layers,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid())

        self.model = nn.Sequential(self.modules)


    def forward(self, inputs):
        x = self.modules[f'L0-conv'](inputs[0])
        for i in range(1, self.num_conv_layers + 1):
            inputs[i] = self.modules[f'L{i}-upsample'](inputs[i])
            x = torch.cat((inputs[i], x), dim=1)
            x = self.modules[f'L{i}-downsample'](x)
            x = self.modules[f'L{i}-conv'](x)
        x = self.modules['evaluate'](x)

        return x
