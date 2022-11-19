from collections import OrderedDict

import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_space_size, conv_scalar, num_image_chan, num_conv_layers):
        super().__init__()
        layers = OrderedDict()

        # Initialize number of filters for each layer
        feature_map_sizes = [conv_scalar * 2 ** (i + 1) for i in range(num_conv_layers, -1, -1)]

        # Initialize Latent Vector Space
        layers['L0-conv'] = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_space_size,
                               out_channels=feature_map_sizes[0],
                               kernel_size=4,
                               stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(feature_map_sizes[0]),
            nn.ReLU(True))

        # Apply Transposed Convolution for each Conv-module
        for i in range(len(feature_map_sizes) - 1):
            layers[f'L{i + 1}-conv'] = nn.Sequential(
                nn.ConvTranspose2d(in_channels=feature_map_sizes[i],
                                   out_channels=feature_map_sizes[i + 1],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.BatchNorm2d(feature_map_sizes[i + 1]),
                nn.ReLU(True))

        # Generate output image
        layers['Output'] = nn.Sequential(
            nn.ConvTranspose2d(in_channels=feature_map_sizes[-1],
                               out_channels=num_image_chan,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False),
            nn.Tanh())

        # Add modules to model
        self.model = nn.Sequential(layers)

    def forward(self, x):
        return self.model(x)
