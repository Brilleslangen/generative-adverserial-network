import torch.nn as nn
from traitlets.config.application import OrderedDict


class Generator(nn.Module):
    def __init__(self, latent_space_size, image_size, num_image_chan, num_layers):
        super().__init__()
        modules = OrderedDict()

        # Initialize number of filters for each layer
        feature_map_sizes = [image_size * 2 ** (i + 1) for i in range(num_layers, -2, -1)]

        # Initialize Latent Vector Space
        modules['LS-conv'] = nn.ConvTranspose2d(
            in_channels=latent_space_size,
            out_channels=feature_map_sizes[0],
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False)
        modules['LS-norm'] = nn.BatchNorm2d(feature_map_sizes[0])
        modules['LS-relu'] = nn.ReLU(True)
        print('initialized')

        # Apply Transposed Convolution for each Conv-module
        for i in range(len(feature_map_sizes) - 2):
            print(f"in: {feature_map_sizes[i]}, out: {feature_map_sizes[i + 1]}")
            modules[f'L{i}-conv'] = nn.ConvTranspose2d(
                in_channels=feature_map_sizes[i],
                out_channels=feature_map_sizes[i + 1],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False)
            modules[f'L{i}-norm'] = nn.BatchNorm2d(feature_map_sizes[i + 1])
            modules[f'L{i}-relu'] = nn.ReLU(True)

        # Generate output image
        modules['Output'] = nn.ConvTranspose2d(
            in_channels=feature_map_sizes[-2],
            out_channels=num_image_chan,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False)
        modules['Output-activation'] = nn.Tanh()

        # Add modules to model
        self.model = nn.Sequential(modules)

    def forward(self, x):
        return self.model(x)
