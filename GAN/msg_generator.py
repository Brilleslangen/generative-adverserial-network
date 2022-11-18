import torch.nn as nn
from traitlets.config.application import OrderedDict


class Generator(nn.Module):
    def __init__(self, latent_space_size, image_size, num_image_chan, num_conv_layers):
        super(Generator, self).__init__()
        self.modules = OrderedDict()
        self.num_conv_layers = num_conv_layers

        # Initialize number of filters for each layer
        feature_map_sizes = [image_size * 2 ** (i + 1) for i in range(num_conv_layers, -1, -1)]

        # Initialize Latent Vector Space
        self.modules['L0-conv'] = nn.Sequential(
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
            # print(f"in: {feature_map_sizes[i]}, out: {feature_map_sizes[i + 1]}") # Debugging
            self.modules[f'L{i + 1}-conv'] = nn.Sequential(
                nn.ConvTranspose2d(in_channels=feature_map_sizes[i],
                                   out_channels=feature_map_sizes[i + 1],
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.BatchNorm2d(feature_map_sizes[i + 1]),
                nn.ReLU(True))

        # Render Output-layers
        for i in range(len(feature_map_sizes)):
            # Generate output image
            self.modules[f'Output-{i}'] = nn.Sequential(
                nn.ConvTranspose2d(in_channels=feature_map_sizes[i],
                                   out_channels=num_image_chan,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.Tanh())

        # Add modules to model
        self.model = nn.Sequential(self.modules)

    def forward(self, x):
        outputs = []

        for i in range(self.num_conv_layers + 1):
            x = self.modules[f'L{i}-conv'](x)
            o = self.modules[f'Output-{i}'](x)
            outputs.append(o)

        # for output in outputs:
        #    print(len(output), len(output[0]), len(output[0][0]), len(output[0][0][0]))

        return outputs[-1]
