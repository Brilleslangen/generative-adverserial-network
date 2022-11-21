import torch.nn as nn


class MsgGenerator(nn.Module):
    def __init__(self, latent_space_size, conv_scalar, num_image_chan, num_conv_layers):
        super().__init__()
        self.layers = nn.ModuleDict()
        self.num_conv_layers = num_conv_layers

        # Initialize number of filters for each layer
        feature_map_sizes = [conv_scalar * 2 ** (i + 1) for i in range(num_conv_layers, -1, -1)]

        # Populate samples from Latent Vector
        self.layers['L0-conv'] = nn.Sequential(
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
            self.layers[f'L{i + 1}-conv'] = nn.Sequential(
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
            self.layers[f'Output-{i}'] = nn.Sequential(
                nn.ConvTranspose2d(in_channels=feature_map_sizes[i],
                                   out_channels=num_image_chan,
                                   kernel_size=4,
                                   stride=2,
                                   padding=1,
                                   bias=False),
                nn.Tanh())

    def forward(self, x):
        outputs = []

        for i in range(self.num_conv_layers + 1):
            x = self.layers[f'L{i}-conv'](x)
            o = self.layers[f'Output-{i}'](x)
            outputs.append(o)

        return list(reversed(outputs))
