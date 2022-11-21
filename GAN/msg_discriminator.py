import torch
import torch.nn as nn


class MsgDiscriminator(nn.Module):
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
            in_size = conv_scalar * 2 ** (i - 1)
            injection_size = in_size // 2  # One third injected images for each layer
            out_size = conv_scalar * 2 ** i

            # Convolute generator output image to generate preferred injection proportion
            self.layers[f'L{i}-injection-proportioning'] = nn.ConvTranspose2d(channels, injection_size, kernel_size=1)

            self.layers[f'L{i}-combine'] = nn.Conv2d(in_channels=in_size + injection_size,
                                                     out_channels=out_size,
                                                     kernel_size=1,
                                                     stride=1,
                                                     padding=0)

            self.layers[f'L{i}-conv'] = nn.Sequential(
                nn.Conv2d(in_channels=out_size,
                          out_channels=out_size * 2,
                          kernel_size=4,
                          stride=2,
                          padding=1),
                nn.BatchNorm2d(out_size * 2),
                nn.LeakyReLU(negative_slope=0.2, inplace=True))

            self.layers[f'L{i}-downscale'] = nn.Conv2d(out_size * 2, out_size, kernel_size=1)

        # Final convolution layer with sigmoid
        self.layers['evaluate'] = nn.Sequential(
            nn.Conv2d(in_channels=conv_scalar * 2 ** num_conv_layers,
                      out_channels=1,
                      kernel_size=4,
                      stride=1,
                      padding=0,
                      bias=False),
            nn.Sigmoid())
        print(self)

    def forward(self, inputs):
        for s in inputs:
            print(s.shape)

        x = self.layers[f'L0-conv'](inputs[0])
        for i in range(1, self.num_conv_layers + 1):
            inputs[i] = self.layers[f'L{i}-injection-proportioning'](inputs[i])
            x = torch.cat((inputs[i], x), dim=1)
            x = self.layers[f'L{i}-combine'](x)
            print('combinee', x.shape)
            x = self.layers[f'L{i}-conv'](x)
            print('conv', x.shape)
            x = self.layers[f'L{i}-downscale'](x)

        return self.layers['evaluate'](x)
