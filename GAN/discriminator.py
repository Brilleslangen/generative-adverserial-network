import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, fm_size, channels, num_layers):
        super().__init__()
        layers = OrderedDict()

        # Layer 1: Convolution with leaky ReLU
        layers['L1-conv'] = nn.Conv2d(
            in_channels=channels, 
            out_channels=fm_size, 
            kernel_size=4,
            stride=2, 
            padding=1, 
            bias=False)
        layers['L1-norm'] = nn.BatchNorm2d(fm_size)
        layers['L1-leaky-relu'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
        for i in range(2, num_layers+2):
            layers[f'L{i}-conv'] = nn.Conv2d(
                in_channels=fm_size * 2 ** (i-2),
                out_channels=fm_size * 2 ** (i-1),
                kernel_size=4,
                stride=2,
                padding=1
            )
            layers[f'L{i}-norm'] = nn.BatchNorm2d(fm_size * 2 ** (i-1))
            layers[f'L{i}-leaky-relu'] = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Final convolution layer with sigmoid
        layers[f'L{num_layers+2}-conv'] = nn.Conv2d(
            in_channels=fm_size * 2 ** (num_layers),
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        layers[f'L{num_layers+2}-sigmoid'] = nn.Sigmoid()

        self.model = nn.Sequential(layers)

        print(self.model)
    def forward(self, input):
        return self.model(input)
