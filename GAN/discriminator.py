import torch.nn as nn
from traitlets.config.application import OrderedDict


class Discriminator(nn.Module):
    def __init__(self, fm_size, channels):
        super().__init__()
        layers = OrderedDict()

        # Layer 1: Convolution with leaky ReLU
        layers['L1-conv'] = nn.Conv2d(channels, fm_size, 4, 2, 1, bias=False)
        layers['L1-leaky-relu'] = nn.LeakyReLU(0.2, inplace=True)

        # Layer 2: Convolution with batch normalization and leaky ReLU
        layers['L2-conv'] = nn.Conv2d(fm_size, fm_size * 2, 4, 2, 1, bias=False)
        layers['L2-batch'] = nn.BatchNorm2d(fm_size * 2)
        layers['L2-leaky-relu'] = nn.LeakyReLU(0.2, inplace=True)

        # Layer 3: Convolution with batch normalization and leaky ReLU
        layers['L3-conv'] = nn.Conv2d(fm_size * 2, fm_size * 4, 4, 2, 1, bias=False)
        layers['L3-batch'] = nn.BatchNorm2d(fm_size * 4)
        layers['L3-leaky-relu'] = nn.LeakyReLU(0.2, inplace=True)

        # Layer 4: Convolution with batch normalization and leaky ReLU
        layers['L4-conv'] = nn.Conv2d(fm_size * 4, fm_size * 8, 4, 2, 1, bias=False)
        layers['L4-batch'] = nn.BatchNorm2d(fm_size * 8)
        layers['L4-leaky-relu'] = nn.LeakyReLU(0.2, inplace=True)

        # Layer 5: Final convolution layer with sigmoid
        layers['L5-conv'] = nn.Conv2d(fm_size * 8, 1, 4, 1, 0, bias=False)
        layers['L5-sigmoid'] = nn.Sigmoid()

        self.model = nn.Sequential(layers)

    def forward(self, input):
        return self.model(input)
