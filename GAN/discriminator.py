# import the necessary packages
from torch.nn import ConvTranspose2d
from torch.nn import BatchNorm2d
from torch.nn import Conv2d
from torch.nn import Linear
from torch.nn import LeakyReLU
from torch.nn import ReLU
from torch.nn import Tanh
from torch.nn import Sigmoid
from torch import flatten
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2):
        super(Discriminator, self).__init__()
        # first set of CONV => RELU layers
        self.conv1 = Conv2d(in_channels=depth, out_channels=32,
                            kernel_size=4, stride=2, padding=1)
        self.leakyRelu1 = LeakyReLU(alpha, inplace=True)
        # second set of CONV => RELU layers
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                            stride=2, padding=1)
        self.leakyRelu2 = LeakyReLU(alpha, inplace=True)
        # first (and only) set of FC => RELU layers
        self.fc1 = Linear(in_features=3136, out_features=512)
        self.leakyRelu3 = LeakyReLU(alpha, inplace=True)
        # sigmoid layer outputting a single value
        self.fc2 = Linear(in_features=512, out_features=1)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        # pass the input through first set of CONV => RELU layers
        x = self.conv1(x)
        x = self.leakyRelu1(x)
        # pass the output from the previous layer through our second
        # set of CONV => RELU layers
        x = self.conv2(x)
        x = self.leakyRelu2(x)
        # flatten the output from the previous layer and pass it
        # through our first (and only) set of FC => RELU layers
        x = flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu3(x)
        # pass the output from the previous layer through our sigmoid
        # layer outputting a single value
        x = self.fc2(x)
        output = self.sigmoid(x)
        # return the output
        return output
