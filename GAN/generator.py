# import the necessary packages
from torch import nn
from torch.nn import BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn import ReLU
from torch.nn import Tanh


class Generator(nn.Module):
    def __init__(self, inputDim=100, outputChannels=1):
        super(Generator, self).__init__()
        # first set of CONVT => RELU => BN
        self.ct1 = ConvTranspose2d(in_channels=inputDim,
                                   out_channels=128, kernel_size=4, stride=2, padding=0,
                                   bias=False)
        self.relu1 = ReLU()
        self.batchNorm1 = BatchNorm2d(128)
        # second set of CONVT => RELU => BN
        self.ct2 = ConvTranspose2d(in_channels=128, out_channels=64,
                                   kernel_size=3, stride=2, padding=1, bias=False)
        self.relu2 = ReLU()
        self.batchNorm2 = BatchNorm2d(64)
        # last set of CONVT => RELU => BN
        self.ct3 = ConvTranspose2d(in_channels=64, out_channels=32,
                                   kernel_size=4, stride=2, padding=1, bias=False)
        self.relu3 = ReLU()
        self.batchNorm3 = BatchNorm2d(32)
        # apply another upsample and transposed convolution, but
        # this time output the TANH activation
        self.ct4 = ConvTranspose2d(in_channels=32,
                                   out_channels=outputChannels, kernel_size=4, stride=2,
                                   padding=1, bias=False)
        self.tanh = Tanh()

    def forward(self, x):
        # pass the input through our first set of CONVT => RELU => BN
        # layers
        x = self.ct1(x)
        x = self.relu1(x)
        x = self.batchNorm1(x)
        # pass the output from previous layer through our second
        # CONVT => RELU => BN layer set
        x = self.ct2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)
        # pass the output from previous layer through our last set
        # of CONVT => RELU => BN layers
        x = self.ct3(x)
        x = self.relu3(x)
        x = self.batchNorm3(x)
        # pass the output from previous layer through CONVT2D => TANH
        # layers to get our output
        x = self.ct4(x)
        output = self.tanh(x)
        # return the output
        return output
