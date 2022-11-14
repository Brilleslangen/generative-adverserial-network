import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader

from gan import Gan
from discriminator import Discriminator
from generator import Generator

# Import GAN modules
import sys

sys.path.append('./../GAN')

# Latent vector size
ls_size = 100

# Training batch size
batch_size = 32

# Feature map size for generator and discriminator
fm_size = 64

# Number of channels in image
num_img_chan = 1

dataTransforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)]
)

# load the MNIST dataset and stack the training and testing data
# points so we have additional training data
print("[INFO] loading MNIST dataset...")
trainData = MNIST(root="data", train=True, download=True,
                  transform=dataTransforms)
testData = MNIST(root="data", train=False, download=True,
                 transform=dataTransforms)
data = torch.utils.data.ConcatDataset((trainData, testData))
# initialize our dataloader
dataloader = DataLoader(data, shuffle=True,
                        batch_size=batch_size)

# Initiate Discriminator and Discriminator
generator = Generator(ls_size, num_img_chan)
discriminator = Discriminator(depth=1)

# Initiate Generative Adverserial Network
# gan = Gan(generator, discriminator, dataloader, batch_size, ls_size)
gan = Gan(generator, discriminator, dataloader, batch_size, ls_size)
gan.train()
