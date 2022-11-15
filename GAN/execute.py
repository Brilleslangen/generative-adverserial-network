import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader

from gan import Gan
from discriminator import Discriminator
from generator import Generator

# Latent vector size
ls_size = 100

# Training batch size
batch_size = 32

# Feature map size for generator and discriminator
fm_size = 64

# Number of channels in image
num_img_chan = 1

# Conv layers
num_layers = 3

# Load data
datasets = ['mnist', 'abstract-art']
set_type = datasets[0]
dataset = ""

if set_type == datasets[0]:
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root=".", train=True, download=True, transform=transform)

elif set_type == datasets[1]:
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    dataset = ImageFolder(root="../abstract-art/Abstract_gallery", transform=transform)
    num_img_chan = 3

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set batch
real_batch = next(iter(dataloader))

# Display images
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(), (1, 2, 0)))

# Initiate Discriminator and Discriminator
generator = Generator(ls_size, fm_size, num_img_chan, num_layers)
discriminator = Discriminator(fm_size, num_img_chan, num_layers)

# Initiate Generative Adversarial Network
# gan = Gan(generator, discriminator, dataloader, batch_size, ls_size)
gan = Gan(generator, discriminator, dataloader, batch_size, ls_size)
gan.train()
