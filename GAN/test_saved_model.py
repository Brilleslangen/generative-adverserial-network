import torch
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from matplotlib import pyplot as plt
from torchvision.datasets import MNIST, ImageFolder
from torch.utils.data import DataLoader

from gan import Gan, display_images
from discriminator import Discriminator
from generator import Generator

# Latent vector size
ls_size = 100

# Training batch size
batch_size = 32

# Feature map size for generator and discriminator
fm_size = 64

# Number of channels in image
num_img_chan = 3

# Conv layers
num_conv_layers = 3

# Load data
datasets = ['mnist', 'abstract-art']
set_type = datasets[1]
dataset = ""

if set_type == datasets[0]:
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = torchvision.datasets.MNIST(root="./datasets", train=True, download=True, transform=transform)

elif set_type == datasets[1]:
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    dataset = ImageFolder(root="./datasets/abstract-art-gallery/Abstract_gallery", transform=transform)
    num_img_chan = 3

elif set_type == datasets[2]:
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    dataset = ImageFolder(root="./bayc", transform=transform)
    num_img_chan = 3

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Set batch
real_batch = next(iter(dataloader))

# Display images
display_images(real_batch[0])

# Initiate Discriminator and Discriminator
generator = Generator(ls_size, fm_size, num_img_chan, num_conv_layers)
discriminator = Discriminator(fm_size, num_img_chan, num_conv_layers)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=1,
                                      betas=(0.5, 0.999), weight_decay=0.0002 / 5)
gen_optim = torch.optim.Adam(generator.parameters(), lr=1,
                                     betas=(0.5, 0.999), weight_decay=0.0002 / 5)

# Generate pic from loaded models
checkpoint = torch.load("model.pt")

generator.load_state_dict(checkpoint['generator_state'])
discriminator.load_state_dict(checkpoint['discriminator_state'])
disc_optim.load_state_dict(checkpoint['disc_optim'])
gen_optim.load_state_dict(checkpoint['gen_optim'])

generator.eval()
discriminator.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(device)

generator.to(device)
discriminator.to(device)

images = generator(torch.randn(256, 100, 1, 1, device=device))
display_images(images, 'trained', f'abstract-art-epoch-{1}')

