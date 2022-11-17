import opendatasets as od
import torchvision
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch
from gan import Gan, display_images
from discriminator import Discriminator
from generator import Generator

# Latent vector size
ls_size = 100

# Training batch size
batch_size = 32

# Feature map size for generator and discriminator
fm_size = 64

# Conv layers
num_conv_layers = 3

# Available datasets
datasets = ['mnist-numbers', 'abstract-art', 'bored-apes-yacht-club', 'celeba-dataset']


def select_dataset(set_name):
    # Load data
    ds_root = "./datasets"
    dataset = None

    # 1 channel datasets
    if set_name == datasets[0]:
        channels = 1
        transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root=ds_root, train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), channels

    # 3 channel-datasets
    channels = 3
    transform = transforms.Compose([transforms.Resize(fm_size), transforms.CenterCrop(fm_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    if set_name == datasets[1]:
        directory = f'{ds_root}/abstract-art-gallery'
        if not os.path.isdir(directory):
            od.download("https://www.kaggle.com/datasets/bryanb/abstract-art-gallery", data_dir=ds_root)
        dataset = ImageFolder(root=f'{directory}/Abstract_gallery', transform=transform)

    elif set_name == datasets[2]:
        directory = f'{ds_root}/bored-apes-yacht-club'
        if not os.path.isdir(directory):
            od.download("https://www.kaggle.com/datasets/stanleyjzheng/bored-apes-yacht-club", data_dir=ds_root)
        dataset = ImageFolder(root=f'{directory}', transform=transform)
        
    elif set_name == datasets[3]:
        directory = f'{ds_root}/celeba-dataset'
        if not os.path.isdir(directory):
            od.download("https://www.kaggle.com/datasets/jessicali9530/celeba-dataset", data_dir=ds_root)
        dataset = ImageFolder(root=f'{directory}', transform=transform)
            

    return DataLoader(dataset, batch_size=batch_size, shuffle=True), channels


def run(ds_index, epochs, display_dataset=False, display_frequency=10):
    # Select dataset
    dataset_name = datasets[ds_index]
    dataloader, color_channels = select_dataset(dataset_name)

    if display_dataset:
        real_batch = next(iter(dataloader))
        display_images(real_batch[0])

    # Initiate Discriminator and Discriminator
    generator = Generator(ls_size, fm_size, color_channels, num_conv_layers)
    discriminator = Discriminator(fm_size, color_channels, num_conv_layers)
    gan = Gan(generator, discriminator, dataloader, dataset_name, display_frequency, batch_size, ls_size, epochs)
    
    # Initiate from trained models
    gan.load_model("test")
    
    # Set generator to eval, and display image from loaded model
    generator.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    generator.to(device)
    images = generator(torch.randn(256, 100, 1, 1, device=device))
    display_images(images, 'trained', f'abstract-art-epoch-{1}')

    # Initiate Generative Adversarial Network
    gan.train()


# Queue of GAN trainings
#run(ds_index=0, epochs=20, display_frequency=1)
run(ds_index=1, epochs=200, display_frequency=1)
#run(ds_index=2, epochs=200, display_frequency=1)
#run(ds_index=3, epochs=50, display_frequency=1)
