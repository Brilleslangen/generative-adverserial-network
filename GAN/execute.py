import opendatasets as od
import torchvision
import torch
import os
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from gan import Gan, display_images
from generator import Generator
from discriminator import Discriminator
from msg_discriminator import MsgDiscriminator
from msg_generator import MsgGenerator

# Latent vector size
ls_size = 100

# Training batch size
batch_size = 32

# Size of image (quadratic)
image_size = 64

# Scala for sizing feature maps for generator and discriminator layers
conv_scalar = 16

# Conv layers
num_conv_layers = 3

# Learning rate
lr = 0.0002

# Available datasets
datasets = ['mnist-numbers', 'abstract-art', 'bored-apes-yacht-club', 'celeba-dataset']


def select_dataset(set_name):
    ds_root = "./datasets"
    dataset = None

    # 1 channel datasets
    if set_name == datasets[0]:
        channels = 1
        transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
                                        transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        dataset = torchvision.datasets.MNIST(root=ds_root, train=True, download=True, transform=transform)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True), channels

    # 3 channel-datasets
    channels = 3
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size),
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


def run(ds_index, epochs, display_dataset=False, display_frequency=10, tf_model=None, msg=False):
    # Select dataset
    dataset_name = datasets[ds_index]
    dataloader, color_channels = select_dataset(dataset_name)

    if display_dataset:
        real_batch = next(iter(dataloader))
        display_images(real_batch[0])

    # Initiate Discriminator and Discriminator
    generator = Generator(ls_size, conv_scalar, color_channels, num_conv_layers)
    discriminator = Discriminator(conv_scalar, color_channels, num_conv_layers)

    if msg:
        generator = MsgGenerator(ls_size, conv_scalar, color_channels, num_conv_layers)
        discriminator = MsgDiscriminator(conv_scalar, color_channels, num_conv_layers)

    # Initiate Generative Adversarial Network
    gan = Gan(generator, discriminator, dataloader, dataset_name,
              display_frequency, batch_size, ls_size, epochs, lr, tf_model)
    gan.train()


def display_images_from_model(ds_index, model_name):
    # Select dataset
    dataset_name = datasets[ds_index]
    dataloader, color_channels = select_dataset(dataset_name)

    # Initiate generator and load model
    PATH = f'./models/{model_name}-model.pt'
    checkpoint = torch.load(PATH)

    generator = Generator(ls_size, conv_scalar, color_channels, num_conv_layers)
    generator.load_state_dict(checkpoint['generator_state'])

    # Print
    generator.eval()
    seed = torch.randn(256, 100, 1, 1)
    images = generator(seed)
    display_images(images)
    generator.train()


# Queue of GAN trainings
run(ds_index=0, epochs=20, display_frequency=1)
run(ds_index=1, epochs=200, display_frequency=1)
run(ds_index=2, epochs=200, display_frequency=1)
run(ds_index=3, epochs=50, display_frequency=1)
