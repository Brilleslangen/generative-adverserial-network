import sys
import numpy as np
import torchvision.utils as tvutils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import os


def weights_init(model):
    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def display_images(images, directory=None, filename=None):
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title(f'{filename}')
    plt.imshow(np.transpose(tvutils.make_grid(images[:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    if filename is None or directory is None:
        plt.show()
    else:
        path = f'../results/{directory}'
        initiate_directory('/results')
        initiate_directory(path)
        plt.savefig(f'{path}/{filename}.png', bbox_inches='tight')
    plt.close(fig)


def initiate_directory(path):
    if not os.path.isdir(path):
        os.mkdir(path)


class Gan:
    def __init__(self, generator, discriminator, dataloader, ds_name, display_frequency,
                 batch_size=32, latent_space_size=100, epochs=200, learning_rate=0.0002):
        # Decide which device we want to run on
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Initiate
        self.device = device
        self.generator = generator.apply(weights_init).to(device=self.device)
        self.discriminator = discriminator.apply(weights_init).to(device=self.device)
        self.dataloader = dataloader
        self.ds_name = ds_name
        self.display_frequency = display_frequency
        self.batch_size = batch_size
        self.latent_space_size = latent_space_size
        self.epochs = epochs
        self.disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate,
                                           betas=(0.5, 0.999), weight_decay=0.0002 / self.epochs)
        self.gen_optim = torch.optim.Adam(self.generator.parameters(), lr=learning_rate,
                                          betas=(0.5, 0.999), weight_decay=0.0002 / self.epochs)

    def init_train_conditions(self):
        # Hyper parameters
        episodes = len(self.dataloader)

        # Loss function
        loss = torch.nn.BCELoss().to(self.device)

        # Training variables
        benchmark_seed = torch.randn(256, 100, 1, 1, device=self.device)
        real = 1
        fake = 0

        return episodes, loss, benchmark_seed, real, fake

    def save_model(self, model_name):
        path = '../models'
        initiate_directory(path)

        # Save model
        filename = f'{model_name}-model.pt'

        torch.save({
            'discriminator_state': self.discriminator.state_dict(),
            'generator_state': self.generator.state_dict(),
            'disc_optim': self.disc_optim.state_dict(),
            'gen_optim': self.gen_optim.state_dict(),
        }, f'{path}/{filename}')

    def load_model(self, model_name):
        PATH = f'../models/{model_name}-model.pt'
        checkpoint = torch.load(PATH)

        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.disc_optim.load_state_dict(checkpoint['disc_optim'])
        self.gen_optim.load_state_dict(checkpoint['gen_optim'])

    def train(self):
        # Initiate Training Conditions
        episodes, loss, benchmark_seed, real, fake = self.init_train_conditions()

        print('-------------------------------\n'
              f'Dataset: {self.ds_name}\n'
              f'Epochs: {self.epochs}\n'
              f'Episodes per epoch: {episodes}\n'
              '-------------------------------')

        # Training
        for epoch in range(self.epochs):
            gen_loss = 0
            disc_loss = 0

            for i, batch in enumerate(self.dataloader):
                # Get batch of real images
                self.discriminator.zero_grad()
                real_samples = batch[0].to(self.device)
                sample_size = len(real_samples)

                # Create real and fake labels
                real_labels = torch.full((sample_size,), real, dtype=torch.float, device=self.device)
                fake_labels = torch.full((sample_size,), fake, dtype=torch.float, device=self.device)

                # Discriminate real samples and calculate loss
                pred_real = self.discriminator(real_samples).view(-1)
                error_real = loss(pred_real, real_labels)
                error_real.backward()

                # Discriminate fake samples and calculate loss
                latent_vector = torch.randn(sample_size, self.latent_space_size, 1, 1, device=self.device)
                fake_samples = self.generator(latent_vector)

                pred_fake = self.discriminator(fake_samples.detach()).view(-1)
                error_fake = loss(pred_fake, fake_labels)
                error_fake.backward()

                # Calculate total discriminator loss
                disc_batch_loss = error_real + error_fake
                self.disc_optim.step()

                # Generate fake samples and calculate loss
                self.generator.zero_grad()
                pred_fake = self.discriminator(fake_samples).view(-1)

                gen_batch_loss = loss(pred_fake, real_labels)
                gen_batch_loss.backward()

                self.gen_optim.step()

                gen_loss += gen_batch_loss
                disc_loss += disc_batch_loss

                if i % (math.ceil(episodes / 100)) == 0:
                    sys.stdout.write(f'\rEpoch {epoch + 1}: {((i + 1) * 100) // episodes}%')

            print(f'\nGenerator loss: {gen_loss / episodes}\n'
                  f'Discriminator loss: {disc_loss / episodes}\n'
                  '-------------------------------')

            # Save images
            if (epoch + 1) % self.display_frequency == 0:
                self.generator.eval()
                images = self.generator(benchmark_seed)
                display_images(images, self.ds_name, f'{self.ds_name}-epoch-{epoch + 1}')
                self.generator.train()
