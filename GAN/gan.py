import sys
import numpy as np
import torchvision.utils as tvutils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def weights_init(model):
    classname = model.__class__.__name__

    if classname.find("Conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)

    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)


def display_images(images):
    fig = plt.figure(figsize=(12, 12))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(tvutils.make_grid(images[:32], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.show()
    plt.close(fig)


class Gan:
    def __init__(self, generator, discriminator, dataloader, batch_size=32, latent_space_size=100):
        # Decide which device we want to run on
        device = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device)

        # Initiate
        self.device = device
        self.generator = generator.apply(weights_init).to(device=self.device)
        self.discriminator = discriminator.apply(weights_init).to(device=self.device)
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.latent_space_size = latent_space_size

    def init_train_conditions(self):
        # Hyper parameters
        lr = 0.0002
        epochs = 20
        episodes = len(self.dataloader)

        # Loss functions and optimizers
        loss = torch.nn.BCELoss().to(self.device)
        disc_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr,
                                      betas=(0.5, 0.999), weight_decay=0.0002 / epochs)
        gen_optim = torch.optim.Adam(self.generator.parameters(), lr=lr,
                                     betas=(0.5, 0.999), weight_decay=0.0002 / epochs)

        # Training variables
        benchmark_seed = torch.randn(256, 100, 1, 1, device=self.device)
        real = 1
        fake = 0

        print(f'episodes: {episodes}')

        return lr, epochs, episodes, loss, disc_optim, gen_optim, benchmark_seed, real, fake

    def train(self):
        # Initiate Training Conditions
        lr, epochs, episodes, loss, disc_optim, gen_optim, noise_seed, real, fake = self.init_train_conditions()

        # Training
        for epoch in range(epochs):

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
                disc_optim.step()

                # Generate fake samples and calculate loss
                self.generator.zero_grad()
                pred_fake = self.discriminator(fake_samples).view(-1)

                gen_batch_loss = loss(pred_fake, real_labels)
                gen_batch_loss.backward()

                gen_optim.step()

                gen_loss += gen_batch_loss
                disc_loss += disc_batch_loss

                if i % (round(episodes / 100)) == 0:
                    sys.stdout.write(f'\rEpoch {epoch + 1}: {((i + 1) * 100) // episodes}%')

            print(f'\n - Generator loss: {gen_loss / episodes},'
                  f' Discriminator loss: {disc_loss / episodes}')

            # Print image every other epoch
            self.generator.eval()
            images = self.generator(noise_seed)
            display_images(images)
            self.generator.train()                 

