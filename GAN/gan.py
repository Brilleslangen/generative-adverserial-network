import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Gan():
    def __init__(self, generator, discriminator, dataloader, batch_size=32, latent_space_size=100):
        # Decide which device we want to run on
        device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.device("cuda")

        # Initiate
        self.device = device
        self.generator = generator.to(device=self.device)
        self.discriminator = discriminator.to(device=self.device)
        self.dataloader = dataloader
        self.batch_size = batch_size
        self.latent_space_size = latent_space_size

    def init_train_conditions(self):
        # Hyperparameters
        lr = 0.0002
        num_epochs = 200
        loss = torch.nn.BCELoss().to(self.device)
        batches = len(self.dataloader)
        print(f'batches: {batches}')

        self.discriminator.train()
        self.generator.train()

        return lr, num_epochs, loss, batches

    def train(self):
        # Hyperparameters
        lr, num_epochs, loss, batches = self.init_train_conditions()

        discriminator_optim = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
        generator_optim = torch.optim.Adam(self.generator.parameters(), lr=lr)

        for i, (real_img, _) in enumerate(self.dataloader):
            real_img = real_img.to(self.device)
            label_size = real_img.size(0)

            # Labels for real and fake images
            real_imgs_label = torch.full([label_size, 1, 1, 1], 1.0, dtype=real_img.dtype, device=self.device)
            fake_imgs_label = torch.full([label_size, 1, 1, 1], 0.0, dtype=real_img.dtype, device=self.device)

            # Random noise vector
            vec = torch.randn([label_size, 100, 1, 1], device=self.device)

            # ---- TRAINING DISCRIMINATOR ----
            # Reset discriminator gradient
            self.discriminator.zero_grad()

            # Predicate using discriminator on real image
            pred = self.discriminator(real_img)
            discriminator_loss_real = loss(pred, real_imgs_label)
            discriminator_loss_real.backward()
            discriminator_real = pred.mean().item()

            # Predicate using discriminator on a fake image from the generator
            fake_img = self.generator(vec)
            pred = self.discriminator(fake_img.detach())
            discriminator_loss_fake = loss(pred, fake_imgs_label)
            discriminator_loss_fake.backward()
            discriminator_fake1 = pred.mean().item()

            # Add losses to complete loss equation, update weights
            discriminator_loss = discriminator_loss_real + discriminator_loss_fake
            discriminator_optim.step()

            # ---- TRAINING GENERATOR ----
            # Reset generator gradient
            self.generator.zero_grad()

            # Discriminator predicates fake image
            pred = self.discriminator(fake_img)
            generator_loss = loss(pred, real_imgs_label)

            # Update weights
            generator_loss.backward()
            generator_optim.step()
            discriminator_fake2 = pred.mean().item()

            if (i + 1) % 100 == 0 or (i + 1) == batches:
                print(f"Train stage: adversarial "
                      f"D Loss: {discriminator_loss.item():.6f} G Loss: {generator_loss.item():.6f} "
                      f"D(Real): {discriminator_real:.6f} D(Fake1)/D(Fake2): {discriminator_fake1:.6f}/{discriminator_fake2:.6f}.")
                fig = plt.figure()
                sample_img = self.generator(vec)
                for j in range(sample_img.size(0)):
                    plt.imshow(sample_img.detach()[j, 0, :, :], cmap='gray_r', interpolation='none')
                    plt.title("Generated data")
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis('off')
                plt.show()
