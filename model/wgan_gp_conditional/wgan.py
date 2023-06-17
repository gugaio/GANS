import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter
from model.wgan_gp_conditional.penalty import gradient_penalty

class WGan:


    def __init__(self, discriminator, generator, hyperparams):
        self.hyperparams = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Discriminator
        self.discriminator = discriminator.to(self.device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"], betas=(0.0, 0.9))
        # Generator
        self.generator = generator.to(self.device)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"], betas=(0.0, 0.9))

        # Fixed noise for visualization
        self.fixed_noise = self.generator.sample_noise().to(self.device)
        self.writer_fake = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/fake")
        self.writer_real = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/real")

    def train(self, train_loader):
        self.discriminator.train()
        self.generator.train()
        epochs = self.hyperparams["num_epochs"]
        for epoch in range(epochs):
            for batch_idx, (real, labels) in enumerate(train_loader):
                batch_size = real.shape[0]
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} Batch Size {batch_size}", end='\r')
                lossD, lossG = self._train_batch(real, labels, batch_size)

                if batch_idx % 100 == 0 and batch_idx > 0:
                    self._log_batch_count(epochs, epoch, batch_idx, len(train_loader), batch_size)
                    self._log_losses(lossD, lossG)
                    self._write_images_at_tensorboard(real, labels, epoch)

    def _train_batch(self, batch_real, labels, batch_size):
        loss_disc = None
        loss_gen = None
        for n in range(self.hyperparams["n_critic"]):
            batch_real = batch_real.to(self.device)
            labels = labels.to(self.device)
            batch_fake = self.generate_fake_images(labels) 
            disc_real = self.discriminator(batch_real, labels).reshape(-1)
            disc_fake = self.discriminator(batch_fake, labels).reshape(-1)
            gp = gradient_penalty(self.discriminator, labels, batch_real, batch_fake, self.device)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) + self.hyperparams["lambda_gp"] * gp
            self.discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            self.discriminator_optimizer.step()

        batch_fake = self.generate_fake_images(labels) 
        gen_fake = self.discriminator(batch_fake, labels).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.generator.zero_grad()
        loss_gen.backward()
        self.generator_optimizer.step()
        return loss_disc, loss_gen

    def generate_fake_images(self, labels):
        noise = self.generator.sample_noise().to(self.device)
        fake = self.generator(noise, labels)
        return fake

    def _log_batch_count(self, total_epochs, current_epoch, batch_idx, total_batchs, batch_size):
        print(f"Epoch [{current_epoch}/{total_epochs}] Batch {batch_idx}/{total_batchs} Batch Size {batch_size}")
        

    def _log_losses(self, lossD, lossG) :
        print(f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}")


    def _write_images_at_tensorboard(self, real, labels, step):
        with torch.no_grad():
                fake = self.generate_fake_images(labels)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                self.writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                self.writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
