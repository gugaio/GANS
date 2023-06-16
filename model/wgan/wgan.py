import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.tensorboard.writer import SummaryWriter


class WGan:


    def __init__(self, discriminator, generator, hyperparams):
        self.hyperparams = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Discriminator
        self.discriminator = discriminator.to(self.device)
        self.discriminator_optimizer = optim.RMSprop(self.discriminator.parameters(), lr=hyperparams["learning_rate"])
        # Generator
        self.generator = generator.to(self.device)
        self.generator_optimizer = optim.RMSprop(self.generator.parameters(), lr=hyperparams["learning_rate"])    

        # Fixed noise for visualization
        self.fixed_noise = self.generator.sample_noise().to(self.device)
        self.writer_fake = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/fake")
        self.writer_real = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/real")

    def train(self, train_loader):
        self.discriminator.train()
        self.generator.train()
        epochs = self.hyperparams["num_epochs"]
        for epoch in range(epochs):
            for batch_idx, (real, _) in enumerate(train_loader):
                batch_size = real.shape[0]
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} Batch Size {batch_size}", end='\r')
                lossD, lossG = self._train_batch(real, batch_size)

                if batch_idx % 100 == 0 and batch_idx > 0:
                    self._log_batch_count(epochs, epoch, batch_idx, len(train_loader), batch_size)
                    self._log_losses(lossD, lossG)
                    self._write_images_at_tensorboard(real, epoch)

    def _train_batch(self, batch_real, batch_size):
        loss_disc = None
        loss_gen = None
        for n in range(self.hyperparams["n_critic"]):
            batch_real = batch_real.to(self.device)
            batch_fake = self.generate_fake_images(batch_size) 
            disc_real = self.discriminator(batch_real).reshape(-1)
            disc_fake = self.discriminator(batch_fake).reshape(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake))
            self.discriminator.zero_grad()
            loss_disc.backward(retain_graph=True)
            self.discriminator_optimizer.step()

            for p in self.discriminator.parameters():
                p.data.clamp_(-self.hyperparams["clip_value"], self.hyperparams["clip_value"])
        batch_fake = self.generate_fake_images(batch_size) 
        gen_fake = self.discriminator(batch_fake).reshape(-1)
        loss_gen = -torch.mean(gen_fake)
        self.generator.zero_grad()
        loss_gen.backward()
        self.generator_optimizer.step()
        return loss_disc, loss_gen

    def generate_fake_images(self, batch_size):
        noise = self.generator.sample_noise().to(self.device)
        fake = self.generator(noise)
        return fake

    def _log_batch_count(self, total_epochs, current_epoch, batch_idx, total_batchs, batch_size):
        print(f"Epoch [{current_epoch}/{total_epochs}] Batch {batch_idx}/{total_batchs} Batch Size {batch_size}")
        

    def _log_losses(self, lossD, lossG) :
        print(f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}")


    def _write_images_at_tensorboard(self, real, step):
        with torch.no_grad():
                fake = self.generator(self.fixed_noise)

                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                self.writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                self.writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
