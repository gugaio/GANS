from model.discriminator import Discriminator
from model.generator import Generator
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class Gan:


    def __init__(self, hyperparams):
        self.hyperparams = hyperparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Discriminator
        self.discriminator = Discriminator(hyperparams).to(self.device)
        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=hyperparams["learning_rate"])
        # Generator
        self.generator = Generator(hyperparams).to(self.device)
        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=hyperparams["learning_rate"])    
        # Loss function
        self.criterion = nn.BCELoss()
        # Fixed noise for visualization
        self.fixed_noise = torch.randn((hyperparams["batch_size"], hyperparams["z_dim"])).to(self.device) 
        self.writer_fake = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/fake")
        self.writer_real = SummaryWriter(f"dataset/tensorboard/GAN_MNIST/real")

    def train(self, train_loader):
        epochs = self.hyperparams["num_epochs"]
        for epoch in range(epochs):
            for batch_idx, (real, _) in enumerate(train_loader):
                batch_size = real.shape[0]
                print(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} Batch Size {batch_size}", end='\r')
                lossD, lossG = self._train_batch(real, batch_size)

                if batch_idx == 0:
                    self._log_batch_count(epochs, epoch, batch_idx, len(train_loader), batch_size)
                    self._log_losses(lossD, lossG)
                    self._write_images_at_tensorboard(real, epoch)


    def _log_batch_count(self, total_epochs, current_epoch, batch_idx, total_batchs, batch_size):
        print(f"Epoch [{current_epoch}/{total_epochs}] Batch {batch_idx}/{total_batchs} Batch Size {batch_size}")
        
    def _log_losses(self, lossD, lossG) :
        print(f"Loss D: {lossD:.4f}, loss G: {lossG:.4f}")

    def _train_batch(self, batch_real_2D, batch_size):  
        batch_real_1D = self.reshape_real_images_as_1D(batch_real_2D) 
        batch_fake_1D = self.generate_fake_images_as_1D(batch_size) 
        lossD_real = self.train_discriminator_with_real_images(batch_real_1D)
        lossD_fake = self.train_discriminator_with_fake_images(batch_fake_1D)
        self.update_discriminator_weights(lossD_real, lossD_fake)
        self.update_generator_weights(batch_fake_1D)
        return lossD_real, lossD_fake
    
    def _write_images_at_tensorboard(self, real, step):
        with torch.no_grad():
                fake = self.generator(self.fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                self.writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                self.writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

    def reshape_real_images_as_1D(self, batch_real_2D):
        return self.reshape_2D_to_1D(batch_real_2D).to(self.device)


    def generate_fake_images_as_1D(self, batch_size):
        z_dim = self.generator.z_dim
        noise = torch.randn(batch_size, z_dim).to(self.device)
        fake = self.generator(noise)
        return fake


    def update_discriminator_weights(self, lossD_real, lossD_fake):
        lossD = (lossD_real + lossD_fake) / 2        
        self.discriminator.zero_grad()
        lossD.backward(retain_graph=True)
        self.discriminator_optimizer.step()


    def update_generator_weights(self, batch_fake_1D):
        lossG_fake = self.train_discriminator_with_fake_images(batch_fake_1D)
        self.generator.zero_grad()
        lossG_fake.backward()
        self.generator_optimizer.step()


    def reshape_2D_to_1D(self, real):
        return real.view(-1, 784)


    def train_discriminator_with_real_images(self, real):        
        disc_real = self.discriminator(real).view(-1)
        ### Discriminator Loss: max log(D(real)) + log(1 - D(G(z)))
        return self.criterion(disc_real, torch.ones_like(disc_real))


    def train_discriminator_with_fake_images(self, fake):
        disc_fake = self.discriminator(fake).view(-1)
        ### Discriminator Loss: max log(D(real)) + log(1 - D(G(z)))
        return self.criterion(disc_fake, torch.zeros_like(disc_fake))
