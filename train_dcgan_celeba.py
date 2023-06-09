from model.gan import Gan
from model.dcgan.discriminator import Discriminator
from model.dcgan.generator import Generator
from loaders import celeba

hyperparams = {
    "image_width": 32,
    "image_size": 32 * 32,
    "channels_image": 3,
    "z_dim": 100,
    "features_discriminator": 64,
    "features_generator": 64,
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 32
}

# MNIST dataset
loader = celeba(hyperparams["batch_size"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = Gan(discriminator, generator, hyperparams)
gan.train(loader)
