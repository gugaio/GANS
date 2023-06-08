from model.gan import Gan
from model.simplegan.discriminator import Discriminator
from model.simplegan.generator import Generator
from loaders import mnist_loader

hyperparams = {
    "image_size": 28 * 28,
    "z_dim": 64,
    "generator_hidden_size": 256,
    "discriminator_hidden_size": 128,
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 32
}

# MNIST dataset
loader = mnist_loader(hyperparams["batch_size"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = Gan(discriminator, generator, hyperparams)
gan.train(loader)
