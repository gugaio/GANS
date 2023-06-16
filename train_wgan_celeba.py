from model.wgan.wgan import WGan
from model.wgan.discriminator import Discriminator
from model.wgan.generator import Generator
from loaders import celeba

hyperparams = {
    "image_width": 64,
    "image_size": 64 * 64,
    "channels_image": 3,
    "channels_noise": 100,
    "features_discriminator": 64,
    "features_generator": 64,
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "clip_value": 0.01,
    "n_critic": 5
}

# MNIST dataset
loader = celeba(hyperparams["batch_size"], image_size=hyperparams["image_width"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = WGan(discriminator, generator, hyperparams)
gan.train(loader)
