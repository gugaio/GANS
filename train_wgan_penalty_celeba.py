from model.wgan_gp.wgan import WGan
from model.wgan_gp.discriminator import Discriminator
from model.wgan_gp.generator import Generator
from loaders import celeba

hyperparams = {
    "image_width": 64,
    "image_size": 64 * 64,
    "channels_image": 3,
    "z_dim": 100,
    "features_discriminator": 64,
    "features_generator": 64,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "n_critic": 5,
    "lambda_gp": 10
}

# MNIST dataset
loader = celeba(hyperparams["batch_size"], image_size=hyperparams["image_width"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = WGan(discriminator, generator, hyperparams)
gan.train(loader)
