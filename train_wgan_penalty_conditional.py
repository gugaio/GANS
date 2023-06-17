from model.wgan_gp_conditional.wgan import WGan
from model.wgan_gp_conditional.discriminator import Discriminator
from model.wgan_gp_conditional.generator import Generator
from loaders import mnist_loader

hyperparams = {
    "image_size": 64,
    "image_area": 64 * 64,
    "channels_image": 1,
    "z_dim": 100,
    "features_discriminator": 64,
    "features_generator": 64,
    "learning_rate": 1e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "n_critic": 5,
    "lambda_gp": 10,
    "num_classes": 100,
    "embed_size": 100
}

# MNIST dataset
loader = mnist_loader(hyperparams["batch_size"], resize=hyperparams["image_size"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = WGan(discriminator, generator, hyperparams)
gan.train(loader)
