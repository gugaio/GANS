from model.gan import Gan
from model.dcgan.discriminator import Discriminator
from model.dcgan.generator import Generator
from loaders import mnist_loader

hyperparams = {
    "image_width": 32,
    "image_size": 32 * 32,
    "channels_image": 1,
    "z_dim": 100,
    "features_discriminator": 64,
    "features_generator": 64,
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 32
}

# MNIST dataset
loader = mnist_loader(hyperparams["batch_size"], resize=hyperparams["image_width"])

# Gan Model
discriminator = Discriminator(hyperparams)
generator = Generator(hyperparams)
gan = Gan(discriminator, generator, hyperparams)

model_size =  sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
print("Number of parameters in discriminator: %d" % model_size)
model_size =  sum(p.numel() for p in generator.parameters() if p.requires_grad)
print("Number of parameters in generator: %d" % model_size)

gan.train(loader)
