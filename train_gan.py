import torch
import torch.nn as nn
import torch.optim as optim
from loaders import mnist_loader
from model.gan import Gan

from torch.utils.tensorboard import SummaryWriter

hyperparams = {
    "image_size": 784,
    "z_dim": 64,
    "learning_rate": 3e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "generator_hidden_size": 256,
    "discriminator_hidden_size": 128,
}

# MNIST dataset
loader = mnist_loader(hyperparams["batch_size"])

# Gan Model
gan = Gan(hyperparams)
gan.train(loader)
