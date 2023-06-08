import torch
import torch.nn as nn
import torch.optim as optim
from loaders import mnist_loader2
from model.dc_gan import DcGan

from torch.utils.tensorboard.writer import SummaryWriter

hyperparams = {
    "image_size": 64,
    "z_dim": 64,
    "learning_rate": 2e-4,
    "num_epochs": 50,
    "batch_size": 32,
    "channels_image": 1,
    "features_discriminator": 64,
    "features_generator": 64,
    "channels_noise": 100
}

# MNIST dataset
loader = mnist_loader2(hyperparams["batch_size"])

# Gan Model
gan = DcGan(hyperparams)
gan.train(loader)
