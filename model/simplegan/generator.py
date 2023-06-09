import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.z_dim = hyperparams["z_dim"]
        hidden_size = hyperparams["generator_hidden_size"]
        image_size = hyperparams["image_size"]
        self.batch_size = hyperparams["batch_size"]

        self.model = nn.Sequential(
            nn.Linear(self.z_dim, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, image_size),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.model(noise).reshape(-1, 1, 28, 28)
    
    def sample_noise(self):
        return torch.randn((self.batch_size, self.z_dim))
    

##### TEST #####

def test():
    batch_size, z_dim, in_channels, H, W = 8, 64, 1, 28, 28

    x = torch.randn((batch_size, z_dim))
    x = x.view(batch_size, -1)

    hyperparams = {
        "image_size": in_channels*H*W,
        "z_dim": z_dim,
        "generator_hidden_size": 256,
        "batch_size": batch_size,
    }
    model = Generator(hyperparams)
    print(model(x).shape)

if __name__ == "__main__":
    test()