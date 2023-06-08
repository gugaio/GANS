import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        hidden_size = hyperparams["discriminator_hidden_size"]
        image_size = hyperparams["image_size"]

        self.model = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
    
##### TEST #####

def test():
    batch_size, in_channels, H, W = 8, 1, 28, 28
    x = torch.randn((batch_size, in_channels, H, W))
    x = x.view(batch_size, -1)
    hyperparams = {
        "image_size": in_channels*H*W,
        "discriminator_hidden_size": 128
    }
    model = Discriminator(hyperparams)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()