import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=4, stride=stride, padding=1, bias=False, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, hyperparameters):
        super().__init__()
        in_channels = hyperparameters["in_channels"]
        features = hyperparameters["features"]
        image_and_mask_in_channels = in_channels*2
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels=image_and_mask_in_channels, out_channels=features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"), #256x256 -> 128x128
            nn.LeakyReLU(0.2)
        )
        layers = []
        in_channels = features[0]
        for out_channels in features[1:]:
            stride=1 if out_channels == features[-1] else 2
            layers.append(Block(in_channels, out_channels, stride))
            in_channels = out_channels
        layers.append(nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=4, stride=1,padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

def test_block():
    Batch, in_channels, out_channels, H, W = 1, 3, 64, 256, 256
    x = torch.randn((Batch, in_channels, H, W))
    model = Block(in_channels, out_channels)
    preds = model(x)
    print(preds.shape)

    hyperparameters = {
        "in_channels": 3,
        "features": [64, 128, 256, 512] #256x256 -> 30x30
    }
    model = Discriminator(hyperparameters)
    x = torch.randn((Batch, in_channels, H, W))
    y = torch.randn((Batch, in_channels, H, W))
    preds = model(x, y)
    print(preds.shape)


if __name__ == "__main__":
    test_block()
