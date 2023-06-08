import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, hyperparameters) -> None:
        super(Discriminator, self).__init__()

        channels_image = hyperparameters["channels_image"]
        features_disc = hyperparameters["features_discriminator"]

        self.model = nn.Sequential(            
            nn.Conv2d(in_channels=channels_image, out_channels=features_disc, kernel_size=4, stride=2, padding=1), #64x64 -> 31x31
            nn.LeakyReLU(0.2),
            self._block(features_disc, features_disc*2, 4, 2, 1), # 31x31 -> 16x16
            self._block(features_disc*2, features_disc*4, 4, 2, 1), # 16x16 -> 8x8
            nn.Conv2d(features_disc*4, out_channels=1, kernel_size=4, stride=2, padding=0), # 1x1x1
            nn.Sigmoid()
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        assert x.shape[1:] == (1, 32, 32), f"Expected (1, 32, 32) but got {x.shape[1:]}"
        return self.model(x)
    
##### TEST #####

def test():
    batch_size, in_channels, H, W = 8, 1, 32, 32
    x = torch.randn((batch_size, in_channels, H, W))
    hyperparams = {
        "channels_image": in_channels,
        "features_discriminator": 32
    }
    model = Discriminator(hyperparams)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()