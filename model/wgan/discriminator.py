import torch
import torch.nn as nn

class Discriminator(nn.Module):

    def __init__(self, hyperparameters) -> None:
        super(Discriminator, self).__init__()

        channels = hyperparameters["channels_image"]
        features = hyperparameters["features_discriminator"]

        self.model = nn.Sequential(            
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=4, stride=2, padding=1), #64x64 -> 31x31
            nn.LeakyReLU(0.2),
            self._block(features, features*2, 4, 2, 1), # 31x31 -> 16x16
            self._block(features*2, features*4, 4, 2, 1), # 16x16 -> 8x8
            self._block(features*4, features*8, 4, 2, 1), # 16x16 -> 8x8
            nn.Conv2d(features*8, out_channels=1, kernel_size=4, stride=2, padding=0), # 1x1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2)
        )
    
    def forward(self, x):
        return self.model(x)
    
##### TEST #####

def test():
    batch_size, in_channels, H, W = 8, 3, 64, 64
    x = torch.randn((batch_size, in_channels, H, W))
    hyperparams = {
        "channels_image": in_channels,
        "features_discriminator": 64
    }
    model = Discriminator(hyperparams)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()