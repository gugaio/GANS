import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.channels_noise = hyperparams["channels_noise"]
        features_gen = hyperparams["features_generator"]
        channels_image = hyperparams["channels_image"]
        self.model = nn.Sequential(            
            self._block(in_channels=self.channels_noise, out_channels=features_gen *8, kernel_size=4, stride=1, padding=0), #1x1 -> 4x4
            self._block(in_channels=features_gen *8, out_channels=features_gen *4, kernel_size=4, stride=2, padding=1), #8x8 -> 16x16
            self._block(in_channels=features_gen *4, out_channels=features_gen *2, kernel_size=4, stride=2, padding=1), #16x16 -> 32x32
            nn.ConvTranspose2d(features_gen * 2, channels_image, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )


    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.ReLU(),
        )


    def forward(self, x):
        return self.model(x)
    
##### TEST #####

def test():
    batch_size, channels_noise, H, W = 8, 100, 1, 1
    x = torch.randn((batch_size, channels_noise, H, W))
    hyperparams = {
        "channels_noise": channels_noise,
        "features_generator": 32,
        "channels_image": 1
    }
    model = Generator(hyperparams)
    preds = model(x)
    print(preds.shape)

if __name__ == "__main__":
    test()