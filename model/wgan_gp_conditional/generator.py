import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, hyperparams):
        super().__init__()
        self.z_dim = hyperparams["z_dim"]
        features_gen = hyperparams["features_generator"]
        channels_image = hyperparams["channels_image"]
        self.batch_size = hyperparams["batch_size"]
        embed_size = hyperparams["embed_size"]
        num_classes = hyperparams["num_classes"]
        self.model = nn.Sequential(            
            self._block(in_channels=self.z_dim + embed_size, out_channels=features_gen *16, kernel_size=4, stride=1, padding=0), #1x1 -> 4x4
            self._block(in_channels=features_gen *16, out_channels=features_gen *8, kernel_size=4, stride=2, padding=1), #4x4 -> 8x8
            self._block(in_channels=features_gen *8, out_channels=features_gen *4, kernel_size=4, stride=2, padding=1), #8x8 -> 16x16
            self._block(in_channels=features_gen *4, out_channels=features_gen *2, kernel_size=4, stride=2, padding=1), #16x16 -> 32x32
            nn.ConvTranspose2d(features_gen * 2, channels_image, kernel_size=4, stride=2, padding=1), #32x32 -> 64x64
            nn.Tanh(),
        )
        self.embed = nn.Embedding(num_classes, embed_size)


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
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


    def forward(self, x, labels):
        embedding = self.embed(labels) # (batch_size, embed_size)
        embedding = embedding.unsqueeze(2).unsqueeze(3) # (batch_size, embed_size, 1, 1)
        x = torch.cat([x, embedding], dim=1) # (batch_size, z_dim + embed_size, 1, 1)
        return self.model(x)
    
    def sample_noise(self):
        H, W = 1, 1
        return torch.randn((self.batch_size, self.z_dim, H, W))
    
##### TEST #####

def test():
    batch_size, z_dim, H, W = 8, 100, 1, 1
    embed_size = 100
    x = torch.randn((batch_size, z_dim, H, W))
    num_classes = 10
    labels = torch.randint(0, num_classes, (batch_size,))
    hyperparams = {
        "z_dim": z_dim,
        "features_generator": 64,
        "channels_image": 1,
        "batch_size": batch_size,
        "embed_size": embed_size,
        "num_classes": num_classes,
    }
    model = Generator(hyperparams)
    preds = model(x, labels)
    print(preds.shape)

if __name__ == "__main__":
    test()