import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def mnist_loader(batch_size, resize=-1):   
    pipeline = []
    if resize>0:
        pipeline.append(transforms.Resize(resize))
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    transform = transforms.Compose(pipeline)

    dataset = datasets.MNIST(root='dataset/', train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
