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

def celeba(batch_size, image_size=32, root_path='celeba/'):   
    pipeline = []
    pipeline.append(transforms.Resize((image_size,image_size)))
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(
        [0.5 for _ in range(3)], [0.5 for _ in range(3)]
    ))
    transform = transforms.Compose(pipeline)

    dataset = datasets.ImageFolder(root=root_path, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
