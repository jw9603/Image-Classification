from torchvision import datasets, transforms

def load_mnist(is_train=True, flatten=True):

    dataset = datasets.MNIST(
        './data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
    )

    x = dataset.data.float() / 255.
    y = dataset.targets
    
    print(f"mnist size of X:{x.shape}") # torch.Size([60000, 28, 28])
    print(f"mnist size of Y:{x.shape}") # torch.Size([60000, 28, 28])

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y
