from cfg import *
from torchvision import datasets, transforms
from torchvision.datasets.cifar import CIFAR10
from torchvision.datasets.imagenet import ImageNet
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomResizedCrop, RandomGrayscale, RandomHorizontalFlip, RandomRotation, RandomErasing


train_transformer = Compose([
    # Resize(INPUT_SIZE),
    RandomResizedCrop(INPUT_SIZE),
    RandomGrayscale(),
    RandomHorizontalFlip(),
    RandomRotation(10),
    ToTensor(),
    RandomErasing(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

test_transformer = Compose([
    Resize(INPUT_SIZE),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

mnist_train_transformer = Compose([
    # RandomResizedCrop(INPUT_SIZE),
    # RandomRotation(10),
    Resize(INPUT_SIZE),
    ToTensor()
])

mnist_test_transformer = Compose([
    Resize(INPUT_SIZE),
    ToTensor()
])


def get_dataset():
    if DATASET == 'cifar10':
        train_set = CIFAR10(DATASET_ROOT, train=True, transform=train_transformer, download=True)
        test_set = CIFAR10(DATASET_ROOT, train=False, transform=test_transformer, download=True)
        return train_set, test_set
    elif DATASET == 'imagenet':
        train_set = ImageNet(DATASET_ROOT, split='train', transform=train_transformer)
        test_set = ImageNet(DATASET_ROOT, split='val', transform=test_transformer)
        return train_set, test_set
    elif DATASET == 'mnist':
        train_set = MNIST(DATASET_ROOT, train=True, transform=mnist_train_transformer,  download=True)
        test_set = MNIST(DATASET_ROOT, train=False, transform=mnist_test_transformer, download=True)
        return train_set, test_set
    else:
        raise NotImplementedError("Unsupported dataset")
