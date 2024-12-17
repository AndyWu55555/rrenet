import os
import random

import numpy as np
import torch
import torch.utils.data
import torchvision
from torchvision import transforms


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(2025)


def cifar100_dataloader(args):
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            transform=transform_test
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, test_loader


def cifar10_dataloader(args):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            transform=transform_train,
            download=True
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            transform=transform_test
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    return train_loader, test_loader


def get_dataloader(args):
    assert args.dataset in ['cifar10', 'cifar100']
    if args.dataset == 'cifar100':
        return cifar100_dataloader(args)
    elif args.dataset == 'cifar10':
        return cifar10_dataloader(args)
