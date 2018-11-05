from typing import *
import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to the dataset location')

parser.add_argument('--data-out', type=str, metavar='DIR',
                    help='path to the dataset location')

parser.add_argument('--arch', type=str, default='resnet18',
                    help='Conv Net Architecture')

parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-e', '--epochs', default=10, type=int,
                    help='Number of epochs')

learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
args = parser.parse_args()

# ----------------------------
import torch
import torch.nn as nn

import torch.optim

import torch.utils.data

import torchvision
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets.folder as data

# torchvision.set_image_backend('accimage')

model = resnet.resnet18()

if args.arch == 'resnet50':
    model = resnet.resnet50()

criterion = nn.CrossEntropyLoss()

# CHANGE 1
from apex.fp16_utils import network_to_half

model = network_to_half(model.cuda())

criterion = criterion.cuda()

train_dataset = data.ImageFolder(
    args.data + '/train/',
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
)

loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=None,
    num_workers=args.workers,
    pin_memory=True)


def preprocess(loader):
    for epoch in range(0, args.epochs):

        for index, (x, y) in enumerate(loader):
            x = x.half()
            y = y.half()

            torch.save((x, y), args.data_out + '/img_' + str(index) + '.pt')

        if epoch > 10:
            break


print('Preprocessing ...')
preprocess(loader)


class DatasetTensorFolder(torch.utils.data.dataset.Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = list(self.folder_visitor(self.folder))

    @staticmethod
    def folder_visitor(folder) -> List[str]:
        import fnmatch
        import os

        classes = set()

        for root, _, files in os.walk(folder):
            name = root.split('/')[-1]
            if root != folder:
                classes.add(name)

            for item in fnmatch.filter(files, "*"):
                yield '{}/{}/{}'.format(folder, name, item)

    def __getitem__(self, index):
        return torch.load(self.files[index])

    def __len__(self):
        return len(self.files)


train_dataset = DatasetTensorFolder(args.data_out)

loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=None,
    num_workers=args.workers,
    pin_memory=True)


optimizer = torch.optim.SGD(
    model.parameters(),
    learning_rate,
    momentum=momentum,
    weight_decay=weight_decay)

# CHANGE 2
import apex.fp16_utils.fp16_optimizer as apex_optimizer

optimizer = apex_optimizer.FP16_Optimizer(optimizer, static_loss_scale=256)

# ---------------------------------
import time

model.train()

compute_start = 0
compute_end = 0
compute_avg = 0
compute_count = 0

loading_start = 0
loading_end = 0
loading_avg = 0

skip = 20
count = 0
torch.backends.cudnn.benchmark = True

for epoch in range(0, args.epochs):
    loading_start = time.time()

    for index, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()
        loading_end = time.time()

        # Forward
        compute_start = time.time()
        output = model(x)
        loss = criterion(output, y.long())
        floss = loss.item()

        # CHANGE 3
        # Backward
        optimizer.zero_grad()
        optimizer.backward(loss)

        # Update
        optimizer.step()
        compute_end = time.time()

        # Update Stats
        if count > skip:
            compute_avg += compute_end - compute_start
            compute_count += 1

            loading_avg += loading_end - loading_start
            loading_start = time.time()

        count += 1
        # do only 10 batches per epochs
        if index > 10:
            break

    if compute_count > 0:
        cavg = compute_avg / compute_count
        print('Compute: {:.4f} s  {:.4f} img/s'.format(cavg, args.batch_size / cavg), end='\t')

        lavg = loading_avg / compute_count
        print('Loading: {:.4f} s  {:.4f} img/s'.format(lavg, args.batch_size / lavg))

    # do only 10 `epochs`
    if epoch > 10:
        break






