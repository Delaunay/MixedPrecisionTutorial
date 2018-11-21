import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', type=str, metavar='DIR',
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
import torchvision.models.resnet as resnet

model = resnet.resnet18()

if args.arch == 'resnet50':
    model = resnet.resnet50()

criterion = nn.CrossEntropyLoss()
model = model.cuda()

criterion = criterion.cuda()

import benzina.torch

train_dataset = benzina.torch.ImageNet(args.data)

loader = benzina.torch.NvdecodeDataLoader(
    train_dataset,
    batch_size=args.batch_size,
    seed=0,
    shape=(224, 224),
    warp_transform=None,
    oob_transform=(0, 0, 0),
    scale_transform=1 / 255,
    bias_transform=-0.5
)

optimizer = torch.optim.SGD(
    model.parameters(),
    learning_rate,
    momentum=momentum,
    weight_decay=weight_decay)

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

        # Backward
        optimizer.zero_grad()
        loss.backward()

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
        print('Loading: {:.4f} s'.format(lavg), end='\t')

        total = cavg + lavg
        print('Total: {:.4f} s {:.4f} img/s'.format(total, args.batch_size / total))

    # do only 10 `epochs`
    if epoch > 10:
        break






