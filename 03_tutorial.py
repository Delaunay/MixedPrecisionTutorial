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

parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')

parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                         '--static-loss-scale.')

learning_rate = 0.1
momentum = 0.9
weight_decay = 1e-4
args = parser.parse_args()

# ----------------------------
import torch
import torch.nn as nn

import torch.optim

import torch.utils.data
import torch.utils.data.distributed

import torchvision
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets.folder as data

from apex.fp16_utils import network_to_half

model = resnet.resnet18

if args.arch == 'resnet50':
    model = resnet.resnet50

criterion = nn.CrossEntropyLoss()

# CHANGE 1
# cast model to half precision
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

optimizer = torch.optim.SGD(
    model.parameters(),
    args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay)

# CHANGE 2
# Wrap the optimizer to get loss scaling
import apex.fp16_utils.fp16_optimizer as apex_optimizer

optimizer = apex_optimizer.FP16_Optimizer(optimizer,
    static_loss_scale=args.static_loss_scale,
    dynamic_loss_scale=args.dynamic_loss_scale)

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

# Enable pytorch auto tune
torch.backends.cudnn.benchmark = True

for epoch in range(0, args.epochs):
    loading_start = time.time()

    for index, (x, y) in enumerate(data):
        loading_end = time.time()

        # Forward
        compute_start = time.time()
        output = model(x)
        loss = criterion(output, y.long())
        floss = loss.item()

        # Backward
        optimizer.zero_grad()

        # CHANGE 3
        # the backward step is done by the optimizer to handle
        # loss scaling
        optimizer.backward(loss)
        optimizer.step()
        compute_end = time.time()

        # Update Stats
        compute_avg += compute_end - compute_start
        compute_count += 1

        loading_avg += loading_end - loading_start
        loading_start = time.time()

        # do only 10 batches per epochs
        if index > 10:
            break

        print('Compute: {:.4f} s  {:.4f} img/s'.format(compute_avg, args.batch_size / compute_avg))
        print('Loading: {:.4f} s  {:.4f} img/s'.format(loading_avg, args.batch_size / loading_avg))

    # do only 10 `epochs`
    if epoch > 10:
        break






