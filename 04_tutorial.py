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

import torchvision
import torchvision.models.resnet as resnet
import torchvision.transforms as transforms
import torchvision.datasets.folder as data

#torchvision.set_image_backend('accimage')

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
        #transforms.ToTensor(),
        #transforms.Normalize(
        #    mean=[0.485, 0.456, 0.406],
        #    std=[0.229, 0.224, 0.225]
        #),
    ])
)


class Prefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).float().view(1, 3, 1, 1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).float().view(1, 3, 1, 1)
        self.next_target = None
        self.next_input = None
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        self.next_input, self.next_target = next(self.loader)

        with torch.cuda.stream(self.stream):
            #  Send to device
            self.next_input = self.next_input.cuda(async=True)
            self.next_target = self.next_target.cuda(async=True)

            # Convert to half
            self.next_input = self.next_input.half()

            # Normalize on the GPU
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __next__(self):
        # Wait for next_input to be read
        torch.cuda.current_stream().wait_stream(self.stream)

        input = self.next_input
        target = self.next_target

        self.preload()
        return input, target

    next = __next__


def fast_collate(batch):
    """
        from Apex by NVIDIA
    """
    import numpy as np

    imgs = [img[0] for img in batch]

    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)

    w = imgs[0].size[0]
    h = imgs[0].size[1]

    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)

    for i, img in enumerate(imgs):

        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)

        if nump_array.ndim < 3:
            nump_array = np.expand_dims(nump_array, axis=-1)

        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)

    return tensor, targets


loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=None,
    num_workers=args.workers,
    pin_memory=True,
    collate_fn=fast_collate)

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
    fetcher = Prefetcher(loader)

    for index, (x, y) in enumerate(fetcher):
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






