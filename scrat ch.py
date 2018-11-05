from typing import *
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


def load_tensor(idx, queue):
    x, y = torch.load(idx)
    queue.put((x.float(), y.long()))


class DatasetTensorFolder(torch.utils.data.dataset.Dataset):
    def __init__(self, folder, worker=4):
        from multiprocessing import Pool, Manager

        self.folder = folder
        self.workers = Pool(worker)
        self.manager = Manager()

        # limit 4 batch to be loaded
        self.queue = self.manager.Queue(maxsize=4)

        self.files = list(self.folder_visitor(self.folder))

        self.result = [self.workers.apply_async(load_tensor, args=(name, self.queue)) for name in self.files]

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
        return self.queue.get(block=True)

    def __len__(self):
        return len(self.files)


train_dataset = DatasetTensorFolder(args.data, worker=args.workers)
loader = train_dataset

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

    for index, (x, y) in enumerate(loader):
        x = x.cuda()
        y = y.cuda()

        # do only 10 batches per epochs
        if index > 10:
            break

    # do only 10 `epochs`
    if epoch > 10:
        break






