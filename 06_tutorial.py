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

# torchvision.set_image_backend('accimage')

model = resnet.resnet18()

if args.arch == 'resnet50':
    model = resnet.resnet50()

criterion = nn.CrossEntropyLoss()

# CHANGE 1
from apex.fp16_utils import network_to_half

model = network_to_half(model.cuda())

criterion = criterion.cuda()

# >>>>
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, half=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=0, exec_async=True)

        out_type = types.FLOAT
        if half:
            out_type = types.FLOAT16

        print('Reading from {}'.format(data_dir))
        self.input = ops.FileReader(
            file_root=data_dir,
            shard_id=0,
            num_shards=1,
            random_shuffle=False)

        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)

        self.rrc = ops.RandomResizedCrop(device="gpu", size =(crop, crop))

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            output_dtype=out_type,
            output_layout=types.NCHW,
            crop=(crop, crop),
            image_type=types.RGB,
            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
            std=[0.229 * 255,0.224 * 255,0.225 * 255])

        self.coin = ops.CoinFlip(probability=0.5)
        self.jpegs = None
        self.labels = None

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        images = self.decode(self.jpegs)
        images = self.rrc(images)
        output = self.cmnp(images, mirror=rng)

        return [output, self.labels]


class DALISinglePipeAdapter:
    def __init__(self, dali_iterator):
        self.iterator = dali_iterator

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        val = self.iterator.next()
        return val[0][0][0].cuda(), val[0][1][0].cuda()


pipe = HybridTrainPipe(
    batch_size=args.batch_size,
    num_threads=args.workers,
    device_id=0,
    data_dir=args.data + '/train',
    crop=224)

pipe.build()
pipe.run()

loader = DALISinglePipeAdapter(DALIGenericIterator(
    pipe, ["data", "label"], size=int(pipe.epoch_size("Reader"))))
# <<<<

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






