import argparse

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--data', type=str, metavar='DIR',
                    help='path to the dataset location')

parser.add_argument('--data-out', type=str, metavar='DIR',
                    help='path to the dataset location')

parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256)')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

args = parser.parse_args()

# ----------------------------
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets.folder as data


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

    for index, (x, y) in enumerate(loader):
        x = x.half()
        y = y.half()

        torch.save((x, y), args.data_out + '/img_' + str(index) + '.pt')

        if index > 10:
            break


print('Preprocessing ...')
preprocess(loader)

print('Done')
