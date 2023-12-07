import os
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

from mcunet.utils import (
    AverageMeter,
    accuracy,
    set_running_statistics,
)

__all__ = ["build_val_data_loader", "calib_bn", "validate"]


def build_val_data_loader(data_dir, resolution, batch_size=128, split=0):
    # split = 1: real val set, split = 1: holdout validation set
    assert split in [0, 1]
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    kwargs = {"num_workers": min(8, os.cpu_count()), "pin_memory": False}

    val_transform = transforms.Compose(
        [
            transforms.Resize(
                (resolution, resolution)
            ),  # if center crop, the person might be excluded
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_dataset = datasets.ImageFolder(data_dir, transform=val_transform)

    val_dataset = torch.utils.data.Subset(
        val_dataset, list(range(len(val_dataset)))[split::2]
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    return val_loader


def calib_bn(net, data_dir, resolution, batch_size=128, num_images=2000):
    # print('Creating dataloader for resetting BN running statistics...')
    data_loader = build_val_data_loader(data_dir, resolution, batch_size, split=1)
    set_running_statistics(net, data_loader)


def validate(model, val_loader):
    model.eval()
    val_loss = AverageMeter()
    val_top1 = AverageMeter()

    with tqdm(total=len(val_loader), desc="Validate") as t:
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.cuda(), target.cuda()

                output = model(data)
                val_loss.update(F.cross_entropy(output, target).item())
                top1 = accuracy(output, target, topk=(1,))[0]
                val_top1.update(top1.item(), n=data.shape[0])
                t.set_postfix({"loss": val_loss.avg, "top1": val_top1.avg})
                t.update(1)

    return val_top1.avg
