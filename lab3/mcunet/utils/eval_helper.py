import os.path as osp
import numpy as np
import math
from tqdm import tqdm

import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms, datasets

from mcunet.utils import AverageMeter, accuracy
from mcunet.utils import set_running_statistics


def evaluate_ofa_subnet(
    ofa_net,
    path,
    net_config,
    batch_size,
    device="cuda:0",
    subsample_factor=1,
    num_workers=16,
):
    ofa_net.set_active_subnet(**net_config)
    subnet = ofa_net.get_active_subnet().to(device)
    calib_bn(subnet, path, net_config["image_size"], batch_size)
    dataset = datasets.ImageFolder(osp.join(path, "val"))
    if subsample_factor > 1:
        chosen_indexes = list(range(len(dataset)))[::subsample_factor]
    else:
        chosen_indexes = list(range(len(dataset)))
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        sampler=sub_sampler,
    )
    top1 = validate(
        subnet, path, net_config["image_size"], data_loader, batch_size, device
    )
    return top1


def calib_bn(net, path, image_size, batch_size, num_images=2000):
    # print('Creating dataloader for resetting BN running statistics...')
    dataset = datasets.ImageFolder(
        osp.join(path, "train"),
        transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=32.0 / 255.0, saturation=0.5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    )
    chosen_indexes = np.random.choice(list(range(len(dataset))), num_images)
    sub_sampler = torch.utils.data.sampler.SubsetRandomSampler(chosen_indexes)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sub_sampler,
        batch_size=batch_size,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
    )
    # print('Resetting BN running statistics (this may take 10-20 seconds)...')
    set_running_statistics(net, data_loader)


def validate(net, path, image_size, data_loader, batch_size=100, device="cuda:0"):
    net = net.to(device)

    data_loader.dataset.transform = transforms.Compose(
        [
            transforms.Resize(int(math.ceil(image_size / 0.875))),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cudnn.benchmark = True
    criterion = nn.CrossEntropyLoss().to(device)

    net.eval()
    net = net.to(device)
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="Validate") as t:
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                # compute output
                output = net(images)
                loss = criterion(output, labels)
                # measure accuracy and record loss
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))

                losses.update(loss.item(), images.size(0))
                top1.update(acc1[0].item(), images.size(0))
                top5.update(acc5[0].item(), images.size(0))
                t.set_postfix(
                    {
                        "loss": losses.avg,
                        "top1": top1.avg,
                        "top5": top5.avg,
                        "img_size": images.size(2),
                    }
                )
                t.update(1)

    print(
        "Results: loss=%.5f,\t top1=%.1f,\t top5=%.1f"
        % (losses.avg, top1.avg, top5.avg)
    )
    return top1.avg
