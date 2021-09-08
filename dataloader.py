import torch
import torchvision
from torch.utils.data import DataLoader

from torchvision import transforms


class Dataset(object):
    def __init__(self, batch_size, use_gpu, num_workers):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081))
        ])

        pin_memory = True if use_gpu else False

        trainset = torchvision.datasets.ImageFolder('Dbs/Db1/data/Db1_a', transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        validset = torchvision.datasets.ImageFolder('Dbs/Db1/data/Db1_av', transform=transform)
        validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False,
                                                  num_workers=num_workers, pin_memory=pin_memory)

        self.trainloader = trainloader
        self.validloader = validloader
        self.num_classes = 99


__factory = {
    'dataset': Dataset
}


def create(name, batch_size, use_gpu, num_workers):
    if name not in __factory.keys():
        raise KeyError("Not the correct dataset!!".format(name))

    return __factory[name](batch_size, use_gpu, num_workers)

