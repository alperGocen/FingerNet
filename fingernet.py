import torch.nn as nn


class FingerNet(nn.Module):
    """
    Creates the FingerNet model
    params : num_classes -> the number of classes in the classification problem
    """
    def __init__(self, num_classes):
        super(FingerNet, self).__init__()

        # Conv1
        self.conv1_1 = nn.Conv2d(1, 20, kernel_size=4, stride=1)
        self.bn1_1 = nn.BatchNorm2d(20)
        self.elu1_1 = nn.ELU()

        self.conv1_2 = nn.Conv2d(20, 40, kernel_size=4, stride=1)
        self.bn1_2 = nn.BatchNorm2d(20)
        self.elu1_2 = nn.ELU()

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Conv2
        self.conv2_1 = nn.Conv2d(40, 80, kernel_size=3, stride=1)
        self.bn2_1 = nn.BatchNorm2d(80)
        self.elu2_1 = nn.ELU()

        self.conv2_2 = nn.Conv2d(80, 80, kernel_size=3, stride=1)
        self.bn2_2 = nn.BatchNorm2d(80)
        self.elu2_2 = nn.ELU()

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Conv3
        self.conv3_1 = nn.Conv2d(80, 120, kernel_size=2, stride=1)
        self.bn3_1 = nn.BatchNorm2d(120)
        self.elu3_1 = nn.ELU()

        self.conv3_2 = nn.Conv2d(120, 120, kernel_size=2, stride=1)
        self.bn3_2 = nn.BatchNorm2d(120)
        self.elu3_2 = nn.ELU()

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        # Conv4
        self.conv4_1 = nn.Conv2d(120, 160, kernel_size=2, stride=2)
        self.bn4_1 = nn.BatchNorm2d(160)
        self.elu4_1 = nn.ELU()

        self.conv4_2 = nn.Conv2d(160, 160, kernel_size=2, stride=1)
        self.bn4_2 = nn.BatchNorm2d(160)
        self.elu4_2 = nn.ELU()

        # Conv5
        self.conv5_1 = nn.Conv2d(160, 320, kernel_size=2, stride=2)
        self.bn5_1 = nn.BatchNorm2d(320)
        self.elu5_1 = nn.ELU()

        self.conv5_2 = nn.Conv2d(320, 320, kernel_size=2, stride=1)
        self.bn5_2 = nn.BatchNorm2d(320)
        self.elu5_2 = nn.ELU()

        # local
        self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc1 = nn.Linear(5*5*320, 1)
        self.fc2 = nn.Linear(1, num_classes)

    def forward(self, x):
        x = self.elu1_1(self.bn1_1(self.conv1_1(x)))
        x = self.elu1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.elu2_1(self.bn2_1(self.conv2_1(x)))
        x = self.elu2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.elu3_1(self.bn3_1(self.conv3_1(x)))
        x = self.elu3_2(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.elu4_1(self.bn4_1(self.conv4_1(x)))
        x = self.elu4_2(self.bn4_2(self.conv4_2(x)))
        x = self.elu5_1(self.bn5_1(self.conv5_1(x)))
        x = self.elu5_2(self.bn5_2(self.conv5_2(x)))

        x = x.view(-1, 5*5*320)
        x = self.fc1(x)
        y = self.fc2(x)

        return x, y


__factory = {
    'finNet': FingerNet
}


def create(name, num_classes):
    if name not in __factory.keys():
        raise KeyError("Unknown model: {}".format(name))
    return __factory[name](num_classes)


if __name__ == '__main__':
    pass


