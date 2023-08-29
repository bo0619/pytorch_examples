# FPN 不适用于图像分类，因为图像分类不太关注语义信息
# backbone: modified AlexNet
# head: 3 detection head
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 96, kernel_size=11, stride=2, padding=5)
        self.bn1 = nn.BatchNorm2d(96)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(384)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(384)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(384, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    # figure3中没有3*3的卷积，但是4.1节有详细的文字叙述
    def conv3x3(self, in_channels, out_channels, stride=1, padding=1):
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding)

    def _add_and_upsample(self, left, up):
        up = nn.Upsample(scale_factor=2, mode='nearest').to(left.device)(up)
        left = nn.Conv2d(in_channels=left.shape[1], out_channels=up.shape[1], kernel_size=1, stride=1, padding=0).to(left.device)(left)
        return left + up

    def forward(self, x):
        out1 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        out2 = self.pool2(self.relu2(self.bn2(self.conv2(out1))))
        out3 = self.pool3(self.relu3(self.bn3(self.conv3(out2))))
        out4 = self.pool4(self.relu4(self.bn4(self.conv4(out3))))

        pyramid1 = out4
        pyramid2 = self._add_and_upsample(out3, pyramid1)
        pyramid3 = self._add_and_upsample(out2, pyramid2)

        pyramid2 = self.conv3x3(in_channels=pyramid2.shape[1], out_channels=pyramid1.shape[1]).to(pyramid2.device)(pyramid2)
        pyramid3 = self.conv3x3(in_channels=pyramid3.shape[1], out_channels=pyramid2.shape[1]).to(pyramid3.device)(pyramid3)

        # 三个检测头
        pred1 = self.fc2(self.fc1(torch.flatten(self.avgpool(pyramid1), 1)))
        pred2 = self.fc2(self.fc1(torch.flatten(self.avgpool(pyramid2), 1)))
        pred3 = self.fc2(self.fc1(torch.flatten(self.avgpool(pyramid3), 1)))

        return pred1, pred2, pred3

