import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchsummary import summary
from matplotlib import pyplot as plt

transform = transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageFolder(root='data/flower_images/train', transform=transform)
test_dataset = ImageFolder(root='data/flower_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

class GroupedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, cardinality=32):
        super(GroupedResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.conv3(out)
        out += self.downsample(residual)
        out = F.relu(out)
        return out
    
class ResNeXt(nn.Module):
    def __init__(self, params:list, num_classes=5):
        """
        params: num of repeated residual blocks.

        e.g. the params of resnext-18 is [2, 2, 2, 2].
        """
        super(ResNeXt, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.block2 = self._build_basic_block(64, 128, params[0], stride=1)
        self.block3 = self._build_basic_block(128, 256, params[1], stride=2)
        self.block4 = self._build_basic_block(256, 512, params[2], stride=2)
        self.block5 = self._build_basic_block(512, 1024, params[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_classes)


    def _build_basic_block(self, in_channels, out_channels, num_blocks, stride=1, cardinality=32):
        layers = []
        layers.append(GroupedResidualBlock(in_channels, out_channels, stride, cardinality=cardinality))
        for _ in range(1, num_blocks):
            layers.append(GroupedResidualBlock(out_channels, out_channels, stride=1, cardinality=cardinality))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)  
        out = self.block5(out)      
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out
    
def resnext_18():
    return ResNeXt([2, 2, 2, 2])

def resnext_34():
    return ResNeXt([3, 4, 6, 3])

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = resnext_34().to(device)
summary(model, (3, 224, 224))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses, acces, eval_losses, eval_acces = [], [], [], []

print('-----------------------------------------------------Start Training-----------------------------------------------------')
for i in range(50):
    train_loss = 0
    train_acc = 0
    model.train()
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        out = model(image)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / image.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('i: {}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(i, train_loss / len(train_loader), train_acc / len(train_loader)))
    
    eval_loss = 0
    eval_acc = 0
    model.eval()
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        eval_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        eval_acc += acc
    eval_losses.append(eval_loss / len(test_loader))
    eval_acces.append(eval_acc / len(test_loader))
    print('i: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(i, eval_loss / len(test_loader), eval_acc / len(test_loader)))


# 保存模型
torch.save(model.state_dict(), 'models/ResNeXt-34.pth')

# 绘制训练、测试loss曲线
plt.title('train and test loss')
plt.plot(np.arange(len(losses)), losses)
plt.plot(np.arange(len(eval_losses)), eval_losses)
plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# 绘制accuracy曲线
plt.title('train and test accuracy')
plt.plot(np.arange(len(acces)), acces)
plt.plot(np.arange(len(eval_acces)), eval_acces)
plt.legend(['Train Acc', 'Test Acc'], loc='upper right')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
