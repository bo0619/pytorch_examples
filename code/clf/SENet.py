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

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class SEModule(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SEResNetModule(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEModule(out_channels, out_channels)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        se_out = self.se(self.bn2(self.conv2(self.relu1(self.bn1(self.conv1(x))))))
        return F.relu(se_out + identity)
    
class SEResNet(nn.Module):
    def __init__(self, params:list, num_classes=5):
        """
        params: num of repeated blocks 

        e.g. the params of SEResNet-18 is [2, 2, 2, 2]
        """
        super(SEResNet, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.block2 = self._build_se_basicblock(64, 64, params[0])
        self.block3 = self._build_se_basicblock(64, 128, params[1], stride=2)
        self.block4 = self._build_se_basicblock(128, 256, params[2], stride=2)
        self.block5 = self._build_se_basicblock(256, 512, params[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)


    def _build_se_basicblock(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(SEResNetModule(in_channels, out_channels, stride))
        for i in range(1, num_blocks):
            layers.append(SEResNetModule(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x) 
        x = self.block4(x)
        x = self.block5(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SEResNet([3, 4, 6, 3]).to(device)
summary(model,(3, 224, 224))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
acces = []
eval_losses = []
eval_acces = []

# 训练
print('-----------------------------------------------------Start Training-----------------------------------------------------')
for epoch in range(35):
    train_loss = 0
    train_acc = 0
    model.train()
    for img, label in train_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)
        loss = criterion(out, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = out.max(1)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        train_acc += acc
    losses.append(train_loss / len(train_loader))
    acces.append(train_acc / len(train_loader))
    print('epoch: {}, Train Loss: {:.4f}, Train Acc: {:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
    
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
    print('epoch: {}, Test Loss: {:.4f}, Test Acc: {:.4f}'.format(epoch, eval_loss / len(test_loader), eval_acc / len(test_loader)))

# 保存模型
torch.save(model.state_dict(), 'models/SEResNet-34.pth')

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