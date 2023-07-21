# 本代码全部为GitHub CopilotX自动生成

import os
import numpy as np
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from matplotlib import pyplot as plt
from torchsummary import summary
from PIL import Image

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)



class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_convs):
        super(DenseBlock, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.layers = []
        for i in range(num_convs):
            self.layers.append(self.conv_block(in_channels + i * growth_rate, growth_rate))
        self.layers = nn.ModuleList(self.layers)
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        for layer in self.layers:
            out = layer(x)
            x = torch.cat((x, out), dim=1)
        return x
    

class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        return self.layers(x)
    

class DenseNet(nn.Module):
    def __init__(self, num_classes: int = 5, growth_rate: int = 32, num_convs: list = [6, 12, 24, 16]) -> None:
        super(DenseNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        in_channels = 64
        for i, num_conv in enumerate(num_convs):
            self.features.add_module(f'dense_block_{i + 1}', DenseBlock(in_channels, growth_rate, num_conv))
            in_channels += num_conv * growth_rate
            if i != len(num_convs) - 1:
                self.features.add_module(f'transition_block_{i + 1}', TransitionBlock(in_channels, in_channels // 2))
                in_channels = in_channels // 2
        self.features.add_module('norm', nn.BatchNorm2d(in_channels))
        self.features.add_module('relu', nn.ReLU())
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DenseNet().to(device)
summary(model, (3, 224, 224))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
acces = []
eval_losses = []
eval_acces = []

# 训练
print('-----------------------------------------------------Start Training-----------------------------------------------------')
for epoch in range(50):
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
torch.save(model.state_dict(), 'models/AlexNet.pth') 

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