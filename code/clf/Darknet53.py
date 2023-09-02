# Darknet53 for YOLOv3
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

train_dataset = ImageFolder(root='/kaggle/input/flower-5/flower_images/train', transform=transform)
test_dataset = ImageFolder(root='/kaggle/input/flower-5/flower_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

class DBLBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DBLBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.conv1 = DBLBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = DBLBlock(in_channels=out_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class Darknet53(nn.Module):
    def __init__(self, num_classes):
        super(Darknet53, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            self._build_n_resblock(1, 64, 32),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            self._build_n_resblock(2, 128, 64),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            self._build_n_resblock(8, 256, 128),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            self._build_n_resblock(8, 512, 256),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            self._build_n_resblock(4, 1024, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.Linear(4096, self.num_classes)
        )

    def _build_n_resblock(self, n:int, in_channels, out_channels):
        model = nn.Sequential()
        for _ in range(n):
            model.add_module(f"resblock{_+1}", ResBlock(in_channels=in_channels, out_channels=out_channels))
        
        return model

    def forward(self, input):
        x = self.features(input)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Darknet53(5).to(device)
summary(model, (3, 224, 224))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


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
torch.save(model.state_dict(), '/kaggle/working/Darknet-53.pth')


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