import os
import numpy as np
import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
from torchsummary import summary


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/train', transform=transform)
test_dataset = torchvision.datasets.ImageFolder(root='data/flower_images/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SpatialPyramidPooling, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        N, C, H, W = x.size()
        for i in range(self.num_levels):
            level = i + 1
            kernel_size = (int(np.ceil(H / level)), int(np.ceil(W / level)))
            stride = (int(np.floor(H / level)), int(np.floor(W / level)))
            padding = (int(np.floor((kernel_size[0] * level - H + 1) / 2)), int(np.floor((kernel_size[1] * level - W + 1) / 2)))
            pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
            tensor = pooling(x)
            if i == 0:
                spp = tensor.view(N, -1)
            else:
                spp = torch.cat((spp, tensor.view(N, -1)), 1)
        return spp

# 一个随便的三层CNN
class SPPNet(nn.Module):
    def __init__(self, num_classes):
        super(SPPNet, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=1, stride=2)

        self.spp = SpatialPyramidPooling(num_levels=3, pool_type='max_pool')

        self.fc1 = nn.Linear(256 * 14, 1024)
        self.fc2 = nn.Linear(1024, self.num_classes)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.act3(x)
        x = self.pool3(x)

        x = self.spp(x)
        x = F.dropout(x, p=0.5)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
model = SPPNet(num_classes=10).to(device)
summary(model, (3, 224, 224))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

losses = []
acces = []
eval_losses = []
eval_acces = []


# 训练
print('-----------------------------------------------------Start Training-----------------------------------------------------')
for epoch in range(25):
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
