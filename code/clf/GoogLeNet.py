import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision
from matplotlib import pyplot as plt
from torchsummary import summary


transform = transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
])

train_dataset = ImageFolder(root='data/flower_images', transform=transform)
test_dataset = ImageFolder(root='data/flower_images', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


class InceptionModule(nn.Module):
    def __init__(self, in_channels, params:list):
        """
        params: [#1*1, #3*3, #5*5, pool_proj]
        """
        super(InceptionModule, self).__init__()
        self.branch1 = nn.Conv2d(in_channels, params[0], kernel_size=1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, params[1], kernel_size=1),
            nn.Conv2d(params[1], params[1], kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, params[2], kernel_size=1),
            nn.Conv2d(params[2], params[2], kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, params[3], kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self, in_channels, num_classes=5):
        super(GoogLeNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block3 = nn.Sequential(
            InceptionModule(192, [64, 128, 32, 32]),
            InceptionModule(256, [128, 192, 96, 64]),
        )

        self.block4 = nn.Sequential(
            InceptionModule(480, [192, 208, 48, 64]),
            InceptionModule(512, [160, 224, 64, 64]),
            InceptionModule(512, [128, 256, 64, 64]),
            InceptionModule(512, [112, 288, 64, 64]),
            InceptionModule(528, [256, 320, 128, 128]),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            InceptionModule(832, [256, 320, 128, 128]),
            InceptionModule(832, [384, 384, 128, 128]),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = x.view(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)

        return x
    
model = GoogLeNet(3, 5)
summary(model, (3, 224, 224))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

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

# 保存模型
torch.save(model.state_dict(), 'models/GoogLeNet.pth') 

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