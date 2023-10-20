import warnings
warnings.filterwarnings("ignore")

import torch 
from torch import nn 
from torch.nn import functional as F
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(0)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.act = nn.LeakyReLU(negative_slope=0.1)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class YOLO(nn.Module):
    def __init__(self, S=7, B=2, num_classes=4):
        super(YOLO, self).__init__()
        self.S = S
        self.B = B
        self.num_classes = num_classes

        self.conv_1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.pool_4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.fc_1 = nn.Linear(in_features=512*1*1, out_features=512)
        self.fc_2 = nn.Linear(in_features=512, out_features=(self.S * self.S*(self.num_classes + B*5)))

        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.pool_1(self.conv_1(x))
        x = self.pool_2(self.conv_2(x))
        x = self.pool_3(self.conv_3(x))
        x = self.pool_4(self.conv_4(x))
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = self.sig(x)
        return x
    


if __name__ == "__main__":
    model = YOLO(num_classes=4).to(device)
    summary(model, (3, 960, 1280))
    img = torch.randn(1, 3, 960, 1280).to(device)
    pred = model(img)
    print(pred.shape)
