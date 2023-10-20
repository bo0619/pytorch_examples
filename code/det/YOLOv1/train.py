import torch 
from torch import nn 
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets import YOLODataset
from model import YOLO
from loss import YOLOv1Loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    loop_train = tqdm(train_loader, leave=True)
    for _, (data, target) in enumerate(loop_train):
        input = data.to(device)
        target = target.to(device)
        output = model(input)

        criterion = YOLOv1Loss(S=7, B=2, C=4)
        loss_coord, loss_obj, loss_noobj, loss_class, loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        loop_train.set_description(f"Epoch [{epoch}/{100}]")
        loop_train.set_postfix(loss=float(loss), loss_coord=float(loss_coord), loss_obj=float(loss_obj), loss_noobj=float(loss_noobj), loss_class=float(loss_class))

        # print(f"loss_coord:{loss_coord}, loss_obj:{loss_obj}, loss_noobj:{loss_noobj}, loss_class:{loss_class}, loss:{loss}")
    train_loss /= len(train_loader.dataset)
    print(f"train_loss:{train_loss}")

def validate(model, val_loader):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            input = data.to(device)
            target = target.to(device)
            output = model(input)

            criterion = YOLOv1Loss(S=7, B=2, C=4)
            _, _, _, _, loss = criterion(output, target)
            val_loss += loss.item()

    val_loss /= len(val_loader.dataset)
    print(f"val_loss:{val_loss}")


if __name__ == "__main__":
    train_dataset = YOLODataset(img_path='data/images/train', label_path='data/labels/train', S=7, B=2, C=4, transforms=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    val_dataset = YOLODataset(img_path='data/images/val', label_path='data/labels/val', S=7, B=2, C=4, transforms=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    model = YOLO(S=7, B=2, num_classes=4)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        print(f"Epoch {epoch}")
        train(model, train_loader, optimizer, epoch)
        validate(model, val_loader)
        if (epoch+1) % 25 == 0:
            torch.save(model.state_dict(), f"model_{epoch+1}.pth")