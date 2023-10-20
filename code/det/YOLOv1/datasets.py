import os 
import cv2 
import matplotlib.pyplot as plt
import numpy as np
import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class YOLODataset(Dataset):
    def __init__(self, img_path, label_path, S, B, C, transforms=None):
        self.img_path = img_path
        self.label_path = label_path
        self.S = S
        self.B = B
        self.C = C
        self.transforms = transforms
        self.imgs = os.listdir(self.img_path)
        self.labels = os.listdir(self.label_path)

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.img_path, self.imgs[idx]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transforms:
            img = self.transforms(img)
        else:
            img = torch.from_numpy(img).float()
        
        cxywh = []
        with open(os.path.join(self.label_path, self.labels[idx]), 'r') as f:
            for line in f.readlines():
                line = line.strip().split(' ')
                cxywh.append([int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])])

        label = self.cxywh2label(cxywh, self.S, self.B, self.C)
        return img, label

    def cxywh2label(self, bboxes, S, B, C):
        """
        [(c,x,y,w,h), (c,x,y,w,h), ...]
        """
        label = np.zeros((S, S, C + 5*B))
        for c, x, y, w, h in bboxes:
            cell_x = int(S * x)
            cell_y = int(S * y)
            label[cell_y, cell_x, 0:5] = np.array([c, x, y, w, h])
            label[cell_y, cell_x, 5:10] = np.array([c, x, y, w, h])
            label[cell_y, cell_x, 10+c] = 1
        return label
        

if __name__ == "__main__":
    dataset = YOLODataset(img_path='data/images/train', label_path='data/labels/train', S=7, B=2, C=4, transforms=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    for img, label in dataloader:
        plt.imshow(img[0].permute(1, 2, 0))
        plt.show()
        print(label.shape)
        print(label[0, 0, 0:5])
        print(label[0, 0, 5:10])
        print(label[0, 0, 10:])
        break