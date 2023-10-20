import os
import torch 
from torch import nn 
from datasets import YOLODataset

class YOLOv1Loss(nn.Module):
    def __init__(self, S, B, C, lambda_coord=5, lambda_noobj=0.5):
        super(YOLOv1Loss, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    @staticmethod
    def mse(pred, target):
        return (pred - target)**2
    
    def compute_iou(self, bbox1, bbox2):
            # bbox: x y w h
        bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()

        area1 = bbox1[2] * bbox1[3]  # bbox1's area
        area2 = bbox2[2] * bbox2[3]  # bbox2's area

        max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
        min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
        max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
        min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

        if max_left >= min_right or max_top >= min_bottom:
            return 0
        else:
            # iou = intersect / union
            intersect = (min_right - max_left) * (min_bottom - max_top)
            return intersect / (area1 + area2 - intersect)

    def forward(self, pred, target):
        """
        pred: (batch_size, S, S, C+5B)
        target: (batch_size, S, S, C+5B)
        """
        loss_coord_xy = 0
        loss_coord_wh = 0
        loss_obj = 0
        loss_noobj = 0
        loss_class = 0

        pred = pred.reshape(-1, self.S, self.S, self.C+5*self.B)
        target = target.reshape(-1, self.S, self.S, self.C+5*self.B)

        batch_size = pred.shape[0]

        for b in range(batch_size):
            for y in range(self.S):
                for x in range(self.S):
                    if target[b, y, x, 0] == 1:
                        bbox1 = torch.tensor(
                            [pred[b, y, x, 1], pred[b, y, x, 2], pred[b, y, x, 3], pred[b, y, x, 4]]
                        )
                        bbox2 = torch.tensor(
                            [target[b, y, x, 6], target[b, y, x, 7], target[b, y, x, 8], target[b, y, x, 9]]
                        )
                        gt = torch.tensor(
                            [target[b, y, x, 1], target[b, y, x, 2], target[b, y, x, 3], target[b, y, x, 4]]
                        )

                        iou1 = self.compute_iou(bbox1, gt)
                        iou2 = self.compute_iou(bbox2, gt)

                        if iou1 > iou2:
                            loss_coord_xy += self.lambda_coord * torch.sum(self.mse(pred[b, y, x, 1:3], target[b, y, x, 1:3]))
                            loss_coord_wh += self.lambda_coord * torch.sum(self.mse(pred[b, y, x, 3:5].sqrt(), target[b, y, x, 3:5].sqrt()))
                            loss_obj += torch.sum(self.mse(pred[b, y, x, 0], iou1))
                            loss_noobj += self.lambda_noobj * torch.sum(self.mse(pred[b, y, x, 5], 0))
                        
                        else:
                            loss_coord_xy += self.lambda_coord * torch.sum(self.mse(pred[b, y, x, 6:8], target[b, y, x, 6:8]))
                            loss_coord_wh += self.lambda_coord * torch.sum(self.mse(pred[b, y, x, 8:10].sqrt(), target[b, y, x, 8:10].sqrt()))
                            loss_obj += torch.sum(self.mse(pred[b, y, x, 5], iou2))
                            loss_noobj += self.lambda_noobj * torch.sum(self.mse(pred[b, y, x, 0], 0))

                        loss_class += torch.sum(self.mse(pred[b, y, x, 10:], target[b, y, x, 10:]))
                    
                    else:
                        loss_noobj += self.lambda_noobj * torch.sum(self.mse(pred[b, y, x, [0, 5]], 0))

                    loss = loss_coord_xy + loss_coord_wh + loss_obj + loss_noobj + loss_class
        
        return (loss_coord_xy + loss_coord_wh) / batch_size, \
                loss_obj / batch_size, \
                loss_noobj / batch_size, \
                loss_class / batch_size, \
                loss / batch_size

if __name__ == "__main__":
    dataset = YOLODataset(img_path='data/images/train', label_path='data/labels/train', S=7, B=2, C=4)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for img, label in dataloader:
        print(img.shape)
        print(label.shape)
        break

    model = YOLOv1Loss(S=7, B=2, C=4)
    pred = torch.randn(1, 7, 7, 14)
    loss_coord, loss_obj, loss_noobj, loss_class, loss = model(pred, label)
    print(loss_coord, loss_obj, loss_noobj, loss_class, loss)