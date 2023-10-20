import cv2
import torch
from datasets import YOLODataset
from model import YOLO

S = 7
B = 2
C = 4
class_names = ["1", "2", "3", "4"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model = YOLO(S=S, B=B, num_classes=C)
# model.load_state_dict(torch.load('model_50.pth'))
# model.to(device)
# model.eval()



# dataset = YOLODataset(img_path='data/images/test', label_path='data/labels/test', S=S, B=B, C=C)
# img, target = dataset[0]
# img = img.permute(2, 0, 1).unsqueeze(0).to(device)
# output = model(img).reshape(S, S, C+5*B)
# print(output.detach().cpu().numpy())

def draw_bbox(img, bboxes, class_names):
    h, w = img.shape[:2]
    n = bboxes.shape[0]
    bboxes = bboxes.detach().cpu().numpy()
    print(bboxes)
    for i in range(n):
        p1 = (int((bboxes[i, 1] - bboxes[i, 3]/2) * w), int((bboxes[i, 2] - bboxes[i, 4]/2) * h))
        p2 = (int((bboxes[i, 1] + bboxes[i, 3]/2) * w), int((bboxes[i, 2] + bboxes[i, 4]/2) * h))
        class_name = class_names[int(bboxes[i, 0])]
        cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def predict_img(img, model, input_size, S, B, C, conf_threshold=0.3, iou_threshold=0.5):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_img = torch.Tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    pred = model(pred_img)[0].detach().cpu().reshape(S, S, C+5*B)
    cxywh = pred2cxywh(pred, S, C, conf_threshold, iou_threshold)
    return cxywh

def pred2cxywh(pred, S, C, conf_threshold=0.3, iou_threshold=0.5):
    bboxes = torch.zeros((S * S, 5 + C))
    print(pred.shape)
    for x in range(S):
        for y in range(S):
            conf1 = pred[x, y, 0]
            conf2 = pred[x, y, 5]
            if conf1 > conf2:
                bboxes[(x * S + y), 0] = pred[x, y, 0]
                bboxes[(x * S + y), 1:5] = torch.Tensor([pred[x, y, 1], pred[x, y, 2], pred[x, y, 3], pred[x, y, 4]])
                bboxes[(x * S + y), 5:] = torch.Tensor(pred[x, y, 10:])
            else:
                bboxes[(x * S + y), 0] = pred[x, y, 5]
                bboxes[(x * S + y), 1:5] = torch.Tensor([pred[x, y, 6], pred[x, y, 7], pred[x, y, 8], pred[x, y, 9]])
                bboxes[(x * S + y), 5:] = torch.Tensor(pred[x, y, 10:])

    cxywh = nms(bboxes, C, conf_threshold, iou_threshold)
    return cxywh

def nms(bboxes, C, conf_threshold=0.1, iou_threshold=0.3):
    bbox_prob = bboxes[:, 5:].clone().detach()
    bbox_conf = bboxes[:, 0].clone().detach().unsqueeze(1).expand_as(bbox_prob)
    bbox_cls_spec_conf = bbox_conf * bbox_prob
    bbox_cls_spec_conf[bbox_cls_spec_conf < conf_threshold] = 0

    for c in range(C):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices
        for i in range(bboxes.shape[0]):
            if bbox_cls_spec_conf[rank[i], c] == 0:
                continue
            for j in range(i+1, bboxes.shape[0]):
                if bbox_cls_spec_conf[rank[j], c] != 0:
                    iou = calculate_iou(bboxes[rank[i], 1:5], bboxes[rank[j], 1:5])
                    if iou > iou_threshold:
                        bbox_cls_spec_conf[rank[j], c] = 0
    
    bboxes = bboxes[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    ret = torch.ones((bboxes.shape[0], 6))

    if bboxes.size()[0] == 0:
        return torch.tensor([])
    
    ret[:, 1:5] = bboxes[:, 1:5]
    ret[:, 0] = torch.argmax(bbox_cls_spec_conf, dim=1)
    ret[:, 5] = torch.max(bbox_cls_spec_conf, dim=1).values
    return ret

def calculate_iou(bbox1, bbox2):
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



if __name__ == "__main__":
    model = YOLO(S=S, B=B, num_classes=C).to(device)
    model.load_state_dict(torch.load('model_50.pth'))
    print("model loaded")

    img = cv2.imread('data/images/test/20230716125025_1.jpg')
    cxywh = predict_img(img, model, 448, S, B, C)
    if cxywh.size()[0] == 0:
        print("no bbox")
    else:
        img = draw_bbox(img, cxywh, class_names)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()