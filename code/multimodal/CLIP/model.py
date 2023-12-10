import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertModel, BertTokenizer
from timm import create_model
from dataset import CLIPDataset

proj_dim = 128

img_dir = "train2014/imgs"
txt_dir = "train2014/labels"
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = create_model("resnet50", pretrained=True, pretrained_cfg_overlay=dict(file="D:/DL_Datasets/MSCOCO/resnet50/resnet50.pth"), num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.model(x)
    

class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = BertModel.from_pretrained('D:/DL_Datasets/MSCOCO/bert-base-uncased', local_files_only=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.target_token_idx = 0

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, self.target_token_idx, :]

# input_ids = torch.Tensor([[[  101,  2485,  6279,  1997,  8026,  2015,  1997,  2833,  2008,  2421,
#           22953, 21408,  3669,  1998,  7852,  1012,   102,     0,     0,     0, 
#               0,     0,     0,     0,     0,     0,     0,     0,     0,     0, 
#               0,     0]]])
# attention_mask = torch.Tensor([[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 
#           0, 0, 0, 0, 0, 0, 0, 0, 0]]])
# text_encoder = TextEncoder()

# print(text_encoder(input_ids.long(), attention_mask.long()).shape)


class ProjectionHead(nn.Module):
    def __init__(self, embedding_dim, proj_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, proj_dim)
        self.fc2 = nn.Linear(proj_dim, proj_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out1 = self.fc1(x)
        out2 = self.relu(out1)
        out3 = self.dropout(out2)
        out4 = self.fc2(out3)
        return out4
    
class CLIP(nn.Module):
    def __init__(self, img_dim, text_dim, proj_dim):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.img_proj = ProjectionHead(img_dim, proj_dim) # [2048, 128]
        self.txt_proj = ProjectionHead(text_dim, proj_dim) # [768, 128]
        self.temp = 0.07

    # def forward(self, batch):
    #     img_features = self.image_encoder(batch["imgs"])
    #     txt_features = self.text_encoder(batch["input_ids"], attention_mask=batch["masks"])
        
    #     img_proj = self.img_proj(img_features)
    #     txt_proj = self.txt_proj(txt_features)

    #     logits = torch.matmul(img_proj, txt_proj.T) / self.temp

    #     imgs_sim = torch.matmul(img_proj, img_proj.T) / self.temp
    #     txts_sim = torch.matmul(txt_proj, txt_proj.T) / self.temp

    #     targets = F.softmax(imgs_sim + txts_sim, dim=-1) / 2 * self.temp

    #     txt_loss = self.cross_entropy(logits, targets)
    #     img_loss = self.cross_entropy(logits.T, targets.T)
    #     loss = (txt_loss + img_loss) / 2
    #     return loss.mean()
    
    # def cross_entropy(self, logits, targets):
    #     log_softmax = nn.LogSoftmax(dim=-1)
    #     loss = (-targets * log_softmax(logits)).sum(1)
    #     return loss
    
    def forward(self, batch):
        # Encode the image and text
        image_features = self.image_encoder(batch["imgs"])
        text_features = self.text_encoder(batch["input_ids"], attention_mask=batch["masks"])

        # Project the image and text features
        image_features = self.img_proj(image_features)
        text_features = self.txt_proj(text_features)

        # Calculate the cosine similarity
        similarity = F.cosine_similarity(image_features, text_features)

        # Compute cross-entropy losses
        logits_t = torch.mm(image_features, text_features.T)
        logits_i = torch.mm(text_features, image_features.T)

        targets = torch.arange(logits_t.shape[0]).to(logits_t.device)
        loss_i = F.cross_entropy(logits_i, targets, reduction="mean")
        loss_t = F.cross_entropy(logits_t, targets, reduction="mean")

        # Combine losses
        loss = (loss_i + loss_t) / 2

        return loss

