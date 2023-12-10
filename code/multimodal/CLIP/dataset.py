import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer, BertModel

device = "cuda" if torch.cuda.is_available() else "cpu"

class CLIPDataset(Dataset):
    def __init__(self, img_dir, txt_dir, length=77):
        self.img_dir = img_dir
        self.txt_dir = txt_dir
        self.img_names = os.listdir(img_dir)
        self.txt_names = os.listdir(txt_dir)
        self.length = length
        self.transform = transforms.Compose([
            transforms.Resize((448, 448)),
            transforms.ToTensor()
        ])
        self.tokenizer = BertTokenizer.from_pretrained("D:/DL_Datasets/MSCOCO/bert-base-uncased", local_files_only=True)
        self.txt_encoder = BertModel.from_pretrained("D:/DL_Datasets/MSCOCO/bert-base-uncased", local_files_only=True)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        txt_name = self.txt_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        txt_path = os.path.join(self.txt_dir, txt_name)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        # toy example 只选取第一句话
        with open(txt_path, "r", encoding="utf-8") as f:
            txt = f.readlines()[0]

        encoding = self.tokenizer(txt, return_tensors="pt", padding="max_length", max_length=self.length, truncation=True)
        input_ids = encoding["input_ids"]
        mask = encoding["attention_mask"]
        return {
            "imgs": img.to(device),
            "input_ids": input_ids.squeeze(0).to(device),
            "masks": mask.squeeze(0).to(device)
        }

if __name__ == "__main__":
    img_dir = "D:/DL_Datasets/MSCOCO/train2014/imgs"
    txt_dir = "D:/DL_Datasets/MSCOCO/train2014/labels"
    dataset = CLIPDataset(img_dir, txt_dir)
    print(dataset[0]["imgs"].shape, "\n", dataset[0]["input_ids"], "\n", dataset[0]["masks"])
    # print(dataset[0]["imgs"].shape, "\n", dataset[0]["txts"].shape, "\n", dataset[0]["masks"].shape)
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    # batch = next(iter(dataloader))
    # 展示第一个
    # img = batch["imgs"][0]
    # txt = batch["txts"][0]
    # mask = batch["masks"][0]
    # print(txt)
    # print(img, "\n", txt, "\n", mask)



