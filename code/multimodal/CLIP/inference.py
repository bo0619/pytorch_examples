import torch
from PIL import Image
from torchvision import transforms
from transformers import BertTokenizer
from model import CLIP

# 假设你有一个训练好的 CLIP 模型
clip_model = CLIP(img_dim=2048, text_dim=768, proj_dim=128)
# 加载你训练好的模型权重
clip_model.load_state_dict(torch.load("CLIP-Model/res50-bert-2.pth"))
clip_model.eval()
# 编码图片和文本的训练好的模型
img_encoder = clip_model.image_encoder
txt_encoder = clip_model.text_encoder
img_proj = clip_model.img_proj
txt_proj = clip_model.txt_proj


# 读取图片
img = Image.open("123.jpg").convert("RGB")
# 编码图片
img = transforms.Resize((448, 448))(img)
img = transforms.ToTensor()(img)
img = img.unsqueeze(0)
img = img_encoder(img)
img = img_proj(img)

txts = []
labels = ["cat", "dog", "horse", "tiger", "pig", "bird", "car"]
for label in labels:
    txts.append("this is a photo of a " + label)


probs = []
for txt in txts:
    # 编码文本
    tokenizer = BertTokenizer.from_pretrained("D:/DL_Datasets/MSCOCO/bert-base-uncased", local_files_only=True)
    input_ids = tokenizer(txt, return_tensors="pt", padding="max_length", max_length=32, truncation=True)["input_ids"].squeeze(0)
    masks = tokenizer(txt, return_tensors="pt", padding="max_length", max_length=32, truncation=True)["attention_mask"].squeeze(0)
    # print(input_ids.shape)
    txt = txt_encoder(input_ids.unsqueeze(0).long(), masks.unsqueeze(0).long())
    txt = txt_proj(txt)

    # 计算图片和文本的相似度
    sim = torch.matmul(img, txt.T)
    probs.append(sim)

logits = torch.softmax(torch.cat(probs, dim=1), dim=1)
print(logits)
true_label = labels[torch.argmax(logits, dim=1)]
print(true_label)