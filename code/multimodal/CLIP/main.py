import os
import warnings
warnings.filterwarnings("ignore")
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import CLIPDataset
from model import CLIP

if not os.path.exists("./CLIP-Model"):
    os.makedirs("./CLIP-Model")

proj_dim = 128
num_epochs = 3

device = "cuda" if torch.cuda.is_available() else "cpu"

img_dir = "D:/DL_Datasets/MSCOCO/train2014/imgs"
txt_dir = "D:/DL_Datasets/MSCOCO/train2014/labels"
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor()
])

dataset = CLIPDataset(img_dir, txt_dir, length=20)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

model = CLIP(img_dim=2048, text_dim=768, proj_dim=proj_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

losses = []
for epoch in range(num_epochs):
    loop = tqdm(dataloader)
    epoch_loss = 0.0  # 初始化每个epoch的总体损失
    num_batches = len(dataloader)

    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()  # 累积每个batch的损失值到总体损失

        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())

    avg_epoch_loss = epoch_loss / num_batches
    losses.append(avg_epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss}")

    torch.save(model.state_dict(), f"./CLIP-Model/res50-bert-{epoch}.pth")



###############################################################
#                                                             #
#                      numpy pseudo code                      #  
#                                                             #
###############################################################

# imgs, txts = next(iter(dataloader))

# # extract feature representations of each modality
# I_f = image_encoder(imgs).reshape(imgs.shape[0], -1) # I_f        # [32, 43264]
# T_f = language_encoder(txts).reshape(txts.shape[0], -1) # T_f     # [32, 5120]


# # W_i
# W_i = nn.Parameter(torch.randn(I_f.shape[1], proj_dim)) # [43264, 128]
# # W_t
# W_t = nn.Parameter(torch.randn(T_f.shape[1], proj_dim)) # [5120, 128]


# # joint multimodal embedding 
# I_e = F.normalize(I_f @ W_i, dim=1) # [32, 128]
# T_e = F.normalize(T_f @ W_t, dim=1) # [32, 128]


# # scaled pairwise cosine similarities
# temp = nn.Parameter(torch.rand(1)) # [1]
# logits = torch.exp(temp) * I_e @ T_e.T # [32, 32]

# # symmetric loss function
# labels = torch.arange(logits.shape[0]).long() # [32]
# loss_i = F.cross_entropy(logits, labels) # [32]
# loss_t = F.cross_entropy(logits.T, labels) # [32]
# loss = (loss_i + loss_t) / 2.0 # [32]
# print(loss)
