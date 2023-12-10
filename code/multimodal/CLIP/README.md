# OpenAI's CLIP model implemented using PyTorch
## Model Implementation
The text encoder is huggingface's bert-base-uncased, the image encoder is timm's resnet-50.
## Dataset Description
The dataset we use is [MSCOCO's 2014 Train Images](http://images.cocodataset.org/zips/train2014.zip).

Here is an example of the dataset:

img:
![img] (https://github.com/bo0619/pytorch_examples/blob/master/code/multimodal/CLIP/COCO_train2014_000000000009.jpg)

txt: Closeup of bins of food that include broccoli and bread.
