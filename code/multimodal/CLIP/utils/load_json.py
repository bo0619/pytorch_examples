import json

data = json.load(open('annotations/captions_train2014.json'))
image_captions = {}
for annotation in data['annotations']:
    image_id = annotation['image_id']
    caption = annotation['caption']
    image_filename = f"image{image_id}.txt"
    if image_filename not in image_captions:
        image_captions[image_filename] = []
    image_captions[image_filename].append(caption)

# 将标注保存为txt文件
for filename, captions in image_captions.items():
    with open("train2014/labels/"+filename, 'w') as file:
        for caption in captions:
            file.write(f"{caption}\n")
print("Done!")
