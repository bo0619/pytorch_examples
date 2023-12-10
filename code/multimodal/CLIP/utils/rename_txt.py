import os

# 设置文件夹路径
folder_path = 'train2014/labels/'

# 获取文件夹中所有文件名
file_list = os.listdir(folder_path)

# 重命名文件
for filename in file_list:
    if filename.startswith("image"):
        # 提取数字部分
        number = filename.split("image")[1].split(".")[0]
        new_name = f"COCO_train2014_{int(number):012d}.txt"  # 使用字符串格式化确保总长度为12个字符

        # 旧文件路径
        old_file_path = os.path.join(folder_path, filename)
        # 新文件路径
        new_file_path = os.path.join(folder_path, new_name)
        # 重命名文件
        os.rename(old_file_path, new_file_path)

print("Done!")

# val2014\imgs\COCO_val2014_000000000042.jpg
# val2014\labels\COCO_val2014_000000000042.txt

# train2014\imgs\COCO_train2014_000000000009.jpg
# train2014\labels\COCO_train2014_000000000009.txt