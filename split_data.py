import os
import random
import shutil

def split_dataset(dataset_folder, train_ratio=0.8, seed=42):
    random.seed(seed)
    
    # 获取所有子文件夹（类别）
    categories = os.listdir(dataset_folder)
    
    for category in categories:
        category_folder = os.path.join(dataset_folder, category)
        if os.path.isdir(category_folder):
            images = os.listdir(category_folder)
            num_images = len(images)
            
            # 计算用于训练集的图像数量
            num_train = int(num_images * train_ratio)
            
            # 随机打乱图像顺序
            random.shuffle(images)
            
            # 划分训练集和测试集
            train_images = images[:num_train]
            test_images = images[num_train:]
            
            # 创建保存训练集和测试集的文件夹
            train_folder = os.path.join(dataset_folder, 'train', category)
            test_folder = os.path.join(dataset_folder, 'test', category)
            
            os.makedirs(train_folder, exist_ok=True)
            os.makedirs(test_folder, exist_ok=True)
            
            # 将图像复制到相应的文件夹
            for image in train_images:
                src_path = os.path.join(category_folder, image)
                dest_path = os.path.join(train_folder, image)
                shutil.copy(src_path, dest_path)
                
            for image in test_images:
                src_path = os.path.join(category_folder, image)
                dest_path = os.path.join(test_folder, image)
                shutil.copy(src_path, dest_path)

# 指定数据集文件夹路径
dataset_folder = 'vaeimage'

# 划分数据集，指定训练集比例和随机种子
split_dataset(dataset_folder, train_ratio=0.8, seed=42)


