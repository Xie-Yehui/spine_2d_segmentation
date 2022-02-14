import torch
import cv2
import os
import glob
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np


class ISBI_Loader(Dataset):
    def __init__(self, image_pathh, label_pathh):
        self.image_list = []
        if not isinstance(image_pathh, list):
            image_pathh = [image_pathh]
        for image_path_ in image_pathh:
            self.image_list.extend(glob.glob(os.path.join(image_path_, '*.png')))
        self.image_list.sort()

        self.label_list = []
        if not isinstance(label_pathh, list):
            label_pathh = [label_pathh]
        for label_path_ in label_pathh:
            self.label_list.extend(glob.glob(os.path.join(label_path_, '*.png')))
        self.label_list.sort()

    def __getitem__(self, index):

        image_path = self.image_list[index]
        label_path = self.label_list[index]

        # 读取灰度图像
        image = cv2.imread(image_path, 0)
        label = cv2.imread(label_path, 0)

        image = image.astype(np.uint8)
        label = label.astype(np.uint8)
        image = cv2.resize(image, (256, 256), 0)
        label = cv2.resize(label, (256, 256), 0)

        # 归一化
        image = image / 255.0
        label = np.where(label > 0, 1, 0)

        image = image.astype(np.float32)
        image = transforms.ToTensor()(image)
        label = label.astype(np.float32)
        label = transforms.ToTensor()(label)

        return image, label

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    image_pathh = '../data/train/image/*'
    label_pathh = '../data/train/label/*'
    train_dataset = ISBI_Loader(image_pathh, label_pathh)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=6, shuffle=True)
    print("训练集数据个数：", len(train_dataset))
    for image, label in train_loader:
        print("image.shape: {}".format(image.shape))
        print("label.shape: {}".format(label.shape))
        break

    image_pathh = '../data/val/image/*'
    label_pathh = '../data/val/label/*'
    val_dataset = ISBI_Loader(image_pathh, label_pathh)
    print("验证集数据个数：", len(val_dataset))

    image_pathh = '../data/test/image/*'
    label_pathh = '../data/test/label/*'
    test_dataset = ISBI_Loader(image_pathh, label_pathh)
    print("验证集数据个数：", len(test_dataset))
