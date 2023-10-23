import os
import random

import PIL
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, models
from byol_pytorch import BYOL


class HindiDataset(Dataset):
    def __init__(self, folder_path, split_ratio=0.7, train=True, transform=None):
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.train = train
        self.transform = transform

        # random.seed(2023)
        # Get a list of all subfolders in the main folder
        subfolders = sorted(os.listdir(folder_path))
        random.shuffle(subfolders)

        self.all_files = []
        self.labels = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif']

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue
            image_files = sorted(os.listdir(subfolder_path))[30:]
            for file_name in image_files:
                file_path = os.path.join(subfolder_path, file_name)
                _, ext = os.path.splitext(file_name)
                if ext.lower() not in image_extensions:
                    continue
                self.all_files.append(file_path)
                self.labels.append(subfolder)

        # Randomly shuffle the image files and labels together
        zipped_data = list(zip(self.all_files, self.labels))
        random.shuffle(zipped_data)
        self.all_files, self.labels = zip(*zipped_data)

        # Split the data based on the split ratio
        split_index = int(self.split_ratio * len(self.all_files))
        self.train_image_files = self.all_files[:split_index]
        self.train_labels = self.labels[:split_index]

        self.test_image_files = self.all_files[split_index:]
        self.test_labels = self.labels[split_index:]

        self.train_samples = []
        self.test_samples = []

        self.compute_samples(indices=self.train_image_files, isTrain=True)
        self.compute_samples(indices=self.test_image_files)

    def __len__(self):
        if self.train:
            return len(self.train_samples)
        else:
            return len(self.test_samples)

    def compute_samples(self, indices, isTrain=False):
        for index in indices:
            label = int(index.split("_")[1])
            image_path = os.path.join(self.folder_path, index)
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            label_tensor = torch.tensor(label)
            if isTrain:
                self.train_samples.append((image, label_tensor))
            else:
                self.test_samples.append((image, label_tensor))

    def __getitem__(self, index):
        samples = self.train_samples if self.train else self.test_samples
        image, label = samples[index]
        return image, label


# import numpy as np
#
# # Assuming 'image' is the image array
# image = PIL.Image.open("/Users/mac/Downloads/my_training/same/BHSig260/Bengali/001/B-S-1-F-01.tif")
# image_array = np.array(image)
# mean = np.mean(image_array)
# std = np.std(image_array)
#
#
# print("Mean:", mean)
# print("Standard Deviation:", std)
# class_weights = torch.tensor([1.25])
# print(class_weights[0])
# loss_function = nn.BCEWithLogitsLoss(pos_weight=class_weights[0])

resnet = models.resnet18(pretrained=True)
backbone = nn.Sequential(
    *list(resnet.children())[:-1],
    nn.AdaptiveAvgPool2d((1, 1))
)

learner = BYOL(
    backbone,
    image_size=256,
    projection_size=256,
    projection_hidden_size=4096,
    moving_average_decay=0.99
    # hidden_layer='avgpool'
)
print(learner)

if __name__ == '__main__':
    print()
