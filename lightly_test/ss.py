import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from lightly.transforms import SimCLRTransform
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

random.seed(2023)

# Define the transformation pipeline
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])

transform = SimCLRTransform(input_size=100)


class CedarDataset(Dataset):
    def __init__(self, folder_path, split_ratio=0.7, train=True, transform=None):
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.train = train
        self.transform = transform

        # Get a list of all files in the folder
        all_files = sorted(os.listdir(folder_path))

        # Filter out non-image files based on file extensions
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        self.image_files = [file_name for file_name in all_files
                            if os.path.splitext(file_name)[1] in image_extensions]

        self.labels = [int(file_name.split("_")[1]) for file_name in self.image_files]
        unique_labels = list(set(self.labels))

        # Set the random seed
        # random.seed(seed)

        # Randomly shuffle the unique labels
        random.shuffle(unique_labels)

        split_index = int(self.split_ratio * len(unique_labels))
        self.train_labels = unique_labels[:split_index]
        self.test_labels = unique_labels[split_index:]

        self.train_indices = [file_name for file_name in self.image_files if
                              int(file_name.split("_")[1]) in self.train_labels]
        self.test_indices = [file_name for file_name in self.image_files if
                             int(file_name.split("_")[1]) in self.test_labels]

        self.train_samples = []
        self.test_samples = []

        random.shuffle(self.train_indices)
        random.shuffle(self.test_indices)

        self.compute_samples(indices=self.train_indices, isTrain=True)
        self.compute_samples(indices=self.test_indices)

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
                self.train_samples.append({"image": image, "label": label})
            else:
                self.test_samples.append({"image": image, "label": label})

    def __getitem__(self, index):
        samples = self.train_samples if self.train else self.test_samples
        sample = samples[index]
        return sample


image_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
cedar_dataset = CedarDataset(folder_path="/Users/mac/Downloads/data/CEDAR/full_org", split_ratio=0.7,
                             train=True, transform=image_transform_test)

data_loader = DataLoader(cedar_dataset.test_samples, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for x, y in data_loader:
    print(y)
# def get_unique_labels(dataset):
#     labels = set()
#
#     for _, label in dataset:
#         labels.add(label.item())
#
#     return labels
#
#
# # Assuming you have already created an instance of the ImageDataset class called 'dataset'
# train_labels = get_unique_labels(cedar_dataset.train_samples)
# test_labels = get_unique_labels(cedar_dataset.test_samples)
#
# print("Unique labels in training set:", train_labels)
# print("Unique labels in test set:", test_labels)

if __name__ == '__main__':
    print()
