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

            if isTrain:
                self.train_samples.append((image, label))
            else:
                self.test_samples.append((image, label))

    def __getitem__(self, index):
        samples = self.train_samples if self.train else self.test_samples
        sample = samples[index]
        return sample


class HindiDataset(Dataset):
    def __init__(self, folder_path, split_ratio=0.7, train=True, transform=None):
        self.folder_path = folder_path
        self.split_ratio = split_ratio
        self.train = train
        self.transform = transform

        random.seed(2023)
        # Get a list of all subfolders in the main folder
        subfolders = sorted(os.listdir(folder_path))
        random.shuffle(subfolders)

        self.train_samples = []
        self.test_samples = []

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Get a list of all files in the subfolder
            image_files = sorted(os.listdir(subfolder_path))

            # Split the image files into train and test sets based on split_ratio
            split_index = int(self.split_ratio * len(image_files))
            train_files = image_files[:split_index]
            test_files = image_files[split_index:]

            if self.train:
                self.compute_samples(train_files, subfolder, self.train_samples)
            else:
                self.compute_samples(test_files, subfolder, self.test_samples)

        if self.train:
            random.shuffle(self.train_samples)

    def __len__(self):
        if self.train:
            return len(self.train_samples)
        else:
            return len(self.test_samples)

    def compute_samples(self, files, label, sample_list):
        for file_name in files:
            image_path = os.path.join(self.folder_path, label, file_name)
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)

            sample_list.append((image, int(label)))

    def __getitem__(self, index):
        samples = self.train_samples if self.train else self.test_samples
        sample = samples[index]
        return sample


image_transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# cedar_dataset = CedarDataset(folder_path="/Users/mac/Downloads/data/CEDAR/full_org", split_ratio=0.7,
#                              train=True, transform=image_transform_test)

# hindi_dataset = HindiDataset(folder_path="/Users/mac/Downloads/data/BHSig260/Hindi", split_ratio=0.7,
#                              train=True, transform=image_transform_test)
# hindi_dataset_test = HindiDataset(folder_path="/Users/mac/Downloads/data/BHSig260/Hindi", split_ratio=0.7,
#                                   train=False, transform=image_transform_test)
#
# print(len(hindi_dataset_test))
# print(hindi_dataset[5]['label'])
#
# data_loader = DataLoader(hindi_dataset, batch_size=64, shuffle=False)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# for x, y in data_loader:
#     print(y)
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
class CombinedDataset(Dataset):
    def __init__(self, cedar_dataset, hindi_dataset, bengali_dataset):
        self.cedar_dataset = cedar_dataset
        self.hindi_dataset = hindi_dataset
        self.bengali_dataset = bengali_dataset

    def __len__(self):
        return len(self.cedar_dataset) + len(self.hindi_dataset) + len(self.bengali_dataset)

    def __getitem__(self, index):
        if index < len(self.cedar_dataset):
            return self.cedar_dataset[index]
        elif index < len(self.cedar_dataset) + len(self.hindi_dataset):
            return self.hindi_dataset[index - len(self.cedar_dataset)]
        else:
            return self.bengali_dataset[index - len(self.cedar_dataset) - len(self.hindi_dataset)]


hindi_dataset = HindiDataset(folder_path="/Users/mac/Downloads/data/BHSig260/Hindi", split_ratio=1,
                             train=True, transform=image_transform_test)
bengali_dataset = HindiDataset(folder_path="/Users/mac/Downloads/data/BHSig260/Bengali", split_ratio=1,
                               train=True, transform=image_transform_test)
cedar_dataset = CedarDataset(folder_path="/Users/mac/Downloads/data/CEDAR", split_ratio=1,
                             train=True, transform=image_transform_test)

combined_dataset = CombinedDataset(cedar_dataset, hindi_dataset, bengali_dataset)
print(f"the length of the combined is: {len(combined_dataset)}")

num_samples = 0
sig_dataloader = DataLoader(combined_dataset, batch_size=128, num_workers=12, shuffle=True)
device = ""
mean = torch.zeros(3).to(device)
std = torch.zeros(3).to(device)
for data in sig_dataloader:
    for image in data:
        image = image.to(device)
        batch_size = image.size(0)
        num_samples += batch_size
        channels = image.size(1)
        height = image.size(2)
        width = image.size(3)

        reshaped_image = image.view(batch_size, channels, -1)

        mean += reshaped_image.mean(dim=2).sum(dim=0)
        std += reshaped_image.pow(2).mean(dim=2).sum(dim=0)

mean /= num_samples
std = torch.sqrt(std / num_samples - mean.pow(2))

print("Mean:", mean)
print("Standard Deviation:", std)

if __name__ == '__main__':
    print()
