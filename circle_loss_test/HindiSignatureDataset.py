import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
from torch.utils.data import random_split, DataLoader


# random.seed(2023)


def get_augmented_positive(genuine_image):
    aug_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.CenterCrop((100, 170)),
        # transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor()
    ])
    return aug_transform(genuine_image)


# image = Image.open("/Users/mac/Downloads/data/BHSig260/Hindi/001/H-S-1-F-01.tif")
# augmented = get_augmented_positive(image)
# result = augmented.permute(1, 2, 0)
# plt.imshow(result, cmap='gray')
# plt.show()
# plt.imshow(image, cmap='gray')
# plt.show()


class HindiSignatureDataset(Dataset):
    def __init__(self, imageFolder: ImageFolder, K_train: int = 100, K_test: int = 60, isTrain=False, transform=None):
        super(HindiSignatureDataset, self).__init__()
        self.imageFolder = imageFolder
        self.forged_count = 30
        self.genuine_count = 24
        self.transform = transform
        self.sub_folders = random.sample(self.imageFolder.classes, K_train + K_test)

        # Divide the sub-folders into train and test sets
        self.train_sub_folders = self.sub_folders[:K_train]
        print(f"number of train signers: {len(self.train_sub_folders)}")
        self.test_sub_folders = self.sub_folders[K_train:]
        print(f"number of test signers: {len(self.test_sub_folders)}")
        dataset_type = "train" if isTrain else "test"
        print(f"dataset type: {dataset_type}")

        self.train_samples = []
        self.test_samples = []
        self.isTrain = isTrain

        for sub_folder in self.train_sub_folders:
            mclass_index = self.imageFolder.class_to_idx[sub_folder]
            image_paths = [imageFolder.imgs[i][0] for i in range(len(imageFolder)) if
                           imageFolder.imgs[i][1] == mclass_index]
            self.train_samples.extend(image_paths)

        for sub_folder in self.test_sub_folders:
            mclass_index = self.imageFolder.class_to_idx[sub_folder]
            image_paths = [imageFolder.imgs[i][0] for i in range(len(imageFolder)) if
                           imageFolder.imgs[i][1] == mclass_index]
            self.test_samples.extend(image_paths)

    def __getitem__(self, index):
        used_samples = self.train_samples if self.isTrain else self.test_samples
        sub_folder_path = used_samples[index]

        # Get the list of images in the selected sub-folder
        sub_folder_images = [
            path for path, _ in self.imageFolder.imgs
            if sub_folder_path in path
        ]

        # Separate forged and genuine samples
        forged_samples = sub_folder_images[:self.forged_count]
        genuine_samples = sub_folder_images[self.forged_count:]

        forged_choice = random.choice(forged_samples)
        genuine_choice = random.choice(genuine_samples)

        forged_image = Image.open(forged_choice)
        genuine_image = Image.open(genuine_choice)
        augmented_genuine_image = get_augmented_positive(genuine_image)

        if self.transform is not None:
            genuine_image = self.transform(genuine_image)
            forged_image = self.transform(forged_image)
        return genuine_image, augmented_genuine_image, forged_image

    def __len__(self):
        return len(self.train_samples) if self.isTrain else len(self.test_samples)


data_transform = transforms.Compose([
    transforms.Resize((100, 170)),
    transforms.ToTensor()
])

hindi_folder = ImageFolder(root="/Users/mac/Downloads/data/BHSig260/Hindi")
hindiDataset = HindiSignatureDataset(imageFolder=hindi_folder, transform=None, isTrain=True)

train_loader = DataLoader(hindiDataset, batch_size=64, shuffle=True)
test_loader = DataLoader(hindiDataset, batch_size=64, shuffle=False)
print(f"size: {len(hindiDataset)}")
print(len(train_loader))
print(len(test_loader))
if __name__ == '__main__':
    print()
