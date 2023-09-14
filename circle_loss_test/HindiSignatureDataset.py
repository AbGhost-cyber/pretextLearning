import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random

random.seed(2023)


def get_augmented_positive(genuine_image):
    aug_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
        transforms.CenterCrop((110, 200)),
        transforms.GaussianBlur(kernel_size=5),
        transforms.ToTensor()
    ])
    return aug_transform(genuine_image)


image = Image.open("/Users/mac/Downloads/data/BHSig260/Hindi/001/H-S-1-F-01.tif")
augmented = get_augmented_positive(image)
result = augmented.permute(1, 2, 0)
plt.imshow(result, cmap='gray')
plt.show()
plt.imshow(image, cmap='gray')
plt.show()


class HindiSignatureDataset(ImageFolder):
    def __init__(self, root_dir, transform=None):
        super(HindiSignatureDataset, self).__init__(root_dir, transform=transform)
        self.forged_count = 30
        self.genuine_count = 24
        self.transform = transform

    def __getitem__(self, index):
        forged_images = self.samples[:self.forged_count]  # First N images are forged
        genuine_images = self.samples[self.forged_count:]  # Remaining images are genuine

        genuine_image = random.choice(genuine_images)
        forged_image = random.choice(forged_images)

        genuine_image = Image.open(genuine_image)
        augmented_genuine_image = get_augmented_positive(genuine_image)
        forged_image = Image.open(forged_image)

        if self.transform is not None:
            genuine_image = self.transform(genuine_image)
            forged_image = self.transform(forged_image)

        return genuine_image, augmented_genuine_image, forged_image


if __name__ == '__main__':
    print()
