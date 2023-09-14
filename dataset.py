import random

import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets


# The goal of this dataset is to return:
# 1) the original image, 2) patches of the original image and negative image
class JigsawDataset(Dataset):
    def __init__(self, imageFolderDataset, image_dim=224):
        self.imageFolderDataset = imageFolderDataset
        self.image_dim = image_dim
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_dim, self.image_dim)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2062], std=[0.1148])])
        self.center_crop = transforms.CenterCrop(image_dim)

    def __getitem__(self, index):
        positive_image_triple = random.choice(self.imageFolderDataset.imgs)
        while True:
            # Look until a different class image is found
            negative_triple = random.choice(self.imageFolderDataset.imgs)
            if positive_image_triple[1] != negative_triple[1]:
                break

        positive_image = Image.open(positive_image_triple[0])
        negative_image = Image.open(negative_triple[0])

        # Extract patches from the positive image
        patches = self.extract_patches(positive_image)

        positive_image = self.image_transform(positive_image)
        negative_image = self.image_transform(negative_image)

        return {'original': positive_image, 'patches': patches, 'negative': negative_image}

    def __len__(self):
        return len(self.imageFolderDataset.imgs)

    def extract_patches(self, original_img):
        num_division = 3
        patch_size = self.image_dim // num_division
        crop_64 = transforms.RandomCrop((64, 64))  # Crop patches of size 64x64
        color_transform = transforms.Grayscale()

        # resize then random crop
        first_transform = transforms.Compose([
            transforms.Resize(self.image_dim),
            transforms.RandomCrop(self.image_dim)
        ])

        # crop 64, apply jitter and tensor
        final_transform = transforms.Compose([
            crop_64,
            color_transform,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.2062], std=[0.1148])
        ])

        sample = first_transform(original_img)

        # Define the cropped areas based on the patch size, we need 9 patches
        cropped_areas = []
        for i in range(num_division):
            for j in range(num_division):
                # Calculate the coordinates for cropping each patch (L, T, R, B)
                left = i * patch_size
                right = (i + 1) * patch_size
                top = j * patch_size
                bottom = (j + 1) * patch_size
                cropped_area = (left, top, right, bottom)
                cropped_areas.append(cropped_area)

        # Crop the patches from the sample
        samples = [sample.crop(cropped_area) for cropped_area in cropped_areas]

        # Apply the final transformations to each patch
        samples = [final_transform(patch) for patch in samples]

        # Shuffle the patches randomly
        random.shuffle(samples)

        return samples


folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
# folder_dataset = datasets.ImageFolder(root=folder_root)
jigsawDataset = JigsawDataset(imageFolderDataset=folder_dataset)
first_item = jigsawDataset[0]
positive_image = first_item['original']
patches = first_item['patches']
negative_image = first_item['negative']

plt.imshow(positive_image.permute(1, 2, 0), cmap='gray')
fig, axes = plt.subplots(3, 3, figsize=(7, 7))
for i, ax in enumerate(axes.flat):
    patch = patches[i].permute(1, 2, 0)  # Transpose the dimensions for plotting
    print(f"patch shape {patch.shape}")
    ax.imshow(patch, cmap='gray')  # cmap='gray'
    ax.axis('off')

plt.show()

# softplus = nn.Softplus()
#
# input_tensor = torch.tensor([-5.0, 0.0, 5.0])
# output_tensor = softplus(input_tensor)

# print(output_tensor)
if __name__ == '__main__':
    print()
