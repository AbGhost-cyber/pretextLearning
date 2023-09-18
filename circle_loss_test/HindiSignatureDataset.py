from sklearn.cluster import KMeans
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
import random
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

random.seed(2023)


def get_augmented_positive(genuine_image):
    aug_transform = transforms.Compose([
        transforms.CenterCrop((150, 600)),
        transforms.Resize((110, 170)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((180, 180)),  # flip upside down
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    return aug_transform(genuine_image)


# image = Image.open("/Users/mac/Downloads/data/BHSig260/Hindi/001/H-S-1-F-04.tif")
# augmented = get_augmented_positive(image)
# result = augmented.permute(1, 2, 0)
# plt.imshow(result, cmap='gray')
# plt.show()
# plt.imshow(image, cmap='gray')
# plt.show()


class HindiSignatureDataset(Dataset):
    def __init__(self, imageFolder: ImageFolder, K_train: int = 50, K_test: int = 50, isTrain=False, transform=None):
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
            image_paths = [imageFolder.imgs[i] for i in range(len(imageFolder)) if
                           imageFolder.imgs[i][1] == mclass_index]
            self.train_samples.extend(image_paths)

        for sub_folder in self.test_sub_folders:
            mclass_index = self.imageFolder.class_to_idx[sub_folder]
            image_paths = [imageFolder.imgs[i] for i in range(len(imageFolder)) if
                           imageFolder.imgs[i][1] == mclass_index]
            self.test_samples.extend(image_paths)

    def __getitem__(self, index):

        used_samples = self.train_samples if self.isTrain else self.test_samples
        sample_index = random.randint(0, len(used_samples) - 1)
        selected_sample = used_samples[sample_index]
        # Get the list of images in the selected sub-folder
        sub_folder_images = [
            path for path, path_index in used_samples
            if selected_sample[1] == path_index
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

hindi_folder = ImageFolder(root="/Users/mac/Downloads/data/BHSig260/Bengali")
hindiDataset = HindiSignatureDataset(imageFolder=hindi_folder, transform=None, isTrain=False)
# genuine_image, augmented_genuine_image, forged_image = hindiDataset[0]
# # train_loader = DataLoader(hindiDataset, batch_size=64, shuffle=True)
# # test_loader = DataLoader(hindiDataset, batch_size=64, shuffle=False)
print(f"size: {len(hindiDataset)}")

# # Assuming you have a PyTorch model called 'model'
# model.eval()
# embeddings_anchor = []
# embeddings_positive = []
# embeddings_negative = []
# with torch.no_grad():
# for batch in dataloader:
# anchor_inputs, positive_inputs, negative_inputs = batch # assuming you have a dataloader that provides triplets
# anchor_outputs = model(anchor_inputs)
# positive_outputs = model(positive_inputs)
# negative_outputs = model(negative_inputs)
# embeddings_anchor.append(anchor_outputs.numpy())
# embeddings_positive.append(positive_outputs.numpy())
# embeddings_negative.append(negative_outputs.numpy())
# embeddings_anchor = np.concatenate(embeddings_anchor)
# embeddings_positive = np.concatenate(embeddings_positive)
# embeddings_negative = np.concatenate(embeddings_negative)
# ```
#
# 2. Reduce the dimensionality of embeddings using t-SNE:
# ```python
# tsne = TSNE(n_components=2, random_state=0)
# embeddings_anchor_2d = tsne.fit_transform(embeddings_anchor)
# embeddings_positive_2d = tsne.fit_transform(embeddings_positive)
# embeddings_negative_2d = tsne.fit_transform(embeddings_negative)
# ```
#
# 3. Apply K-means clustering:
# ```python
# kmeans_anchor = KMeans(n_clusters=K) # K is the number of desired clusters
# kmeans_positive = KMeans(n_clusters=K)
# kmeans_negative = KMeans(n_clusters=K)
# clusters_anchor = kmeans_anchor.fit_predict(embeddings_anchor_2d)
# clusters_positive = kmeans_positive.fit_predict(embeddings_positive_2d)
# clusters_negative = kmeans_negative.fit_predict(embeddings_negative_2d)
# ```
#
# 4. Visualize the embeddings with cluster labels:
# ```python
# sns.set_palette("bright") # set color palette for clusters
# sns.scatterplot(x=embeddings_anchor_2d[:, 0], y=embeddings_anchor_2d[:, 1], hue=clusters_anchor, palette="bright", label="Anchor")
# sns.scatterplot(x=embeddings_positive_2d[:, 0], y=embeddings_positive_2d[:, 1], hue=clusters_positive, palette="bright", label="Positive")
# sns.scatterplot(x=embeddings_negative_2d[:, 0], y=embeddings_negative_2d[:, 1], hue=clusters_negative, palette="bright", label="Negative")
# plt.legend()
if __name__ == '__main__':
    print()
