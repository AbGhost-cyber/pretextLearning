import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.manifold import TSNE

from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
from PIL import Image
import random
from sklearn.metrics import average_precision_score

from torch import Tensor


class CircleTripleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleTripleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        dist_ap = torch.pairwise_distance(anchor, positive, p=2)  # Compute distance between anchor and positive
        dist_an = torch.pairwise_distance(anchor, negative, p=2)  # Compute distance between anchor and negative

        # circle loss expects dist_ap > 1−m and dis_an < m.
        ap = torch.clamp_min(-dist_ap + self.m, min=0.)
        an = torch.clamp_min(dist_an + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (dist_ap - delta_p) * self.gamma
        logit_n = an * (dist_an - delta_n) * self.gamma

        loss = torch.mean(self.soft_plus(torch.max(logit_n - logit_p, torch.zeros_like(logit_p))))

        return loss


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


class SigNet2(nn.Module):

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # input size = [150, 150, 1]
            nn.Conv2d(1, 96, 11),  # size = [145,210]
            nn.Softplus(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [72, 105]

            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [36, 52]
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),  # size = [18, 26]
            nn.Dropout(p=0.3),

            nn.Flatten(1, -1),  # 18*26*256
            nn.Linear(11 * 20 * 256, 1024),
            nn.Softplus(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 32),
        )

    def forward(self, x1, x2, x3):
        x1 = self.features(x1)
        x2 = self.features(x2)
        x3 = self.features(x3)
        return x1, x2, x3

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.features:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


class SiameseModel2(nn.Module):
    def __init__(self):
        super(SiameseModel2, self).__init__()

        # Load the pretrained ResNet-18 model
        self.backbone = models.resnet34(pretrained=True)
        for params in self.backbone.parameters():
            params.requires_grad = False

        # Modify the first convolutional layer for grayscale images
        self.backbone.conv1 = nn.Conv2d(
            in_channels=1,  # Set the number of input channels to 1 for grayscale
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False
        )

        # Remove the fully connected layer
        self.backbone.fc = nn.Identity()

    def forward(self, img1, img2, img3):
        # Extract features from the backbone
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)
        feat3 = self.backbone(img3)

        return feat1, feat2, feat3


@torch.no_grad()
def evaluate2(model, criterion, dataloader, log_interval=20):
    model.eval()
    correct = 0
    total = 0

    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):

        embeddings_anchor, embeddings_positive, embeddings_negative = model(anchor, positive, negative)

        loss = criterion(embeddings_anchor, embeddings_positive, embeddings_negative)

        # Count the number of correct predictions
        correct += torch.sum(loss <= 0).item()
        total += loss.numel()

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Accuracy: {:.4f}'.format(batch_idx + 1, len(dataloader), correct / total))

    accuracy = correct / total
    print('Accuracy: {:.4f}'.format(accuracy))
    return accuracy


image_transform = transforms.Compose([
    transforms.Resize((100, 170)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.0019], std=[0.0004])
])
hindi_folder = ImageFolder(root="/Users/mac/Downloads/data/BHSig260/Hindi")
train_hindiDataset = HindiSignatureDataset(imageFolder=hindi_folder, transform=image_transform, isTrain=True)
test_hindiDataset = HindiSignatureDataset(imageFolder=hindi_folder, transform=image_transform, isTrain=False)

model = SigNet2()
criterion = CircleTripleLoss(m=0.25, gamma=256)


def cal_accuracy(distances, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d + step, step):
        true_positive = (distances <= threshold_d)
        true_positive_rate = true_positive.sum().float() / len(distances)

        true_negative = (distances > threshold_d)
        true_negative_rate = true_negative.sum().float() / len(distances)

        acc = 0.5 * (true_negative_rate + true_positive_rate)
        max_acc = max(max_acc, acc)

    return max_acc


def compute_mAP(distances):
    labels = [1] * len(distances)  # Assuming all distances are positive pairs
    average_precision = average_precision_score(labels, distances)
    return average_precision


def evaluate(model, criterion, dataloader, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0
    distances = []

    for batch_idx, (anchor, positive, negative) in enumerate(dataloader):
        # anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        embed_anchor, embed_positive, embed_negative = model(anchor, positive, negative)
        loss = criterion(embed_anchor, embed_positive, embed_negative)
        distances.extend(torch.pairwise_distance(embed_anchor, embed_positive, p=2).cpu().tolist())
        number_samples += len(anchor)
        running_loss += loss.item() * len(anchor)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataloader), running_loss / number_samples))

    average_precision = compute_mAP(distances)
    print(f'Average Precision: {average_precision}')

    return running_loss / number_samples, average_precision


test_loader = DataLoader(test_hindiDataset, batch_size=40, shuffle=False)
print(len(test_hindiDataset))

if __name__ == '__main__':
    print()
