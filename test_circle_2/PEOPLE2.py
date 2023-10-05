import os
import random

import numpy as np
import torch
from PIL import Image
from lightly.loss import NTXentLoss
from lightly.transforms import SimCLRTransform
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from test_circle_2.circle_loss import CircleLoss, convert_label_to_similarity
import torch.nn.functional as F


def cosine_similarity(features_p):
    import torch.nn.functional as F

    # Normalize feature vectors to have unit length
    features_p = F.normalize(features_p, dim=0)

    # Calculate dot products between feature vectors
    dot_products = torch.matmul(features_p, features_p.t())

    # Calculate magnitudes of feature vectors
    magnitudes = torch.norm(features_p, dim=0, keepdim=True)

    # Calculate pairwise cosine similarity
    similarity_scores = dot_products / torch.matmul(magnitudes, magnitudes.t())

    return similarity_scores


class FaceDataset(Dataset):
    def __init__(self, folder_path, isTrain=True, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.isTrain = isTrain

        random.seed(2023)
        # Get a list of all subfolders in the main folder
        subfolders = sorted(os.listdir(folder_path))
        random.shuffle(subfolders)

        self.samples = []

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Get a list of all files in the sub-folder
            image_files = sorted(os.listdir(subfolder_path))

            self.compute_samples(image_files, subfolder, self.samples)

        if self.isTrain:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def compute_samples(self, files, label: str, sample_list):
        for file_name in files:
            image_path = os.path.join(self.folder_path, label, file_name)
            image = Image.open(image_path).convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
            mLabel = label.replace('s', '')
            sample_list.append((image, torch.tensor(int(mLabel))))

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample


class SimCLR(nn.Module):
    def __init__(self, num_classes=32, projection_dim=128):
        super(SimCLR, self).__init__()

        # Load the ResNet-18 backbone
        self.backbone = models.resnet18(pretrained=True)

        # Remove the last fully connected layer (the classifier)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # Define the encoder MLP
        self.encoder = nn.Sequential(
            nn.Linear(512, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(inplace=True)
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, num_classes)
        )

    def forward(self, x):
        # Backbone feature extraction
        features = self.backbone(x)
        features = features.view(features.size(0), -1)  # Flatten

        # Encoding
        encoded = self.encoder(features)

        # Projection head
        projection = self.projection_head(encoded)

        projection = F.normalize(projection, dim=1)

        return encoded, projection


mean = 0.2062
std = 0.1148
normalize_dict = {'mean': [mean], 'std': [std]}
transformation = SimCLRTransform(input_size=100, normalize=normalize_dict)
transformation_test = transforms.Compose([transforms.Resize((100, 100)),
                                          transforms.ToTensor()])

# folder_dataset = datasets.ImageFolder(root="")
face_dataset_train = FaceDataset('/Users/mac/research books/signature_research/data/faces/training/', isTrain=True,
                                 transform=transformation)
face_dataset_test = FaceDataset('/Users/mac/research books/signature_research/data/faces/testing/',
                                isTrain=False,
                                transform=transformation_test)


def get_loader(is_train: bool) -> DataLoader:
    return DataLoader(
        dataset=face_dataset_train if is_train else face_dataset_test,
        batch_size=64 if is_train else 2,
        shuffle=is_train,
    )


model = SimCLR()
train_loader = get_loader(is_train=True)
optimizer = torch.optim.Adam(params=model.parameters(), lr=6e-5, eps=1e-8, weight_decay=5e-4)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optimizer = torch.optim.RMSprop(model.parameters(), lr=6e-2, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
criterion = NTXentLoss(temperature=0.1)
losses = []
accuracies = []
num_epochs = 200
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('Training', '-' * 20)
    running_loss = 0
    number_samples = 0

    for batch_idx, batch in enumerate(train_loader):

        optimizer.zero_grad()
        img0, img1 = batch[0]
        e1, p1 = model(img0)
        e2, p2 = model(img1)
        loss = criterion(p1, p2)
        loss.backward()
        optimizer.step()

        number_samples += len(img0)
        running_loss += loss.item() * len(img0)
        if (batch_idx + 1) % 1 == 0 or batch_idx == len(train_loader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(train_loader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0
            # losses.append(loss.item())
    scheduler.step()
torch.save(model.state_dict(), "test_circle_test.pt")
print("finished")

if __name__ == '__main__':
    print()
