import random

import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from PIL import Image
from lightly.loss import NTXentLoss
from lightly.models.modules import (
    NNCLRPredictionHead,
    NNCLRProjectionHead,
    NNMemoryBankModule,
)
import matplotlib.pyplot as plt
from lightly.transforms.simclr_transform import SimCLRTransform
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torchvision import datasets, transforms, models
from torchvision.transforms import Normalize
import torch.nn.functional as F

random.seed(2023)


def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)

        # We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # Look untill the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                # Look untill a different class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = Image.open(img0_tuple[0])
        img0 = img0.convert("RGB")
        img1 = Image.open(img1_tuple[0])
        img1 = img1.convert("RGB")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return 1 - torch.cosine_similarity(x1, x2)


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.pairwise_distance(x1, x2, p=2)


class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='dot'):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, genuine, forged):
        if self.similarity == 'dot':
            sim = self.dot_similarity(genuine, forged)
        elif self.similarity == 'cos':
            sim = self.cosine_similarity(genuine, forged)
        else:
            raise ValueError('This similarity is not implemented.')

        alpha = F.relu(self.margin - sim)
        logit = self.scale * alpha * (sim - self.margin)

        label = torch.ones_like(logit)

        loss = F.binary_cross_entropy_with_logits(logit, label)

        return loss

    def dot_similarity(self, x, y):
        return torch.matmul(x.view(x.size(0), -1), y.view(y.size(0), -1).t())

    def cosine_similarity(self, x, y):
        x = x.view(x.size(0), -1)
        y = y.view(y.size(0), -1)
        x_norm = F.normalize(x, p=2, dim=1)
        y_norm = F.normalize(y, p=2, dim=1)
        return torch.matmul(x_norm, y_norm.t())


def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    abs_x = torch.sqrt(torch.sum(torch.square(x), dim=1, keepdim=True))
    abs_y = torch.sqrt(torch.sum(torch.square(y), dim=1, keepdim=True))
    up = torch.matmul(x, y.t())
    down = torch.matmul(abs_x, abs_y.t())
    return up / down


def dot_similarity(x, y):
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    return torch.mm(x, y.t())


class CircleLossWithoutLabels(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLossWithoutLabels, self).__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dist = dot_similarity(x, y)  # Compute distance between image1 and image2

        margin_pos = torch.clamp_min(-dist + self.m, min=0.)
        margin_neg = torch.clamp_min(dist - self.m, min=0.)

        delta_pos = 1 - self.m
        delta_neg = self.m

        logit_pos = -margin_pos * (dist - delta_pos) * self.gamma
        logit_neg = margin_neg * (dist - delta_neg) * self.gamma

        label_p = torch.ones_like(logit_pos)
        label_n = torch.zeros_like(logit_neg)

        labels = torch.cat([label_p, label_n], dim=0)
        logits = torch.cat([logit_pos, logit_neg], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        mean_loss = loss.mean()

        return mean_loss


class NNCLR(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = NNCLRProjectionHead(512, 512, 128)
        self.prediction_head = NNCLRPredictionHead(128, 512, 128)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        z = z.detach()
        return z, p


class SiameseNetwork(nn.Module):
    def __init__(self, nnclr_model):
        super().__init__()
        self.nnclr_model = nnclr_model
        self.backbone = nnclr_model.backbone

    def forward(self, image1, image2):
        z1 = self.backbone(image1).flatten(start_dim=1)
        z2 = self.backbone(image2).flatten(start_dim=1)
        return z1, z2


resnet = torchvision.models.resnet18()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone)
mean = 0.2062
std = 0.1148
normalize_dict = {'mean': [mean], 'std': [std]}
transform = SimCLRTransform(input_size=100, normalize=normalize_dict)
memory_bank = NNMemoryBankModule(size=2048)
# transform = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomVerticalFlip(),
#     transforms.RandomResizedCrop((100, 100)),
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[mean], std=[std])
# ])
# transform_test = transforms.Compose([
#     transforms.Resize((100, 100)),
#     transforms.Grayscale(),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[mean], std=[std])
# ])
folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/training/")
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transform)
dataloader = torch.utils.data.DataLoader(
    siamese_dataset,
    batch_size=64,
    shuffle=True
)
folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
# siamese_dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                              transform=transform_test)
# dataloader_test = torch.utils.data.DataLoader(
#     siamese_dataset_test,
#     batch_size=1,
#     shuffle=False
# )
# criterion = CircleLossWithoutLabels(m=2, gamma=32)
# model = SiameseNetwork()
# model = NNCLR(backbone)
# model.load_state_dict(torch.load("test4.pt"))
# # Freeze the backbone and projection head parameters
# for param in model.backbone.parameters():
#     param.requires_grad = False
# for param in model.projection_head.parameters():
#     param.requires_grad = False

# Create a new optimizer for the prediction head

# Define the optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)

# Define the learning rate scheduler
# optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
# print("Starting Training")
# for epoch in range(100):
#     total_loss = 0
#     for X0, X1, labels in dataloader:
#         p0, p1 = model(X0, X1)
#         # _, p1 = model(X1)
#         optimizer.zero_grad()
#         loss = criterion(p0, p1, labels)
#         total_loss += loss.detach()
#         loss.backward()
#         optimizer.step()
#     scheduler.step()
#     avg_loss = total_loss / len(dataloader)
#     print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
# torch.save(model.state_dict(), "test6_fine.pt")
# example_batch = next(iter(dataloader))
#
# concatenated = torch.cat((example_batch[0], example_batch[1]), 0)
# imshow(torchvision.utils.make_grid(concatenated))
criterion = NTXentLoss(temperature=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-2)
scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
models.resnet18()
# lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

print("Starting Training")
for epoch in range(100):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        z0, p0 = model(x0)
        z1, p1 = model(x1)
        optimizer.zero_grad()
        z0 = memory_bank(z0, update=False)  # for retrieval hence false
        z1 = memory_bank(z1, update=True)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
    scheduler.step()

torch.save(model.state_dict(), "test4.pt")

# my_test_model = SiameseNetwork()
# state_dict = torch.load('test6_fine.pt')
# my_test_model.load_state_dict(state_dict)
# my_test_model.eval()
# #
# # Grab one image that we are going to test
# dataiter = iter(dataloader_test)
# # x0, _, label1 = next(dataiter)
# my_test_model.eval()
# for i in range(15):
#     # Iterate over 5 images and test them with the first image (x0)
#     x0, x1, label2 = next(dataiter)
#
#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)
#
#     p0, p1 = my_test_model(x0, x1)
#     # _, p1 = my_test_model(x1)
#     distance = torch.pairwise_distance(p0, p1, 2)
#     imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {distance.item():.2f}')
if __name__ == '__main__':
    print()
