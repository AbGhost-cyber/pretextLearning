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
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, image1: Tensor, image2: Tensor, labels: Tensor) -> Tensor:
        dist = cosine_similarity(image1, image2)  # Compute distance between image1 and image2

        margin_pos = torch.clamp_min(-dist + self.m, min=0.)
        margin_neg = torch.clamp_min(dist - self.m, min=0.)

        delta_pos = 1 - self.m
        delta_neg = self.m

        logit_pos = -margin_pos * (dist - delta_pos) * self.gamma
        logit_neg = margin_neg * (dist - delta_neg) * self.gamma

        loss = torch.mean(self.soft_plus((logit_neg - logit_pos)))

        return loss


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

    def forward(self, x: torch.Tensor, y: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        dist = dot_similarity(x, y)  # Compute distance between image1 and image2

        margin_pos = torch.clamp_min(-dist + self.m, min=0.)
        margin_neg = torch.clamp_min(dist - self.m, min=0.)

        delta_pos = 1 - self.m
        delta_neg = self.m

        logit_pos = -margin_pos * (dist - delta_pos) * self.gamma
        logit_neg = margin_neg * (dist - delta_neg) * self.gamma

        # loss = torch.mean(self.soft_plus((logit_neg - logit_pos)))

        label_p = torch.ones_like(logit_pos)
        label_n = torch.zeros_like(logit_neg)

        # label = label.squeeze(1)

        # loss_p = F.binary_cross_entropy_with_logits(logit_p, label_p * label)
        # loss_n = F.binary_cross_entropy_with_logits(logit_n, label_n * (1 - label))

        labels = torch.cat([label_p, label_n], dim=0)
        logits = torch.cat([logit_pos, logit_neg], dim=0)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        mean_loss = loss.mean()

        return mean_loss


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(512, 256)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2


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


class SiameseModel(nn.Module):
    def __init__(self):
        super(SiameseModel, self).__init__()

        # Load the pretrained ResNet-18 model
        self.backbone = models.resnet18(pretrained=True)
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
        self.backbone.fc = nn.LazyLinear(128)

    def forward(self, img1, img2):
        # Extract features from the backbone
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        return feat1, feat2


resnet = torchvision.models.resnet50()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone)
mean = 0.2062
std = 0.1148
normalize_dict = {'mean': [mean], 'std': [std]}
transform = SimCLRTransform(input_size=100, normalize=normalize_dict)
memory_bank = NNMemoryBankModule(size=4096)
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
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Define the learning rate scheduler
# scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)
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
criterion = NTXentLoss(temperature=1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

print("Starting Training")
for epoch in range(20):
    total_loss = 0
    for batch in dataloader:
        x0, x1 = batch[0]
        z0, p0 = model(x0)
        z1, p1 = model(x1)
        z0 = memory_bank(z0, update=False)  # for retrieval hence false
        z1 = memory_bank(z1, update=True)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        total_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    avg_loss = total_loss / len(dataloader)
    print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")

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
