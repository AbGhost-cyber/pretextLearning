import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim, Tensor
import torch.nn.functional as F
from torchvision import models

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.pairwise_distance(x1, x2, p=2)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return 1 - torch.cosine_similarity(x1, x2)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float) -> None:
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x1, x2, label):
        distance = cosine_distance(x1, x2)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss


def accuracy(distances, y, step=0.01):
    min_threshold_d = min(distances)
    max_threshold_d = max(distances)
    max_acc = 0
    same_id = (y == 1)

    for threshold_d in torch.arange(min_threshold_d, max_threshold_d + step, step):
        true_positive = (distances <= threshold_d) & same_id
        true_positive_rate = true_positive.sum().float() / same_id.sum().float()
        true_negative = (distances > threshold_d) & (~same_id)
        true_negative_rate = true_negative.sum().float() / (~same_id).sum().float()

        acc = 0.5 * (true_negative_rate + true_positive_rate)
        max_acc = max(max_acc, acc)
    return max_acc


def train(model, optimizer, criterion, dataloader, log_interval=1):
    model.train()
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        # x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataloader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0


@torch.no_grad()
def eval(model, criterion, dataloader, log_interval=10):
    model.eval()
    running_loss = 0
    number_samples = 0

    distances = []

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        # x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        distances.extend(zip(torch.pairwise_distance(x1, x2, 2).cpu().tolist(), y.cpu().tolist()))

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(dataloader), running_loss / number_samples))

    distances, y = zip(*distances)
    distances, y = torch.tensor(distances), torch.tensor(y)
    max_accuracy = accuracy(distances, y)
    print(f'Max accuracy: {max_accuracy}')
    return running_loss / number_samples, max_accuracy


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, image1: Tensor, image2: Tensor, labels: Tensor) -> Tensor:
        dist = euclidean_distance(image1, image2)  # Compute distance between image1 and image2

        margin_pos = torch.clamp_min(-dist + self.m, min=0.)
        margin_neg = torch.clamp_min(dist - self.m, min=0.)

        delta_pos = 1 - self.m
        delta_neg = self.m

        logit_pos = -margin_pos * (dist - delta_pos) * self.gamma
        logit_neg = margin_neg * (dist - delta_neg) * self.gamma

        loss = torch.mean(self.soft_plus(labels * (logit_neg - logit_pos)))

        return loss


# Showing images
def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 10})

    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Plotting data
def show_plot(iteration, loss):
    plt.plot(iteration, loss)
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
        img1 = Image.open(img1_tuple[0])

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] == img0_tuple[1])], dtype=np.float32))

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


# Load the training dataset
folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/training/")
mean = 0.2062
std = 0.1148
# Resize the images and transform to tensors
transformation = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(100),
                                     transforms.GaussianBlur(kernel_size=3),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[mean], std=[std])
                                     ])

# Initialize the network
siamese_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset,
                                        transform=transformation)

print(len(siamese_dataset))


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.Softplus(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.Softplus(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.Softplus()
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.Softplus(),
            nn.Dropout(p=0.3),

            nn.Linear(1024, 256),
            nn.Softplus(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 12)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similiarity
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
        self.backbone.fc = nn.Identity()

    def forward(self, img1, img2):
        # Extract features from the backbone
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        return feat1, feat2


@torch.no_grad()
def test_siamese_model(siamese_model, dataloader, threshold):
    siamese_model.eval()
    """
    Test a Siamese model and calculate FAR (False Acceptance Rate) and FRR (False Rejection Rate).

    Args:
        siamese_model (torch.nn.Module): The trained Siamese model.
        test_dataset (torch.utils.data.Dataset): The dataset containing test pairs.
        threshold (float): The threshold for classification.

    Returns:
        float: FAR (False Acceptance Rate)
        float: FRR (False Rejection Rate)
    """
    far_count = 0
    frr_count = 0
    correct_count = 0
    total_imposter_pairs = 0
    total_genuine_pairs = 0

    for batch_idx, (x1, x2, label) in enumerate(dataloader):

        output1, output2 = siamese_model(x1, x2)
        similarity = euclidean_distance(output1, output2)

        if label == 0:  # Imposter pair
            total_imposter_pairs += 1
            if similarity <= threshold:
                far_count += 1
        else:  # Genuine pair
            total_genuine_pairs += 1
            if similarity > threshold:
                frr_count += 1

        # Calculate accuracy
        if (similarity > threshold and label == 1) or (similarity <= threshold and label == 0):
            correct_count += 1

    far = far_count / total_imposter_pairs
    frr = frr_count / total_genuine_pairs
    accuracy = correct_count / (total_imposter_pairs + total_genuine_pairs)

    return far, frr, accuracy


# Load the training dataset
train_dataloader = DataLoader(siamese_dataset,
                              shuffle=True,
                              batch_size=64)
folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
test_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
                                     transform=transformation)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
net = SiameseModel()
criterion = CircleLoss(m=1.2, gamma=1024)
# criterion = ContrastiveLoss(margin=2)
# optimizer = optim.RMSprop(net.parameters(), lr=1e-7, eps=1e-8, weight_decay=5e-4, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=1e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

counter = []
loss_history = []
iteration_number = 0
num_epochs = 100
net.train()

print(len(siamese_dataset) // 32)

losses = []
accuracies = []

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('Training', '-' * 20)
    train(net, optimizer, criterion, train_dataloader)
    print('Evaluating', '-' * 20)
    far, frr, acc = test_siamese_model(net, test_dataloader, 0.5)
    print(f"far: {far}")
    print(f"frr, {frr}")
    print(f"acc, {acc}")
    # loss, acc = eval(net, criterion, test_dataloader)
    # losses.append(loss)
    # accuracies.append(acc)
    # scheduler.step()
torch.save(net.state_dict(), "test1.pt")
# plt.plot(losses)
# # plt.plot(accuracies)
# plt.show()
# for epoch in range(100):
#
#     # Iterate over batches
#     for i, (img0, img1, label) in enumerate(train_dataloader, 0):
#
#         # Zero the gradients
#         optimizer.zero_grad()
#
#         # Pass in the two images into the network and obtain two outputs
#         output1, output2 = net(img0, img1)
#
#         # Pass the outputs of the networks and label into the loss function
#         loss_contrastive = criterion(output1, output2, label)
#
#         # Calculate the backpropagation
#         loss_contrastive.backward()
#
#         # Optimize
#         optimizer.step()
#
#         # Every 10 batches print out the loss
#         if i % 10 == 0:
#             print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
#             iteration_number += 10
#
#             counter.append(iteration_number)
#             loss_history.append(loss_contrastive.item())

# show_plot(counter, loss_history)

# test_model = SiameseNetwork()
# state_dict = torch.load('test.pt')
# test_model.load_state_dict(state_dict)
# test_model.eval()
# # Locate the test dataset and load it into the SiameseNetworkDataset
# folder_dataset_test = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/testing/")
# test_dataset = SiameseNetworkDataset(imageFolderDataset=folder_dataset_test,
#                                      transform=transformation)
# test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)
#
# # Grab one image that we are going to test
# dataiter = iter(test_dataloader)
# x0, _, label = next(dataiter)
#
# for i in range(5):
#     # Iterate over 5 images and test them with the first image (x0)
#     _, x1, label2 = next(dataiter)
#
#     # Concatenate the two images together
#     concatenated = torch.cat((x0, x1), 0)
#
#     output1, output2 = net(x0, x1)
#     loss = euclidean_distance(output1, output2)
#     imshow(torchvision.utils.make_grid(concatenated), f'Dissimilarity: {loss.item():.2f}')
if __name__ == '__main__':
    print()
