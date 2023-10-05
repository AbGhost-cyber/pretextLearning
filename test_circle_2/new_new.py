import os

import numpy as np
import torch
from torch import nn, Tensor
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import models, transforms
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm

from circle_loss import convert_label_to_similarity, CircleLoss


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=MNIST(root="./data", train=is_train, transform=ToTensor(), download=True),
        batch_size=batch_size,
        shuffle=is_train,
    )


class Model(nn.Module):
    def __init__(self) -> None:
        super(Model, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
        )

    def forward(self, inp: Tensor) -> Tensor:
        feature = self.feature_extractor(inp).mean(dim=[2, 3])
        return nn.functional.normalize(feature)


import torch

import torch

# def calculate_recall_precision(model, val_loader, thresh=0.75):
#     tp = 0
#     fn = 0
#     fp = 0
#     thresh = 0.75
#     with torch.no_grad():
#         for img, label in val_loader:
#             pred = model(img)
#             gt_label = label[:, 0] == label[:, 1]  # Compare labels across the entire batch
#             pred_label = torch.sum(pred[:, 0] * pred[:, 1],
#                                    dim=1) > thresh  # Calculate predictions for the entire batch
#             tp += torch.sum(gt_label & pred_label).item()
#             fn += torch.sum(gt_label & ~pred_label).item()
#             fp += torch.sum(~gt_label & pred_label).item()
#
#     recall = tp / (tp + fn)
#     precision = tp / (tp + fp)
#
#     return recall, precision

import torch


def calculate_recall_precision_accuracy(model, val_loader):
    model.eval()
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    thresh = 0.75
    for img, label in val_loader:
        pred = model(img)
        gt_label = label[0] == label[1]
        pred_label = torch.sum(pred[0] * pred[1]) > thresh
        if gt_label and pred_label:
            tp += 1
        elif gt_label and not pred_label:
            fn += 1
        elif not gt_label and pred_label:
            fp += 1
        else:
            tn += 1

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("Accuracy: {:.4f}".format(accuracy))


def calculate_recall_precision_accuracy1(model, val_loader):
    model.eval()
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    thresh = 0.75
    for batch in val_loader:
        images, labels = batch

        preds = model(images)

        pred_labels = torch.sum(preds, dim=1) > thresh

        tp += torch.sum((pred_labels == 1) & (labels == 1)).item()
        fn += torch.sum((pred_labels == 0) & (labels == 1)).item()
        fp += torch.sum((pred_labels == 1) & (labels == 0)).item()
        tn += torch.sum((pred_labels == 0) & (labels == 0)).item()

    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    print("Recall: {:.4f}".format(recall))
    print("Precision: {:.4f}".format(precision))
    print("Accuracy: {:.4f}".format(accuracy))


# def calculate_accuracy(model: nn.Module, data_loader: DataLoader) -> float:
#     model.eval()
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for img, label in data_loader:
#             pred = model(img)
#             # Assuming the label is a tensor of class indices
#             _, predicted_labels = torch.max(pred, dim=1)
#             correct += (predicted_labels == label).sum().item()
#             total += img.size(0)
#
#     accuracy = correct / total
#     return accuracy


# def calculate_accuracy(model: nn.Module, data_loader: DataLoader, threshold: float = 0.75) -> float:
#     model.eval()
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for img, label in data_loader:
#             pred = model(img)
#             # Assuming the label is a tensor of class indices
#             _, predicted_labels = torch.max(pred, dim=1)
#             predicted_labels = predicted_labels > threshold  # Apply threshold
#             correct += (predicted_labels == label).sum().item()
#             total += img.size(0)
#
#     accuracy = correct / total
#     return accuracy


def main(resume: bool = True) -> None:
    model = Model()
    optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
    train_loader = get_loader(is_train=True, batch_size=64)
    val_loader = get_loader(is_train=False, batch_size=64)
    criterion = CircleLoss(m=0.25, gamma=80)

    for epoch in range(20):
        model.train()
        print('Epoch {}/{}'.format(epoch, 20))
        print('Training', '-' * 20)
        running_loss = 0
        number_samples = 0

        for batch_idx, (x1, y) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x1)
            loss = criterion(*convert_label_to_similarity(output, y))
            loss.backward()
            optimizer.step()

            number_samples += len(x1)
            running_loss += loss.item() * len(x1)
            if (batch_idx + 1) % 100 == 0 or batch_idx == len(train_loader) - 1:
                print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(train_loader), running_loss / number_samples))
                running_loss = 0
                number_samples = 0
        print("evaluating", '-' * 20)
        calculate_recall_precision_accuracy(model, val_loader)
        calculate_recall_precision_accuracy1(model, val_loader)
        # accuracy = calculate_recall_precision(model, val_loader)
        # print('Accuracy after epoch {}: {:.2%}'.format(epoch, accuracy))


# def calculate_accuracy(model, dataloader, dataset):
#     correct = 0
#     for x, y in dataloader:
#         # validation
#         z = model(x)
#         _, label = torch.max(z, 1)
#         correct += (label == y).sum().item()
#     accuracy = 100 * (correct / len(dataset))
#     return accuracy
#
#
# class NNCLR(nn.Module):
#     def __init__(self, dropout_rate=0.3):
#         super().__init__()
#
#         self.backbone = models.vgg16(pretrained=True)
#         self.projection_head = NNCLRProjectionHead(512, 4096, 256)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.prediction_head = NNCLRPredictionHead(256, 4096, 256)
#
#     def forward(self, x):
#         y = self.backbone.features(x)
#         y = self.backbone.avgpool(y)
#         y = torch.flatten(y, 1)
#         z = self.projection_head(y)
#         # z = self.dropout(z) # Apply dropout to z
#         p = self.prediction_head(z)
#         z = z.detach()
#         return z, p


# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomResizedCrop((224, 224)),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#     transforms.RandomAffine(degrees=10, shear=10),
#     transforms.RandomPerspective(distortion_scale=0.2),
#     transforms.RandomHorizontalFlip(p=0.5),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.9706, 0.9706, 0.9706], std=[0.1418, 0.1418, 0.1418]),
#     transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),
# ])


# class SiameseNetwork(nn.Module):
#     def __init__(self, nnclr_model, dropout_rate=0.0):
#         super().__init__()
#         self.nnclr_model = nnclr_model
#         self.backbone = nnclr_model.backbone
#         self.dropout = nn.Dropout(dropout_rate)
#         self.fc = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, image1, image2):
#         z1 = self.backbone(image1).flatten(start_dim=1)
#         z2 = self.backbone(image2).flatten(start_dim=1)
#         z1 = self.dropout(z1)  # Apply dropout to z1
#         z2 = self.dropout(z2)  # Apply dropout to z2
#         concatenated = torch.cat((z1, z2), dim=1)
#         # Predict similarity using fully connected layers
#         similarity = self.fc(concatenated)
#         return similarity


# from sklearn.utils.class_weight import compute_class_weight
# import numpy as np
#
# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=std, std=std)
# ])
#
# # Assuming you have a data loader called 'trainloader'
# train_labels = []
# for batch_x1, batch_x2, batch_y in trainloader:
#     train_labels.extend(batch_y.tolist())
#
# train_labels = np.array(train_labels)
#
# # Calculate class weights
# class_weights = compute_class_weight('balanced', np.unique(train_labels), train_labels)
# print(class_weights)


# sigNet = SigNet()
# backbone = nn.Sequential(
#         *list(sigNet.children())[:-1],
#         nn.AdaptiveAvgPool2d((1, 1))
#     )

# def calculate_accuracy2(model, data_loader, threshold=0.5):
#     model.eval()
#     correct_predictions = 0
#     total_predictions = 0
#
#     with torch.no_grad():
#         for batch in data_loader:
#             image1, image2, labels = batch
#
#             # Forward pass
#             similarity = model(image1, image2)
#
#             # Apply threshold and convert to predicted labels
#             predicted_labels = (similarity > threshold).float()
#
#             # Calculate accuracy
#             correct_predictions += (predicted_labels == labels).sum().item()
#             total_predictions += labels.size(0)
#
#     accuracy = correct_predictions / total_predictions
#     return accuracy


if __name__ == "__main__":
    main()
