import PIL.Image
import lightly
import torch
from PIL import ImageOps
from lightly.data import SimCLRCollateFunction
from lightly.models.modules import NNCLRPredictionHead
from lightly.transforms import SimCLRTransform
from matplotlib import pyplot as plt
from torch import nn
import numpy as np
from torchvision import transforms

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    # ImageOps.invert,
    # transforms.RandomRotation(degrees=10),
    # transforms.RandomResizedCrop((224,224)),
    # transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=mean, std=std)
])
mean = [0.9706, 0.9706, 0.9706]
std = [0.1418, 0.1418, 0.1418]
normalize_dict = {'mean': mean, 'std': std}
transform_simclr = SimCLRTransform(input_size=224, gaussian_blur=0., random_gray_scale=1)

image = PIL.Image.open("/Users/mac/Downloads/my_training/same/BHSig260/Bengali/001/B-S-1-F-01.tif").convert('RGB')
# Apply transformations to create correlated views
# sample_1, sample_2 = transform_simclr(image)
sample_1 = image_transform(image)
sample_2 = image_transform(image)
# Display the original image and the two augmented views
plt.imshow(image)
plt.show()
plt.imshow(sample_1.permute(1, 2, 0))
plt.show()
plt.imshow(sample_2.permute(1, 2, 0))
plt.show()


# cnn1 = nn.Sequential(
#     nn.Conv2d(1, 96, kernel_size=11, stride=2),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(3, stride=2),
#
#     nn.Conv2d(96, 256, kernel_size=5, stride=1),
#     nn.ReLU(inplace=True),
#     nn.MaxPool2d(2, stride=2),
#
#     nn.Conv2d(256, 384, kernel_size=3, stride=1),
#     nn.ReLU(inplace=True),
#     nn.Flatten()
# )
# fc1 = nn.Sequential(
#     nn.Linear(384 * 7 * 7, 1024),
#     nn.ReLU(),
#     nn.Dropout(p=0.3),
#
#     nn.Linear(1024, 256),
#     nn.ReLU(),
#     nn.Dropout(p=0.5),
#     nn.Linear(256)
# )


# def layer_summary(X_shape, model):
#     X = torch.randn(*X_shape)
#     for layer in model:
#         X = layer(X)
#         print(layer.__class__.__name__, 'output shape:\t', X.shape)


# layer_summary((1, 100, 100), cnn1)
# layer_summary((1, 100, 100), fc1)
# output_cnn1 = cnn1(torch.randn((1, 1, 100, 100)))  # Adjust the input shape to match the number of channels
# output_fc1 = fc1(output_cnn1)
# print(output_fc1.shape)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # Calculate the sigmoid of the logits to get predicted probabilities
        probs = torch.sigmoid(logits)

        # Create the alpha weight matrix
        alpha_weight = torch.where(targets == 1, self.alpha * torch.ones_like(targets),
                                   (1 - self.alpha) * torch.ones_like(targets))

        # Calculate the focal weight
        focal_weight = (1 - probs) ** self.gamma

        # Calculate the positive samples loss term
        pos_loss = -(alpha_weight * focal_weight * torch.log(probs + 1e-8) * targets).mean()

        # Calculate the negative samples loss term
        neg_loss = ((1 - alpha_weight) * focal_weight * torch.log(1 - probs + 1e-8) * (1 - targets)).mean()

        # Combine the loss terms and return the final loss
        loss = pos_loss + neg_loss
        return loss


if __name__ == '__main__':
    print()
