import lightly
import torch
from lightly.data import SimCLRCollateFunction
from lightly.models.modules import NNCLRPredictionHead
from torch import nn
import numpy as np

collate_fn = SimCLRCollateFunction(
    input_size=32,
    gaussian_blur=0.,
    random_gray_scale=1
)
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


# Mean and standard deviation estimated by the actor network
mean = 0.5
std = 1
NNCLRPredictionHead()

# Sample a value from a standard Normal distribution
epsilon = np.random.normal(0, 1)

# Compute the sampled action using the reparameterization trick
sampled_action = mean + epsilon * std

# Use the sampled action for further processing or decision-making
print("Sampled action:", sampled_action)


if __name__ == '__main__':
    print()
