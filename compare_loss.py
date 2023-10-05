from typing import Tuple

from torch import nn, Tensor
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.pairwise_distance(x1, x2, p=2)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return 1 - torch.cosine_similarity(x1, x2, dim=1)


class TripletLoss(nn.Module):
    def __init__(self, margin: float) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        dist_ap = euclidean_distance(anchor, positive)  # Compute distance between anchor and positive
        dist_an = euclidean_distance(anchor, negative)  # Compute distance between anchor and negative

        loss = F.softplus(dist_ap - dist_an + self.margin).mean()

        return loss


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    # Compute pairwise similarity matrix between features
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)

    # Create a matrix indicating label equality between samples
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    # Extract positive and negative pairwise matrices
    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    # Flatten similarity, positive, and negative matrices
    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)

    # Return similarity values for positive and negative pairs
    positive_similarities = similarity_matrix[positive_matrix]
    negative_similarities = similarity_matrix[negative_matrix]

    return positive_similarities, negative_similarities


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
        ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = - ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))

        return loss


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the data points A and B in 3D
A = torch.tensor([1.0, 0.5, 2.0])
B = torch.tensor([-0.5, 1.0, -1.0])

# Create an instance of the CircleLoss class
loss_fn = CircleLoss(m=0.25, gamma=1.0)

# Compute the loss and gradients for sp and sn
sp = torch.autograd.Variable(A, requires_grad=True)
sn = torch.autograd.Variable(B, requires_grad=True)

loss = loss_fn(sp, sn)
loss.backward()

# Access the gradients with respect to sp and sn
grad_sp = sp.grad[2].item()  # Get the gradient along the Z-axis (index 2)
grad_sn = sn.grad[2].item()  # Get the gradient along the Z-axis (index 2)

# Create the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot sp and sn
ax.scatter(sp[0].item(), sp[1].item(), grad_sp, c='red', label='sp')
ax.scatter(sn[0].item(), sn[1].item(), grad_sn, c='blue', label='sn')

# Plot gradients
ax.quiver(sp[0].item(), sp[1].item(), grad_sp, 0, 0, 0.2, color='red', label='Gradient |dL/dsp|')
ax.quiver(sn[0].item(), sn[1].item(), grad_sn, 0, 0, 0.2, color='blue', label='Gradient |dL/dsn|')

# Set labels and title
ax.set_xlabel('sp')
ax.set_ylabel('sn')
ax.set_zlabel('Gradient')
ax.set_title('sp, sn, and Gradients')

# Add a legend
ax.legend()

# Show the plot
plt.show()
#
# # Show the plot
# plt.show()
# feat = nn.functional.normalize(torch.rand(224, 64, requires_grad=True))
# lbl = torch.randint(high=10, size=(224,))
#
# sim = convert_label_to_similarity(feat, lbl)
#
# criterion = CircleLoss(m=0.25, gamma=256)
# circle_loss = criterion(*sim)  # unpack
#
# print(circle_loss)
if __name__ == '__main__':
    print()
