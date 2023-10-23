from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import Tensor, nn


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


# Create a random input tensor and labels

# Create a random input tensor and labels
input_tensor = torch.rand(256, 32, requires_grad=True)  # Set requires_grad=True
labels = torch.randint(high=32, size=(256,))  # Assuming binary labels for simplicity
ll = torch.tensor([[1], [1]])
print(input_tensor.shape)
print(labels.shape)
print(ll.shape)

# Normalize the input tensor
input_normed = nn.functional.normalize(input_tensor)

# Create an instance of CircleLoss
loss_fn = CircleLoss(m=0.25, gamma=1.0)

# Compute the similarity and dissimilarity pairs
sp, sn = convert_label_to_similarity(input_normed, labels)

# Compute the loss
loss = loss_fn(sp, sn)

# Compute gradients
loss.backward()

# Get the gradients of the input tensor
grads = input_tensor.grad

# Normalize gradients
grads_normalized = (grads - grads.min()) / (grads.max() - grads.min())

# Plot the input tensor with overlaid gradients
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(input_tensor.squeeze().detach().cpu().numpy(), cmap='gray')
ax1.set_title('Input Tensor')
ax1.axis('off')
ax2.imshow(grads_normalized.squeeze().detach().cpu().numpy(), cmap='gray')
ax2.set_title('Gradients')
ax2.axis('off')
plt.show()

if __name__ == '__main__':
    print()
