from typing import Tuple

import torch
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


# def cosine_similarity(features_p):
#     import torch.nn.functional as F
#
#     # Normalize feature vectors to have unit length
#     features_p = F.normalize(features_p, dim=1)
#
#     # Calculate dot products between feature vectors
#     dot_products = torch.matmul(features_p, features_p.t())
#
#     # Calculate magnitudes of feature vectors
#     magnitudes = torch.norm(features_p, dim=1, keepdim=True)
#
#     # Calculate pairwise cosine similarity
#     similarity_scores = dot_products / torch.matmul(magnitudes, magnitudes.t())
#
#     return similarity_scores


# features_p = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=torch.float)
# features_n = torch.tensor([[1, 2, 4], [4, 5, 4], [7, 8, 4]], dtype=torch.float)
# sp = cosine_similarity(features_p)
# sn = cosine_similarity(features_n)
# criterion = CircleLoss(m=0.25, gamma=32)
# loss = criterion(sp, sn)
# print(loss)

if __name__ == '__main__':
    print()
