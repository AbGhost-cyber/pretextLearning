from typing import Tuple

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE


def euclidean_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return torch.pairwise_distance(x1, x2, p=2)


def cosine_distance(x1: Tensor, x2: Tensor) -> Tensor:
    return 1 - torch.cosine_similarity(x1, x2, dim=1)


def convert_label_to_similarity(normed_feature: Tensor, label: Tensor) -> Tuple[Tensor, Tensor]:
    similarity_matrix = normed_feature @ normed_feature.transpose(1, 0)
    label_matrix = label.unsqueeze(1) == label.unsqueeze(0)

    positive_matrix = label_matrix.triu(diagonal=1)
    negative_matrix = label_matrix.logical_not().triu(diagonal=1)

    similarity_matrix = similarity_matrix.view(-1)
    positive_matrix = positive_matrix.view(-1)
    negative_matrix = negative_matrix.view(-1)
    return similarity_matrix[positive_matrix], similarity_matrix[negative_matrix]


class OnlineContrastiveLoss(nn.Module):
    def __init__(self, margin: float, selection_type: str = 'hard') -> None:
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.selection_type = selection_type

    def forward(self, anchor_embeddings: Tensor, positive_embeddings: Tensor, negative_embeddings: Tensor):
        anchor_positive_similarity = F.cosine_similarity(anchor_embeddings, positive_embeddings)

        if self.selection_type == 'hard':
            # Select the hardest negative for each anchor
            hardest_negatives = negative_embeddings.max(dim=1)[0]
        elif self.selection_type == 'semi-hard':
            # Select the semi-hard negatives for each anchor
            mask = (negative_embeddings > anchor_positive_similarity.view(-1, 1)).any(dim=1)
            hardest_negatives = torch.masked_select(negative_embeddings,
                                                    mask.view(-1, 1)).view(-1, negative_embeddings.size(-1))
        else:
            raise ValueError("Invalid negative selection method.")

        loss = torch.mean(
            torch.max(torch.tensor(0.0), self.margin - anchor_positive_similarity + hardest_negatives)
        )

        return loss


def online_contrastive_loss(anchor, positive, negatives, margin, negative_selection='hard'):
    # Compute similarity scores between anchor and positive
    anchor_positive_similarity = F.cosine_similarity(anchor, positive)

    if negative_selection == 'hard':
        # Select the hardest negative for each anchor
        hardest_negatives = negatives.max(dim=1)[0]
    elif negative_selection == 'semi-hard':
        # Select the semi-hard negatives for each anchor
        mask = (negatives > anchor_positive_similarity.view(-1, 1)).any(dim=1)
        hardest_negatives = torch.masked_select(negatives, mask.view(-1, 1)).view(-1, negatives.size(-1))
    else:
        raise ValueError("Invalid negative selection method.")

    loss = torch.mean(
        torch.max(torch.tensor(0.0), margin - anchor_positive_similarity + hardest_negatives)
    )

    return loss


class TripletLoss(nn.Module):
    def __init__(self, margin: float) -> None:
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        dist_ap = euclidean_distance(anchor, positive)  # Compute distance between anchor and positive
        dist_an = euclidean_distance(anchor, negative)  # Compute distance between anchor and negative

        loss = F.softplus(dist_ap - dist_an + self.margin).mean()

        return loss


class CircleTripleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleTripleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        dist_ap = euclidean_distance(anchor, positive)  # Compute distance between anchor and positive
        dist_an = euclidean_distance(anchor, negative)  # Compute distance between anchor and negative

        # circle loss expects dist_ap > 1−m and dis_an < m.
        ap = torch.clamp_min(-dist_ap + self.m, min=0.)
        an = torch.clamp_min(dist_an + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (dist_ap - delta_p) * self.gamma
        logit_n = an * (dist_an - delta_n) * self.gamma

        loss = torch.mean(self.soft_plus(torch.max(logit_n - logit_p, torch.zeros_like(logit_p))))

        return loss


class CircleTripleLoss1(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleTripleLoss1, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor, labels: Tensor) -> Tensor:
        labels_diff = labels.unsqueeze(1) - labels.unsqueeze(0)  # Compute label differences
        mask_positive = labels_diff == 0  # Mask for positive pairs
        mask_negative = labels_diff != 0  # Mask for negative pairs

        dist_ap = euclidean_distance(anchor[mask_positive],
                                     positive[mask_positive])  # Compute distance between anchor and positive
        dist_an = euclidean_distance(anchor[mask_negative],
                                     negative[mask_negative])  # Compute distance between anchor and negative

        ap = torch.clamp_min(-dist_ap + self.m, min=0.)
        an = torch.clamp_min(dist_an + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (dist_ap - delta_p) * self.gamma
        logit_n = an * (dist_an - delta_n) * self.gamma

        loss = torch.mean(self.soft_plus(torch.max(logit_n[mask_negative], torch.zeros_like(logit_p[mask_positive]))))

        return loss


class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, sp: torch.Tensor, sn: torch.Tensor) -> torch.Tensor:
        ap = torch.clamp_min(-sp.detach() + 1 + self.m, min=0.)
        an = torch.clamp_min(sn.detach() + self.m, min=0.)

        delta_p = 1 - self.m
        delta_n = self.m

        logit_p = -ap * (sp - delta_p) * self.gamma
        logit_n = an * (sn - delta_n) * self.gamma

        loss_p = self.soft_plus(torch.logsumexp(logit_p, dim=1))
        loss_n = self.soft_plus(torch.logsumexp(logit_n, dim=1))

        loss = (loss_p + loss_n) / 2.0

        return loss.mean()
    # def forward(self, sp: Tensor, sn: Tensor) -> Tensor:
    #     ap = torch.clamp_min(- sp.detach() + 1 + self.m, min=0.)
    #     an = torch.clamp_min(sn.detach() + self.m, min=0.)
    #
    #     delta_p = 1 - self.m
    #     delta_n = self.m
    #
    #     logit_p = - ap * (sp - delta_p) * self.gamma
    #     logit_n = an * (sn - delta_n) * self.gamma
    #
    #     loss = self.soft_plus(torch.logsumexp(logit_n, dim=0) + torch.logsumexp(logit_p, dim=0))
    #
    #     return loss.mean()


# Create random input tensors


# Instantiate the TripletLoss
# margin = 1
# circle_loss = CircleTripleLoss(margin, 256)
# # Compute the loss
# loss = circle_loss(feat, positive, negative)
#
# # # Perform backward pass to compute gradients
# loss.backward()
# print(loss)
#
# # Access the gradients of each tensor
# anchor_grad = feat.grad
# # positive_grad = positive.grad
# negative_grad = negative.grad

# # Reshape gradients for PCA input
# anchor_grad_reshaped = anchor_grad.view(anchor_grad.size(0), -1).detach().numpy()
# positive_grad_reshaped = positive_grad.view(positive_grad.size(0), -1).detach().numpy()
# negative_grad_reshaped = negative_grad.view(negative_grad.size(0), -1).detach().numpy()
#
# # Apply PCA for dimensionality reduction
# pca = PCA(n_components=2)
# anchor_pca = pca.fit_transform(anchor_grad_reshaped)
# positive_pca = pca.fit_transform(positive_grad_reshaped)
# negative_pca = pca.fit_transform(negative_grad_reshaped)
#
# # Visualize the gradients in the reduced space
# plt.scatter(anchor_pca[:, 0], anchor_pca[:, 1], label='Anchor')
# plt.scatter(positive_pca[:, 0], positive_pca[:, 1], label='Positive')
# plt.scatter(negative_pca[:, 0], negative_pca[:, 1], label='Negative')
# plt.legend()
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('Gradients Visualization using PCA')
# plt.show()
# Reshape gradients for t-SNE input
# anchor_grad_reshaped = anchor_grad.view(anchor_grad.size(0), -1).detach().numpy()
# # positive_grad_reshaped = positive_grad.view(positive_grad.size(0), -1).detach().numpy()
# negative_grad_reshaped = negative_grad.view(negative_grad.size(0), -1).detach().numpy()
#
# # Apply t-SNE for nonlinear dimensionality reduction
# perplexity = 3  # Adjust the perplexity value here
# tsne = TSNE(n_components=2, perplexity=perplexity)
# anchor_tsne = tsne.fit_transform(anchor_grad_reshaped)
# # positive_tsne = tsne.fit_transform(positive_grad_reshaped)
# negative_tsne = tsne.fit_transform(negative_grad_reshaped)
#
# # Visualize the gradients in the reduced space
# plt.scatter(anchor_tsne[:, 0], anchor_tsne[:, 1], label='Anchor')
# # plt.scatter(positive_tsne[:, 0], positive_tsne[:, 1], label='Positive')
# plt.scatter(negative_tsne[:, 0], negative_tsne[:, 1], label='Negative')
# plt.legend()
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.title('Gradients Visualization using t-SNE')
# plt.show()
if __name__ == '__main__':
    print()
