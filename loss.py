import torch.nn as nn
import torch.nn.functional as F
import torch


class TripletLoss(nn.Module):

    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, patches, negative):
        distance_positive = F.pairwise_distance(anchor, patches)
        distance_negative = F.pairwise_distance(anchor, negative)
        loss = torch.max(distance_positive - distance_negative + self.margin, torch.tensor(0.0))
        return loss.mean()
