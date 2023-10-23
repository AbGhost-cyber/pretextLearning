import torch
from byol_pytorch.byol_pytorch import MaybeSyncBatchnorm
from torch import nn
import torch.nn.functional as F


class BYOLSiamNet(nn.Module):
    def __init__(self, backbone,
                 projection_task="binary",
                 dim=512, projection_size=512,
                 sync_batch_norm=None,
                 hidden_size=4096):
        super(BYOLSiamNet, self).__init__()
        self.backbone = backbone
        self.dim = dim
        self.projection_size = projection_size
        self.sync_batch_norm = sync_batch_norm
        self.hidden_size = hidden_size
        self.projection_task = projection_task

        self.num_features = self.projection_size if self.projection_task == "metric" else 1

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.sim_sam_mlp = nn.Sequential(
            nn.Linear(self.dim, self.hidden_size, bias=False),
            MaybeSyncBatchnorm(self.sync_batch_norm)(self.hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            MaybeSyncBatchnorm(self.sync_batch_norm)(self.hidden_size),
            nn.ReLU(inplace=True),

            nn.Linear(self.hidden_size, self.projection_size, bias=False),
            MaybeSyncBatchnorm(self.sync_batch_norm)(self.projection_size, affine=False),
            nn.Linear(self.projection_size, self.num_features, bias=False)
        )

    def forward(self, image1, image2):
        z1 = self.backbone(image1).flatten(start_dim=1)
        z2 = self.backbone(image2).flatten(start_dim=1)

        if self.projection_task == "binary":
            z1_relu = F.relu(z1)
            z2_relu = F.relu(z2)
            diff = torch.subtract(z1_relu, z2_relu)
            diff = F.normalize(diff, dim=1, p=2)
            sim = self.fc(diff)
            return sim

        z1 = self.fc(z1)
        z2 = self.fc(z2)

        return z1, z2
