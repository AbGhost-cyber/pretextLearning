import torch
import torchvision
from torch import nn
import torchvision.models as models


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.network = models.resnet18()
        # exclude the last layer since we want to use it for feature extraction
        self.network.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.network = torch.nn.Sequential(*list(self.network.children())[:-1])
        self.projection_original_features = nn.Linear(2048, 128)
        self.connect_patches_feature = nn.Linear(1152, 128)

    def forward_once(self, x):
        return self.network(x)

    def return_image_feature(self, original):
        original_features = self.forward_once(original)
        original_features = original_features.view(-1, 2048)
        original_features = self.projection_original_features(original_features)
        return original_features

    def return_patches_feature(self, patches):
        patches_features = []
        for i, patch in enumerate(patches):
            patch_features = self.return_image_feature(patch)
            patches_features.append(patch_features)

        patches_features = torch.cat(patches_features, dim=1)

        patches_features = self.connect_patches_feature(patches_features)
        return patches_features

    def forward(self, positive, patches, negative):
        positive_features = self.return_image_feature(positive)
        patches_features = self.return_patches_feature(patches)
        negative_features = self.return_image_feature(negative)
        return positive_features, patches_features, negative_features