import torch
from torch import nn


class SigNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoding_layers = nn.Sequential(
            nn.Conv2d(1, 96, 11),  # size = [145,210]
            nn.Softplus(),
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),  # size = [72, 105]

            nn.Conv2d(96, 256, 5, padding=2, padding_mode='zeros'),  # size = [72, 105]
            nn.LocalResponseNorm(size=5, k=2, alpha=1e-4, beta=0.75),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1, padding_mode='zeros'),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(p=0.3),

            nn.Flatten(1, -1),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(17 * 17 * 256, 1024),
            nn.Softplus(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 32),
        )

    def forward_encoding(self, x):
        x_encoding = self.encoding_layers(x)
        return x_encoding

    def forward(self, x1, x2):
        # Forward pass through the encoder
        x1_encoded = self.forward_encoding(x1)
        x2_encoded = self.forward_encoding(x2)

        # Forward pass through the projection head
        projections1 = self.projection_head(x1_encoded)
        projections2 = self.projection_head(x2_encoded)

        return projections1, projections2

    def layer_summary(self, X_shape):
        X = torch.randn(*X_shape)
        for layer in self.features:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)


model = SigNet()
