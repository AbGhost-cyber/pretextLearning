import torch
import torch.nn as nn
import torch.optim as optim


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        # self.fc1 = nn.Sequential(
        #     nn.Linear(384, 1024),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(1024, 256),
        #     nn.ReLU(inplace=True),
        #
        #     nn.Linear(256, 128)
        # )

        self.last_layer = nn.Sequential(
            # nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Linear(768, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward_once(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        # output = self.fc1(output)
        return output

    def forward(self, x1, x2):
        print(x1.shape)
        # Forward pass for branch A
        embedding_a = self.forward_once(x1)
        print(embedding_a.shape)

        # Forward pass for branch B
        embedding_b = self.forward_once(x2)
        print(embedding_b.shape)

        # Concatenate the embeddings
        concatenated = torch.cat((embedding_a, embedding_b), dim=1)

        # Predict similarity using fully connected layers
        similarity = self.last_layer(concatenated)

        return similarity


def focal_loss(y_true, y_pred, gamma=2, alpha=2):
    pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    loss = -alpha * (1 - pt) ** gamma * torch.log(pt)
    loss = torch.mean(loss)
    return loss


# Create Siamese network model
model = ConvNet()

# Define the loss function and optimizer
criterion = focal_loss
optimizer = optim.Adam(model.parameters())

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Example similar and dissimilar image pairs (input_a and input_b) and labels (targets)
input_a = torch.randn((10, 1, 100, 100))
input_b = torch.randn((10, 1, 100, 100))

# # Forward pass
outputs = model(input_a, input_b)
targets = torch.randint(0, 2, size=outputs.shape)

# Compute the focal loss
loss = criterion(targets, outputs)

# Backward and optimize
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"loss: {loss}")

if __name__ == '__main__':
    print()
