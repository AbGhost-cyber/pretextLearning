import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torch
from dataset import JigsawDataset
from models import Network
from loss import TripletLoss


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.show()


folder_dataset = datasets.ImageFolder(root="/Users/mac/research books/signature_research/data/faces/training/")
# folder_dataset = datasets.ImageFolder(root=folder_root)
jigsawDataset = JigsawDataset(imageFolderDataset=folder_dataset)
network = Network()
loss_fn = TripletLoss()

optimizer = optim.Adam(network.parameters(), lr=0.00005)
# Load the training dataset
batch_size = 32
train_dataloader = DataLoader(jigsawDataset, shuffle=True, batch_size=batch_size)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
print(f"dataset size: {len(jigsawDataset)}")
print(f"batch per epoch {len(jigsawDataset) // batch_size}")
counter = []
loss_history = []
iteration_number = 0
epoch = 30
print_every = len(jigsawDataset) // batch_size
# Iterate through the epochs
for epoch in range(epoch):
    # siamese_net.train()
    # Iterate over batches
    for i, item in enumerate(train_dataloader, 0):
        positive_img = item['original']
        patches = item['patches']
        negative_image = item['negative']

#         # Zero the gradient
        optimizer.zero_grad()

        # Pass in the two images into the network and obtain two outputs
        positive_img_features, patches_features, negative_img_features = network(positive_img, patches, negative_image)

        # Pass the outputs of the networks and label into the loss function
        loss_contrastive = loss_fn(positive_img_features, patches_features, negative_img_features)

        # Calculate the backpropagation
        loss_contrastive.backward()

        # Optimize
        optimizer.step()
        # scheduler.step(loss_contrastive)

        # Every 10 batches print out the loss
        if i % print_every == print_every - 1:
            print(f"Epoch number {epoch}\n Current loss {loss_contrastive.item()}\n")
            iteration_number += 10

            counter.append(iteration_number)
            loss_history.append(loss_contrastive.item())
show_plot(counter, loss_history)
torch.save(network.state_dict(), "first.pt")

if __name__ == '__main__':
    print()
