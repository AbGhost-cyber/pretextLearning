import os
import random

import torch
from PIL import Image
from torch import nn
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from test_circle_2.circle_loss import CircleLoss, convert_label_to_similarity


class FaceDataset(Dataset):
    def __init__(self, folder_path, isTrain=True, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.isTrain = isTrain

        random.seed(2023)
        # Get a list of all subfolders in the main folder
        subfolders = sorted(os.listdir(folder_path))
        random.shuffle(subfolders)

        self.samples = []

        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            # Get a list of all files in the sub-folder
            image_files = sorted(os.listdir(subfolder_path))

            self.compute_samples(image_files, subfolder, self.samples)

        if self.isTrain:
            random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def compute_samples(self, files, label: str, sample_list):
        for file_name in files:
            image_path = os.path.join(self.folder_path, label, file_name)
            image = Image.open(image_path).convert("L")

            if self.transform is not None:
                image = self.transform(image)
            mLabel = label.replace('s', '')
            sample_list.append((image, torch.tensor(int(mLabel))))

    def __getitem__(self, index):
        sample = self.samples[index]
        return sample


transformation = transforms.Compose([transforms.Resize((100, 100)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomResizedCrop(100),
                                     transforms.ToTensor()])

# folder_dataset = datasets.ImageFolder(root="")
face_dataset_train = FaceDataset('/Users/mac/research books/signature_research/data/faces/training/',
                                 transform=transformation)
face_dataset_test = FaceDataset('/Users/mac/research books/signature_research/data/faces/testing/',
                                transform=transformation)


def get_loader(is_train: bool, batch_size: int) -> DataLoader:
    return DataLoader(
        dataset=face_dataset_train if is_train else face_dataset_test,
        batch_size=batch_size,
        shuffle=is_train,
    )


class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4),
            nn.BatchNorm2d(num_features=96),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(num_features=384),
            nn.ReLU()
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        # This function will be called for both images
        # Its output is used to determine the similarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        feature = nn.functional.normalize(output)
        return feature


model = SiameseNetwork()
train_loader = get_loader(is_train=True, batch_size=64)
# optimizer = torch.optim.Adam(lr=6e-5, params=model.parameters())
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
optimizer = torch.optim.RMSprop(model.parameters(), lr=6e-2, eps=1e-8, weight_decay=5e-4, momentum=0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
criterion = CircleLoss(m=0.7, gamma=128)
losses = []
accuracies = []
num_epochs = 200
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('Training', '-' * 20)
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, y) in enumerate(train_loader):

        optimizer.zero_grad()
        output = model(x1)
        loss = criterion(*convert_label_to_similarity(output, y))
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        if (batch_idx + 1) % 1 == 0 or batch_idx == len(train_loader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(train_loader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0
            # losses.append(loss.item())
    scheduler.step()
torch.save(model.state_dict(), "test_circle2.pt")
print("finished")

if __name__ == '__main__':
    print()
