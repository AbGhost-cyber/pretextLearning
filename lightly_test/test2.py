import torch
import torchvision
from lightly.loss import NTXentLoss
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


def knn_predict(feature, feature_bank, feature_labels, classes: int, knn_k: int, knn_t: float):
    """Helper method to run kNN predictions on features based on a feature bank
    Args:
        feature: Tensor of shape [N, D] consisting of N D-dimensional features
        feature_bank: Tensor of a database of features used for kNN
        feature_labels: Labels for the features in our feature_bank
        classes: Number of classes (e.g. 10 for CIFAR-10)
        knn_k: Number of k neighbors used for kNN
        knn_t:
    """
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels


# Define the training and validation steps
def training_epoch_end(dataloader_kNN, backbone):
    # Update feature bank at the end of each training epoch
    backbone.eval()
    feature_bank = []
    targets_bank = []
    with torch.no_grad():
        for data in dataloader_kNN:
            img, target, _ = data
            img = img.cuda()
            target = target.cuda()
            feature = backbone(img).squeeze()
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            targets_bank.append(target)
    feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
    targets_bank = torch.cat(targets_bank, dim=0).t().contiguous()
    backbone.train()

    # # Update the feature bank and targets bank
    # feature_bank = torch.cat((feature_bank, feature_bank_new), dim=1)
    # targets_bank = torch.cat((targets_bank, targets_bank_new), dim=1)

    return feature_bank, targets_bank


def validation_step(feature_bank, targets_bank, classes, knn_k, knn_t, backbone, batch):
    # Perform kNN predictions once we have a feature bank
    images, targets, _ = batch
    images = images.cuda()
    feature = backbone(images).squeeze()
    feature = F.normalize(feature, dim=1)
    pred_labels = knn_predict(feature, feature_bank, targets_bank, classes, knn_k, knn_t)
    num = images.size(0)
    top1 = (pred_labels[:, 0] == targets).float().sum().item()
    return num, top1


def validation_epoch_end(outputs, max_accuracy):
    if outputs:
        total_num = 0
        total_top1 = 0.
        for (num, top1) in outputs:
            total_num += num
            total_top1 += top1
        acc = total_top1 / total_num
        if acc > max_accuracy:
            max_accuracy = acc
        print('kNN_accuracy:', acc * 100.0)
        return max_accuracy


resnet = torchvision.models.resnet34()
backbone = nn.Sequential(*list(resnet.children())[:-1])
model = NNCLR(backbone).cuda()

normalize_dict = {'mean': [0.4620, 0.4620, 0.4620], 'std': [0.4650, 0.4650, 0.4650]}
transform = SimCLRTransform(input_size=224, normalize=normalize_dict)
memory_bank = NNMemoryBankModule(size=4096).cuda()
cedar_dataset = CedarDataset(folder_path="CEDAR/full_org", split_ratio=0.7,
                             train=True, transform=transform)

criterion = NTXentLoss(temperature=0.5).cuda()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, eps=1e-8, weight_decay=5e-4, momentum=0.9)
# scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
running_loss = 0
number_samples = 0
num_epochs = 200
log_interval = 15
classes = 55
max_accuracy = 0.0
knn_k = 200
knn_t = 0.1

DataLoader(cedar_dataset.train_samples, batch_size=64, num_workers=12, shuffle=False)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs))
    print('Training', '-' * 20)
    for batch_idx, batch in enumerate(sig_dataloader):
        x0, x1 = batch[0]
        x0, x1 = x0.cuda(), x1.cuda()
        optimizer.zero_grad()
        z0, p0 = model(x0)
        z1, p1 = model(x1)
        z0 = memory_bank(z0, update=False)  # for retrieval hence false
        z1 = memory_bank(z1, update=True)
        loss = 0.5 * (criterion(z0, p1) + criterion(z1, p0))
        loss.backward()
        optimizer.step()
        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(sig_dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx + 1, len(sig_dataloader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0

    # Update the feature bank and targets bank at the end of each epoch
    feature_bank, targets_bank = training_epoch_end(dataloader_kNN, backbone)

    # Run validation at the end of each epoch
    print('Validation', '-' * 20)
    outputs = []
    for batch_idx, batch in enumerate(validation_dataloader):
        output = validation_step(feature_bank, targets_bank, classes, knn_k, knn_t, backbone, batch)
        outputs.append(output)
    max_acc = validation_epoch_end(outputs, max_accuracy)
    print("max accuracy {.4f}".format(max_acc))

# Save the trained model
torch.save(model.state_dict(), "NNCLR_200_epochs_test.pt")


