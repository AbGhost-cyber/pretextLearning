import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


# Define the triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, sn, sp):
        loss = torch.relu(sn - sp + self.margin)
        return loss


# Define the toy scenario
margin = 0.2
triplet_loss = TripletLoss(margin)

# Generate values for sn and sp
sn_values = torch.linspace(0, 2, 100)  # Generate 100 values from 0 to 2
sp_values = torch.linspace(0, 2, 100)

# Calculate the loss for different sn and sp values
loss_values = torch.zeros((100, 100))
for i, sn in enumerate(sn_values):
    for j, sp in enumerate(sp_values):
        loss = triplet_loss(sn, sp)
        loss_values[i, j] = loss.item()

# Plot the loss values
plt.imshow(loss_values, cmap='hot', origin='lower', extent=[0, 2, 0, 2])
plt.colorbar()
plt.xlabel('sn')
plt.ylabel('sp')
plt.title('Triplet Loss')

# Add decision boundaries
decision_boundary = np.linspace(0, 2, 100)
plt.plot(decision_boundary, decision_boundary, 'w--')
plt.plot(decision_boundary, decision_boundary + margin, 'w--')

# Add markers for statuses A and B
status_A = {'sn': 0.8, 'sp': 0.8}
status_B = {'sn': 1.2, 'sp': 1.2}
plt.scatter(status_A['sn'], status_A['sp'], color='r', label='Status A')
plt.scatter(status_B['sn'], status_B['sp'], color='b', label='Status B')

plt.legend()
plt.show()

if __name__ == '__main__':
    print()
