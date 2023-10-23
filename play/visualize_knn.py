import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from torch import nn

# # Generate a toy dataset with two classes
# X, y = make_blobs(n_samples=100, centers=2, random_state=42)
#
# # Train the kNN Classifier
# k = 5
# knn_classifier = KNeighborsClassifier(n_neighbors=k)
# knn_classifier.fit(X, y)
#
# # Define the Grid
# x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
# grid_points = np.c_[xx.ravel(), yy.ravel()]
# print(grid_points.shape)
#
# # Predict the Class Labels
# Z = knn_classifier.predict(grid_points)
# Z = Z.reshape(xx.shape)
#
# # Plot the Decision Boundary
# plt.contourf(xx, yy, Z, alpha=0.8)
# plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title('Decision Boundary of kNN Classifier')
# plt.show()
# t1 = torch.tensor([1, 2])
# t2 = torch.tensor([4, 2])
# tensor_1d = torch.cat((t1, t2))
# print(tensor_1d)
# fc = nn.Sequential(
#     nn.Linear(512, 128),
#     nn.ReLU(),
#     nn.Linear(128, 64),
#     nn.ReLU(),
#     nn.Linear(64, 1),
#     nn.Sigmoid()
# )
# nn.BCELoss()
# nn.BCEWithLogitsLoss()
# print(fc(torch.randn((1, 512))))
import pandas as pd

# Sample data
# data = {
#     'Phone Number': ['123-456-7890', '555-abc-1234', '987-654-3210', "876|678|3469"],
#     'Age': [25, 30, 35, 22]
# }
#
# # Create DataFrame
# df = pd.DataFrame(data)
#
# # Remove non-numeric characters from phone numbers
# df["Phone Number"] = df["Phone Number"].str.replace('[^0-9]', '', regex=True)

# Display the updated DataFrame
# print(df)
sttr = "data/BHSig260/Bengali/078/B-S-2-G-16.tif"
split_str = sttr.split('/')
print(int(split_str[3]))
if __name__ == '__main__':
    print()
