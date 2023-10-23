import matplotlib.pyplot as plt

# Data points
# x = [1, 2, 3, 4, 5]
# y = [2, 4, 6, 8, 10]
# z = [1, 3, 2, 4, 5]
#
# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # Plot the data points
# ax.plot3D(x, y, z)
#
# # Set labels and title
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.set_title('3D Line Plot')
#
# # Show the plot
# plt.show()
import torch

# Create two input tensors
a = torch.tensor([[1, 2, 3],
                  [4, 5, 6]])
b = torch.tensor([[7, 8, 9],
                  [10, 11, 12]])

# Normalize the input tensors
a_normed = torch.nn.functional.normalize(a, dim=1)
b_normed = torch.nn.functional.normalize(b, dim=1)

# Compute the similarity matrix using einsum
similarity_matrix = torch.einsum("nd,md->nm", a_normed, b_normed)

# print(similarity_matrix)

if __name__ == '__main__':
    print()
    # print(int(label))
