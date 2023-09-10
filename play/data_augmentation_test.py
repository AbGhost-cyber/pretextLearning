import PIL.Image
from torchvision import transforms
import matplotlib.pyplot as plt

# Define the transformations
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomResizedCrop(150, scale=(0.8, 1.2)),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
])

image = PIL.Image.open("/Users/mac/Downloads/cat.jpeg")

# Apply transformations to create correlated views
sample_1 = transform(image)
sample_2 = transform(image)

# Display the original image and the two augmented views
plt.imshow(image)
plt.show()
plt.imshow(sample_1.permute(1, 2, 0))
plt.show()
plt.imshow(sample_2.permute(1, 2, 0))
plt.show()

if __name__ == '__main__':
    print()
