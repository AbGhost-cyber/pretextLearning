import PIL.Image
import numpy as np
from lightly.data import SimCLRCollateFunction
from lightly.transforms import SimCLRTransform
from matplotlib import pyplot as plt
from torchvision import transforms

# image_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     # transforms.Normalize(mean=std, std=std)
# ])

image1 = PIL.Image.open('/Users/mac/Downloads/data/CEDAR/full_org/original_20_4.png').convert('RGB')
transform = SimCLRTransform(input_size=224, random_gray_scale=1, gaussian_blur=0.)
transformed_image_a, transformed_image_b = transform(image1)

# Convert transformed images to NumPy arrays and transpose
image_a_np = np.array(transformed_image_a).transpose(1, 2, 0)
image_b_np = np.array(transformed_image_b).transpose(1, 2, 0)

# Plot the images side by side
fig, axes = plt.subplots(1, 2, figsize=(7, 5))

# Plot transformed image A
axes[0].imshow(image_a_np)
axes[0].axis('off')
axes[0].set_title('Transformed Image A')

# Plot transformed image B
axes[1].imshow(image_b_np)
axes[1].axis('off')
axes[1].set_title('Transformed Image B')

plt.tight_layout()
plt.show()



if __name__ == '__main__':
    print()
