import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
# Define the transformation: convert to tensor and normalize
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))  # Mean and Std of CIFAR-10
])

# Download and load the training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=4)
# CIFAR-10 classes
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Get one batch of images and labels
dataiter = iter(trainloader)
images, labels = next(dataiter)

# Select one image and its label (e.g., the first image in the batch)
image = images[0]
label = labels[0]

# Undo normalization for visualization
inv_transform = transforms.Normalize(
    mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
    std=[1/0.2470, 1/0.2435, 1/0.2616]
)

# Apply the inverse transform to make the image visually correct
image = inv_transform(image)

# Convert from tensor to numpy array and permute dimensions to (H, W, C)
image = image.permute(1, 2, 0).numpy()

# Clip the values to [0, 1] for correct visualization
image = image.clip(0, 1)

# Plot the image
plt.figure(figsize=(4, 4))  # Adjust the figure size to 4x4 inches
plt.imshow(image, interpolation='nearest')  # Use nearest neighbor interpolation
plt.title(f'Label: {classes[label]}')
plt.axis('off')  # Turn off axis
plt.show()

