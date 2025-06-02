# Import necessary libraries
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from my_dataset import ImageDataset  # Custom dataset class for loading images

# Define image transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet mean
                         std=[0.229, 0.224, 0.225])    # and standard deviation
])

# Load the dataset using the custom ImageDataset class
# Assumes 'datasets/real' contains real images and 'datasets/ai' contains AI-generated images
dataset = ImageDataset('datasets/real', 'datasets/ai', transform=transform)

# Split the dataset into training (80%) and validation (20%) sets
train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

# Create DataLoader for batching and shuffling the data
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)  # Shuffle for training
val_loader = DataLoader(val_data, batch_size=8)  # No shuffle for validation

# Import model architecture, loss function, and optimizer
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# Load pre-trained ResNet-18 model
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

# Modify the final fully connected (fc) layer to output 2 classes
model.fc = nn.Linear(model.fc.in_features, 2)

# Define loss function (cross-entropy for classification)
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam with learning rate of 0.001)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Set number of training epochs
epochs = 10

# Use GPU if available, else fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Move model to the selected device
model = model.to(device)

# Training loop
for epoch in range(epochs):
    model.train()  # Set model to training mode
    running_loss, correct_preds, total_preds = 0.0, 0, 0  # Initialize metrics

    # Iterate over batches of training data
    for images, labels in train_loader:
        # Move inputs and labels to device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        outputs = model(images)  # Forward pass through the model
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Accumulate loss and compute accuracy
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)  # Get predicted class indices
        correct_preds += (preds == labels).sum().item()  # Count correct predictions
        total_preds += labels.size(0)  # Count total predictions

    # Calculate average loss and accuracy for the epoch
    train_loss = running_loss / len(train_loader)
    train_acc = correct_preds / total_preds

    # Print epoch results
    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
