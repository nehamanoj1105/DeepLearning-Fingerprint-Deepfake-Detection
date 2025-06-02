import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # For progress bar
import timm  # For loading DeiT models

# CONFIG
# Set dataset paths and training hyperparameters
train_dir = 'datasets'  # Dataset folder (should contain subfolders like 'ai' and 'real')
val_dir = 'datasets'    # Using the same folder here, assumes dataset is split internally

batch_size = 32
num_epochs = 10
learning_rate = 1e-4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Use GPU if available

# TRANSFORMS
# Preprocessing transformations applied to each image
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match model input size
    transforms.ToTensor(),  # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                         std=[0.229, 0.224, 0.225])
])

# DATASET & DATALOADER
# Load images from folders using ImageFolder (expects subfolders per class)
train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

# Create DataLoaders for batching and shuffling
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# MODELS
# Load pre-trained ResNet18 and modify the final layer for 2 classes
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 2)  # 2 output classes: 'ai' and 'real'

# Load pre-trained DeiT-small and modify final classification head
deit = timm.create_model('deit_small_patch16_224', pretrained=True)
deit.head = nn.Linear(deit.head.in_features, 2)

# Move both models to the GPU (or CPU)
resnet18 = resnet18.to(device)
deit = deit.to(device)

# Set both models to training mode
resnet18.train()
deit.train()

# LOSS & OPTIMIZER
# Use CrossEntropyLoss for classification
criterion = nn.CrossEntropyLoss()

# Combine parameters from both models into a single optimizer
optimizer = optim.Adam(list(resnet18.parameters()) + list(deit.parameters()), lr=learning_rate)

# TRAINING LOOP
for epoch in range(num_epochs):
    running_loss, correct_preds, total_preds = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)  # Progress bar

    for inputs, labels in loop:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        # Forward pass through both models
        out_resnet = resnet18(inputs)
        out_deit = deit(inputs)

        # Average the predictions (logits) for ensembling
        outputs = (out_resnet + out_deit) / 2

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate predictions and update counters
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        correct_preds += torch.sum(preds == labels.data)
        total_preds += labels.size(0)

        # Show current batch loss in progress bar
        loop.set_postfix(loss=loss.item())

    # Calculate epoch-level metrics
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct_preds.double() / len(train_dataset)

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

# VALIDATION LOOP
resnet18.eval()
deit.eval()

y_true = []  # Ground truth labels
y_pred = []  # Predicted labels

with torch.no_grad():  # No gradient updates in validation
    for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        out_resnet = resnet18(inputs)
        out_deit = deit(inputs)
        outputs = (out_resnet + out_deit) / 2  # Ensemble prediction

        _, preds = torch.max(outputs, 1)  # Get predicted class

        # Store predictions and true labels
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# Calculate validation metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
f1 = f1_score(y_true, y_pred, average='binary')

print("\n=== Validation Results ===")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# SAVE MODELS
# Save the state (weights) of both models into a .pth file
torch.save({
    'resnet18_state_dict': resnet18.state_dict(),
    'deit_state_dict': deit.state_dict(),
}, "ensemble_resnet_deit.pth")

print("\nModels saved as ensemble_resnet_deit.pth")
