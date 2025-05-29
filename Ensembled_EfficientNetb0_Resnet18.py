import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import timm
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Data paths
data_dir = 'dataset'  # your dataset directory with train/val splits

# Image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)

# Models
# EfficientNetB0 from timm
effnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
# ResNet18 from torchvision
resnet = models.resnet18(pretrained=False)
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

effnet.to(device)
resnet.to(device)

# Loss and optimizers
criterion = nn.CrossEntropyLoss()
effnet_optimizer = optim.Adam(effnet.parameters(), lr=1e-4)
resnet_optimizer = optim.Adam(resnet.parameters(), lr=1e-4)

# Training function for one epoch
def train_one_epoch(effnet, resnet, loader, criterion, effnet_opt, resnet_opt):
    effnet.train()
    resnet.train()
    running_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc='Training', leave=False)
    
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        effnet_opt.zero_grad()
        resnet_opt.zero_grad()

        outputs_effnet = effnet(images)
        outputs_resnet = resnet(images)

        # Average logits before softmax
        outputs = (outputs_effnet + outputs_resnet) / 2.0

        loss = criterion(outputs, labels)
        loss.backward()

        effnet_opt.step()
        resnet_opt.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=running_loss/total, acc=correct/total)

    return running_loss / total, correct / total

# Validation function
def validate(effnet, resnet, loader, criterion):
    effnet.eval()
    resnet.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        loop = tqdm(loader, desc='Validation', leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            outputs_effnet = effnet(images)
            outputs_resnet = resnet(images)

            outputs = (outputs_effnet + outputs_resnet) / 2.0

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

            loop.set_postfix(loss=running_loss/total, acc=correct/total)

    acc = correct / total
    print("\nValidation Accuracy: {:.4f}".format(acc))
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

    return running_loss / total, acc

# Train loop
num_epochs = 10

best_acc = 0.0
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")

    train_loss, train_acc = train_one_epoch(effnet, resnet, train_loader, criterion, effnet_optimizer, resnet_optimizer)
    val_loss, val_acc = validate(effnet, resnet, val_loader, criterion)

    if val_acc > best_acc:
        best_acc = val_acc
        # Save both models' weights
        torch.save(effnet.state_dict(), "efficientnet_b0_ensemble.pth")
        torch.save(resnet.state_dict(), "resnet18_ensemble.pth")
        print("Saved best model weights!")

print("Training complete.")
