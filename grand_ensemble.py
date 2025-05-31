import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import timm
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

# === Transforms ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# === Load Dataset ===
full_dataset = datasets.ImageFolder('DataSet', transform=transform)
class_names = full_dataset.classes
print("Classes:", class_names)

# === Split into Train/Val ===
val_ratio = 0.2
val_size = int(len(full_dataset) * val_ratio)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# === DIET Model Definition ===
class DIET(nn.Module):
    def __init__(self):
        super(DIET, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)

# === Train and Evaluation Functions ===
def train_epoch(models, opts, loader):
    for m in models:
        m.train()

    total, correct, loss_sum = 0, 0, 0.0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        for opt in opts:
            opt.zero_grad()

        outputs = [m(images) for m in models]
        avg_output = sum(outputs) / len(models)
        loss = nn.CrossEntropyLoss()(avg_output, labels)
        loss.backward()

        for opt in opts:
            opt.step()

        total += labels.size(0)
        _, predicted = torch.max(avg_output, 1)
        correct += (predicted == labels).sum().item()
        loss_sum += loss.item() * images.size(0)

    return loss_sum / total, correct / total

def evaluate(models, loader):
    for m in models:
        m.eval()

    total, correct, loss_sum = 0, 0, 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = [m(images) for m in models]
            avg_output = sum(outputs) / len(models)
            loss = nn.CrossEntropyLoss()(avg_output, labels)

            total += labels.size(0)
            _, predicted = torch.max(avg_output, 1)
            correct += (predicted == labels).sum().item()
            loss_sum += loss.item() * images.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return loss_sum / total, correct / total, np.array(all_preds), np.array(all_labels)

# === Main Block ===
if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.freeze_support()

    # === Model Initialization ===
    effnet = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2).to(device)

    resnet = models.resnet18(pretrained=False)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet = resnet.to(device)

    diet = DIET().to(device)

    # === Loss & Optimizers ===
    criterion = nn.CrossEntropyLoss()
    opt_effnet = optim.Adam(effnet.parameters(), lr=1e-4)
    opt_resnet = optim.Adam(resnet.parameters(), lr=1e-4)
    opt_diet = optim.Adam(diet.parameters(), lr=1e-4)

    models = [effnet, resnet, diet]
    opts = [opt_effnet, opt_resnet, opt_diet]

    # === Training Loop ===
    num_epochs = 10
    train_accs, val_accs = [], []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        t_loss, t_acc = train_epoch(models, opts, train_loader)
        v_loss, v_acc, preds, labels = evaluate(models, val_loader)
        train_accs.append(t_acc)
        val_accs.append(v_acc)
        print(f"Train Acc: {t_acc:.4f}, Val Acc: {v_acc:.4f}")

    # === Save Models ===
    torch.save(effnet.state_dict(), "effnet_b0_final.pth")
    torch.save(resnet.state_dict(), "resnet18_final.pth")
    torch.save(diet.state_dict(), "diet_final.pth")

    # === Plot Learning Curves ===
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

    # === Confusion Matrix ===
    cm = confusion_matrix(labels, preds)
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], class_names)
    plt.yticks([0,1], class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # === Metrics Bar Chart ===
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)

    plt.bar(['Acc', 'Prec', 'Recall', 'F1'], [accuracy, precision, recall, f1], color='skyblue')
    plt.title("Evaluation Metrics")
    plt.ylim([0.5, 1])
    plt.grid(axis='y')
    plt.show()

    # === Prediction Distribution ===
    counts = np.bincount(preds, minlength=2)
    plt.bar(class_names, counts, color='orange')
    plt.title("Prediction Distribution")
    plt.show()

    # === ShuffleSplit Histogram ===
    ss = ShuffleSplit(n_splits=10, test_size=0.2)
    splits = [len(val) for _, val in ss.split(np.arange(len(full_dataset)))]
    plt.hist(splits, bins=5)
    plt.title("ShuffleSplit Validation Set Size Distribution")
    plt.xlabel("Val Set Size")
    plt.ylabel("Frequency")
    plt.show()

    # === Classification Report ===
    print("\nClassification Report:\n")
    print(classification_report(labels, preds, target_names=class_names))
