import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, real_dir, ai_dir, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Label 0 for real, 1 for AI
        for img_name in os.listdir(real_dir):
            self.images.append(os.path.join(real_dir, img_name))
            self.labels.append(0)

        for img_name in os.listdir(ai_dir):
            self.images.append(os.path.join(ai_dir, img_name))
            self.labels.append(1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

