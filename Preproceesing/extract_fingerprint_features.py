import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from tqdm import tqdm

real_path = "real"
ai_path = "ai"
save_csv_path = "fingerprint_features.csv"
image_size = (128, 128)
lbp_radius = 3
lbp_n_points = 8 * lbp_radius
gabor_ksize = 31

def extract_lbp(image, radius=3, n_points=24):
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_gabor(image):
    feats = []
    for theta in range(4):
        theta_rad = theta / 4. * np.pi
        kernel = cv2.getGaborKernel((gabor_ksize, gabor_ksize), 4.0, theta_rad, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        feats.append(filtered.mean())
        feats.append(filtered.var())
    return np.array(feats)

def extract_features(image):
    lbp_feat = extract_lbp(image, lbp_radius, lbp_n_points)
    gabor_feat = extract_gabor(image)
    return np.concatenate((lbp_feat, gabor_feat))

def load_images_from_folder(folder_path, label):
    features = []
    labels = []
    filenames = os.listdir(folder_path)
    for filename in tqdm(filenames, desc=f"Processing {os.path.basename(folder_path)}", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{percentage:3.0f}%]"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        img = cv2.equalizeHist(img)
        feats = extract_features(img)
        features.append(feats)
        labels.append(label)
    return features, labels

real_features, real_labels = load_images_from_folder(real_path, 0)
ai_features, ai_labels = load_images_from_folder(ai_path, 1)

X = np.vstack((real_features, ai_features))
y = np.array(real_labels + ai_labels)

df = pd.DataFrame(X)
df['label'] = y
df.to_csv(save_csv_path, index=False)

print(f"\nFeature extraction complete. Saved to {save_csv_path}")

