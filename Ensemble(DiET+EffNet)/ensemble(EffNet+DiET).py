import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Load Predictions ===
diet_probs = np.load('diet_preds.npy')  # shape: (507, 2)
diet_labels = np.load('diet_labels.npy')  # shape: (507,)

effnet_probs = np.load('effnet_preds.npy')  # shape: (507, 1) or (507,) if squeezed

# === Prepare EffNet Predictions ===
if effnet_probs.ndim == 1:
    effnet_probs = effnet_probs.reshape(-1, 1)  # shape: (507, 1)

# Convert to 2-class probability: [prob_real, prob_ai]
effnet_probs = np.concatenate([1 - effnet_probs, effnet_probs], axis=1)  # shape: (507, 2)

# === Soft Voting Ensemble ===
ensemble_probs = (diet_probs + effnet_probs) / 2
ensemble_preds = np.argmax(ensemble_probs, axis=1)

# === Evaluation ===
print("Soft Voting Ensemble Results:")

print(confusion_matrix(diet_labels, ensemble_preds))
print(classification_report(diet_labels, ensemble_preds, target_names=["Real", "AI"]))
print("Accuracy:", accuracy_score(diet_labels, ensemble_preds))
