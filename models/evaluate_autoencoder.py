
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from fraud_detection.models.autoencoder import Autoencoder
from fraud_detection.models.AutoencoderPreprocessor import AutoencoderPreprocessor
from fraud_detection.src.data_loader import DataLoader as CSVLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "results/autoencoder_model.pt"

csv_loader = CSVLoader()
df = csv_loader.load_data()

preprocessor = AutoencoderPreprocessor()
X_all, y_all = preprocessor.get_full_data_and_labels(df)

X_tensor = torch.tensor(X_all.values, dtype=torch.float32).to(DEVICE)

model = Autoencoder(input_dim=X_tensor.shape[1]).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    outputs = model(X_tensor)
    reconstruction_error = torch.mean((outputs - X_tensor) ** 2, dim=1).cpu().numpy()


results_df = pd.DataFrame({
    "reconstruction_error": reconstruction_error,
    "true_label": y_all
})
results_df.to_csv("results/reconstruction_errors.csv", index=False)
print("Saved reconstruction errors to results/reconstruction_errors.csv")

fpr, tpr, _ = roc_curve(y_all, reconstruction_error)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Autoencoder")
plt.legend()
plt.tight_layout()
plt.savefig("results/roc_curve.png")
print("Saved ROC curve to results/roc_curve.png")

precision, recall, _ = precision_recall_curve(y_all, reconstruction_error)
avg_precision = average_precision_score(y_all, reconstruction_error)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"PR Curve (AP = {avg_precision:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Autoencoder")
plt.legend()
plt.tight_layout()
plt.savefig("results/pr_curve.png")
print("Saved PR curve to results/pr_curve.png")
