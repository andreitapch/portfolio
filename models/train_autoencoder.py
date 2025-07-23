import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import mlflow.pytorch
from datetime import datetime

from fraud_detection.models.autoencoder import Autoencoder
from fraud_detection.models.AutoencoderPreprocessor import AutoencoderPreprocessor
from fraud_detection.src.data_loader import DataLoader as CSVLoader


mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_autoencoder")


timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
LATENT_DIM = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MODEL_PATH = f"fraud_detection/data/model_output/autoencoder_{timestamp}.pt"
LOSS_PLOT_PATH = f"fraud_detection/data/results/ae_loss_curve_{timestamp}.png"
os.makedirs("fraud_detection/data/model_output", exist_ok=True)
os.makedirs("fraud_detection/data/results", exist_ok=True)

print(f"Using device: {DEVICE}")

csv_loader = CSVLoader()
df = csv_loader.load_data()
preprocessor = AutoencoderPreprocessor()
train_df = preprocessor.get_train_data(df)

X_train = torch.tensor(train_df.values, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_train), batch_size=BATCH_SIZE, shuffle=True)

model = Autoencoder(input_dim=X_train.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

with mlflow.start_run():
    mlflow.log_params({
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "latent_dim": LATENT_DIM
    })

    print("Training Autoencoder...")
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            inputs = batch[0].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] - Loss: {avg_loss:.6f}")
        mlflow.log_metric("loss", avg_loss, step=epoch)

    torch.save(model.state_dict(), MODEL_PATH)
    mlflow.pytorch.log_model(model, artifact_path="model")
    print(f"Model saved to {MODEL_PATH}")

    plt.figure(figsize=(8, 4))
    plt.plot(losses, label="Training Loss")
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    mlflow.log_artifact(LOSS_PLOT_PATH)
    print(f"Loss curve saved to {LOSS_PLOT_PATH}")
