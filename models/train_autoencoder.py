import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import joblib
import mlflow.pytorch
from datetime import datetime
import matplotlib.pyplot as plt

from fraud_detection.models.autoencoder import Autoencoder
from fraud_detection.models.AutoencoderPreprocessor import AutoencoderPreprocessor
from fraud_detection.src.data_loader import DataLoader as CSVLoader

# MLflow setup
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud_autoencoder")

# Timestamp for saved files
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Hyperparameters
EPOCHS = 20
BATCH_SIZE = 512
LEARNING_RATE = 1e-3
LATENT_DIM = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_PATH = os.path.dirname(os.getcwd())

# Output paths
MODEL_PATH = f"{ROOT_PATH}/data/model_output/autoencoder_{timestamp}.pt"
LOSS_PLOT_PATH = f"{ROOT_PATH}/data/results/ae_loss_curve_{timestamp}.png"
SCALER_PATH = f"{ROOT_PATH}/data/model_output/scaler_{timestamp}.pkl"

# Ensure directories exist
os.makedirs(f"{ROOT_PATH}/data/model_output", exist_ok=True)
os.makedirs(f"{ROOT_PATH}/data/results", exist_ok=True)

print(f"Using device: {DEVICE}")

# Load and preprocess data
csv_loader = CSVLoader()
credit_card_df = csv_loader.load_data()

preprocessor = AutoencoderPreprocessor()
X_train, y_train = preprocessor.fit_transform(credit_card_df)

# Save fitted scaler
joblib.dump(preprocessor.scaler, SCALER_PATH)
print(f"Scaler saved to {SCALER_PATH}")

# Convert to PyTorch tensor
X_tensor = torch.tensor(X_train.values, dtype=torch.float32)
train_loader = DataLoader(TensorDataset(X_tensor), batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = Autoencoder(input_dim=X_tensor.shape[1]).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train with MLflow tracking
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

    # Save model and loss curve
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
