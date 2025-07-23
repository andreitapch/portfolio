import os
import torch
import joblib
import pandas as pd
import plotly.graph_objs as go
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from fraud_detection.models.autoencoder import Autoencoder
from fraud_detection.models.AutoencoderPreprocessor import AutoencoderPreprocessor
from fraud_detection.src.data_loader import DataLoader as CSVLoader


def evaluate_autoencoder(timestamp: str, root_path: str):
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(root_path, "data", "model_output", f"autoencoder_{timestamp}.pt")
    scaler_path = os.path.join(root_path, "data", "model_output", f"scaler_{timestamp}.pkl")
    csv_path = os.path.join(root_path, "data", "results", f"reconstruction_errors_{timestamp}.csv")

    csv_loader = CSVLoader()
    credit_card_df = csv_loader.load_data()

    scaler = joblib.load(scaler_path)
    preprocessor = AutoencoderPreprocessor()
    preprocessor.scaler = scaler
    X_all, y_all = preprocessor.transform_only(credit_card_df)
    X_tensor = torch.tensor(X_all.values, dtype=torch.float32).to(DEVICE)

    model = Autoencoder(input_dim=X_tensor.shape[1], latent_dim=16).to(DEVICE)

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    with torch.no_grad():
        outputs = model(X_tensor)
        reconstruction_error = torch.mean((outputs - X_tensor) ** 2, dim=1).cpu().numpy()

    results_df = pd.DataFrame({
        "reconstruction_error": reconstruction_error,
        "true_label": y_all
    })
    results_df.to_csv(csv_path, index=False)

    # ROC
    fpr, tpr, _ = roc_curve(y_all, reconstruction_error)
    roc_auc = auc(fpr, tpr)

    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"ROC AUC = {roc_auc:.4f}"))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random'))

    roc_fig.update_layout(
        title="ROC Curve",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    # PR
    precision, recall, _ = precision_recall_curve(y_all, reconstruction_error)
    avg_precision = average_precision_score(y_all, reconstruction_error)

    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name=f"AP = {avg_precision:.4f}"))
    pr_fig.update_layout(
        title="Precision-Recall Curve",
        xaxis_title="Recall",
        yaxis_title="Precision"
    )

    return {
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "csv_path": csv_path,
        "roc_fig": roc_fig,
        "pr_fig": pr_fig
    }
