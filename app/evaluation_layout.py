from dash import html, dcc
from .utils.model_utils import get_available_model_timestamps
import os

ROOT_PATH = os.getcwd()
MODEL_DIR = os.path.join(ROOT_PATH, "data", "model_output")
timestamps = get_available_model_timestamps(MODEL_DIR)

def create_evaluation_tab():
    timestamps = get_available_model_timestamps(MODEL_DIR)
    return html.Div([
        html.H4("Model Evaluation"),
        dcc.Dropdown(
            id="eval-run-dropdown",
            options=[{"label": ts.replace("_", " "), "value": ts} for ts in timestamps],
            placeholder="Select model run",
            style={"width": "50%"}
        ),
        html.Br(),
        html.H5("ROC Curve"),
        dcc.Graph(id="eval-roc-graph"),
        html.H5("Precision-Recall Curve"),
        dcc.Graph(id="eval-pr-graph"),
        html.Br(),
        html.Div(id="eval-metrics-output", style={"fontWeight": "bold", "fontSize": "18px"})
    ])
