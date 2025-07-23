import os
import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

RESULTS_DIR = "data/results"

def get_available_runs():
    return [
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("ae_loss_curve_") and f.endswith(".png")
    ]

def create_results_tab():
    available_runs = get_available_runs()
    if not available_runs:
        return html.Div("No evaluation results found.", className="text-danger")

    return dbc.Container([
        html.H3("📊 Model Evaluation Results", className="mt-3 text-primary"),
        html.Label("Select model run:", className="mt-3"),
        dcc.Dropdown(
            id="eval-run-dropdown",
            options=[{"label": r, "value": r} for r in available_runs],
            value=available_runs[-1],
            clearable=False
        ),
        html.Div(id="eval-metrics-output", className="mt-4"),
        html.Img(id="eval-loss-img", style={"width": "100%", "marginTop": "20px"}),
    ], fluid=True)
