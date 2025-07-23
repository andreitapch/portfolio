import os
from dash import callback, Input, Output
from fraud_detection.models.evaluate_autoencoder import evaluate_autoencoder


@callback(
    Output("eval-roc-graph", "figure"),
    Output("eval-pr-graph", "figure"),
    Output("eval-metrics-output", "children"),
    Input("eval-run-dropdown", "value")
)
def update_eval_tab(timestamp):
    if not timestamp:
        return {}, {}, "No model selected."

    ROOT_PATH = os.getcwd()
    result = evaluate_autoencoder(timestamp, ROOT_PATH)

    return result["roc_fig"], result["pr_fig"], (
        f"ROC AUC: {result['roc_auc']:.4f} | Avg Precision: {result['avg_precision']:.4f}"
    )
