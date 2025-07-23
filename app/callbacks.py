from dash import callback, Input, Output, dcc, html
import plotly.express as px
import os
import pandas as pd
import logging

from .plots import time_plot, amount_plot
from .layout import create_overview_tab
from .evaluation_layout import create_evaluation_tab
from fraud_detection.models.evaluate_autoencoder import evaluate_autoencoder

df = None
amount_min = amount_max = amount_median = 0


def register_callbacks(dataframe, min_val, max_val, median_val):
    global df, amount_min, amount_max, amount_median
    df = dataframe
    amount_min = min_val
    amount_max = max_val
    amount_median = median_val


@callback(
    Output("tab-overview-content", "style"),
    Output("tab-time-content", "style"),
    Output("tab-amount-content", "style"),
    Output("tab-results-content", "style"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    return (
        {"display": "block"} if tab == "tab-overview" else {"display": "none"},
        {"display": "block"} if tab == "tab-time" else {"display": "none"},
        {"display": "block"} if tab == "tab-amount" else {"display": "none"},
        {"display": "block"} if tab == "tab-results" else {"display": "none"},
    )


@callback(
    Output("total-transactions", "children"),
    Output("percent-fraud", "children"),
    Output("avg-amount", "children"),
    Output("amount-distribution", "figure"),
    Input("amount-range", "value")
)
def update_overview_slider(amount_range):
    if df is None:
        return "N/A", "N/A", "N/A", {}

    filtered_df = df[
        (df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])
        ]

    total = len(filtered_df)
    frauds = filtered_df["Class"].sum()
    percent_fraud = (frauds / total) * 100 if total > 0 else 0
    avg_amount = filtered_df["Amount"].mean() if total > 0 else 0

    fig = px.histogram(
        filtered_df,
        x="Amount",
        color=filtered_df["Class"].map({0: "Non-Fraud", 1: "Fraud"}),
        nbins=50,
        title="Amount Distribution (Filtered)",
        labels={"color": "Transaction Type"},
        barmode="overlay"
    )

    return f"{total:,}", f"{percent_fraud:.2f}%", f"${avg_amount:,.2f}", fig


@callback(
    Output("eval-roc-graph", "figure"),
    Output("eval-pr-graph", "figure"),
    Output("eval-metrics-output", "children"),
    Input("eval-run-dropdown", "value")
)
def update_evaluation_tab(selected_timestamp):
    if not selected_timestamp:
        return {}, {}, "Please select a model run."

    root_path = os.getcwd()
    try:
        print(f"📊 Evaluating autoencoder model for timestamp: {selected_timestamp}")
        result = evaluate_autoencoder(selected_timestamp, root_path)

        roc_fig = result["roc_fig"]
        pr_fig = result["pr_fig"]
        metrics_text = f"ROC AUC: {result['roc_auc']:.4f} | PR AUC: {result['avg_precision']:.4f}"

        return roc_fig, pr_fig, metrics_text

    except Exception as e:
        logging.exception("❌ Failed to evaluate autoencoder model.")
        return {}, {}, f"Error evaluating model: {str(e)}"


@callback(
    Output("time-graph", "figure"),
    Output("amount-graph", "figure"),
    Input("tabs", "value")
)
def update_static_graphs(tab):
    if df is None:
        return {}, {}

    time_fig = time_plot.get_figure(df)
    amount_fig = amount_plot.get_figure(df)
    return time_fig, amount_fig
