from dash import callback, Input, Output, dcc, html
import plotly.express as px
import os
import pandas as pd

from .plots.results_tab import RESULTS_DIR
from .layout import create_overview_tab
from .plots import time_plot, amount_plot

df = None
amount_min = amount_max = amount_median = 0

def register_callbacks(dataframe, min_val, max_val, median_val):
    global df, amount_min, amount_max, amount_median
    df = dataframe
    amount_min = min_val
    amount_max = max_val
    amount_median = median_val

@callback(
    Output("tabs-content", "children"),
    Input("tabs", "value")
)
def render_tab_content(tab):
    if df is None:
        return html.Div("Data not loaded.")

    if tab == "tab-overview":
        return create_overview_tab(amount_min, amount_max, amount_median)
    elif tab == "tab-time":
        return dcc.Graph(figure=time_plot.get_figure(df))
    elif tab == "tab-amount":
        return dcc.Graph(figure=amount_plot.get_figure(df))
    return html.Div("Tab not found.")

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



def register_result_tab_callbacks(app):
    @app.callback(
        Output("eval-loss-img", "src"),
        Output("eval-metrics-output", "children"),
        Input("eval-run-dropdown", "value")
    )
    def update_eval_tab(selected_plot):
        if not selected_plot:
            return None, "No plot selected"

        # Image
        img_path = os.path.join(RESULTS_DIR, selected_plot)
        img_uri = f"/assets/{selected_plot}"

        metrics_text = f"Showing loss curve for: {selected_plot}"

        return img_path, metrics_text
