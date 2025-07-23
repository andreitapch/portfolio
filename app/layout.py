from dash import dcc, html
import dash_bootstrap_components as dbc


def create_overview_tab(amount_min, amount_max, amount_median):
    return dbc.Container([
        dbc.Row(dbc.Col(html.H1("💳 Fraud Detection Dashboard", className="text-center mb-4 text-primary"))),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Total Transactions"),
                html.H2(id="total-transactions")
            ])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("% Fraud"),
                html.H2(id="percent-fraud")
            ])), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H5("Avg Amount"),
                html.H2(id="avg-amount")
            ])), width=4),
        ], className="mb-4"),
        dbc.Row(dbc.Col(
            dcc.RangeSlider(
                min=amount_min,
                max=amount_max,
                step=(amount_max - amount_min) / 100,
                value=[amount_min, amount_max],
                marks={
                    int(amount_min): f"${int(amount_min)}",
                    int(amount_median): f"${int(amount_median)}",
                    int(amount_max): f"${int(amount_max)}"
                },
                id="amount-range"
            )
        )),
        dbc.Row(dbc.Col(dcc.Graph(id="amount-distribution")))
    ], fluid=True)


def create_dashboard_layout(amount_min, amount_max, amount_median):
    return dbc.Container([
        dcc.Tabs(id="tabs", value="tab-overview", children=[
            dcc.Tab(label="💳 Overview", value="tab-overview"),
            dcc.Tab(label="📈 Time Trend", value="tab-time"),
            dcc.Tab(label="💰 Fraud vs Amount", value="tab-amount"),
            dcc.Tab(label="📉 Model Results", value="tab-results"),

        ]),
        html.Div(
            id="tabs-content",
            children=create_overview_tab(amount_min, amount_max, amount_median)
        )
    ], fluid=True)
