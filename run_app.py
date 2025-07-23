import dash
import dash_bootstrap_components as dbc

from app.layout import create_dashboard_layout
from app.callbacks import register_callbacks
from src.data_summary import load_data_and_summary

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP]
)

app.title = "Fraud Detection Dashboard"


credit_card_df, summary = load_data_and_summary()


app.layout = create_dashboard_layout(
    amount_min=summary['amount_min'],
    amount_max=summary['amount_max'],
    amount_median=summary['amount_median']
)


register_callbacks(
    credit_card_df,
    summary['amount_min'],
    summary['amount_max'],
    summary['amount_median']
)

if __name__ == '__main__':
    app.run(debug=True)
