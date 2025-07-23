
import plotly.express as px

def get_figure(df):
    if df is None or df.empty:
        return {}

    df = df.copy()
    df['Fraud_Label'] = df['Class'].map({0: 'Non-Fraud', 1: 'Fraud'})

    # Time is in seconds from the first transaction — group into bins
    df['Time (Hours)'] = df['Time'] / 3600  # convert to hours for interpretability

    fig = px.histogram(
        df,
        x='Time (Hours)',
        color='Fraud_Label',
        nbins=50,
        title='📈 Transactions Over Time',
        labels={'Fraud_Label': 'Transaction Type'},
        barmode='overlay'
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    return fig
