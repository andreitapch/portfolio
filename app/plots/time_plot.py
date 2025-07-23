
import plotly.express as px

def get_figure(df):
    if df is None or df.empty:
        return {}

    df = df.copy()
    df['Fraud_Label'] = df['Class'].map({0: 'Non-Fraud', 1: 'Fraud'})

    df['Time (Hours)'] = df['Time'] / 3600

    fig = px.histogram(
        df,
        x='Time (Hours)',
        color='Fraud_Label',
        nbins=50,
        title='📈 Transactions Over Time',
        labels={'Fraud_Label': 'Transaction Type'},
        barmode='overlay',
        template='plotly_white'
    )

    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    return fig
