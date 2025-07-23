
import plotly.express as px

def get_figure(df):
    if df is None or df.empty:
        return {}

    df = df.copy()
    df['Fraud_Label'] = df['Class'].map({0: 'Non-Fraud', 1: 'Fraud'})

    fig = px.histogram(
        df,
        x='Amount',
        color='Fraud_Label',
        nbins=50,
        title='Transaction Amount Distribution by Fraud Type',
        labels={'Fraud_Label': 'Transaction Type'},
        barmode='overlay',
        template='plotly_white'
    )

    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    return fig
