from .data_loader import DataLoader


def load_data_and_summary():
    loader = DataLoader()
    credit_card_df = loader.load_data()

    amount_min = credit_card_df['Amount'].min()
    amount_max = credit_card_df['Amount'].max()
    amount_median = credit_card_df['Amount'].median()

    summary = {
        'amount_min': amount_min,
        'amount_max': amount_max,
        'amount_median': amount_median
    }
    return credit_card_df, summary