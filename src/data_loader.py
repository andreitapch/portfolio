import os
import pandas as pd

class DataLoader:
    def __init__(self):
        self.data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
        self.data_file = os.path.join(self.data_dir, 'creditcard.csv')
        self.credit_card_df = None

    def load_data(self):
        if not os.path.exists(self.data_file):
            raise FileNotFoundError(f"Data file not found at {self.data_file}. Please make sure it exists.")

        print(f"Loading data from {self.data_file}...")
        self.credit_card_df = pd.read_csv(self.data_file)
        print(f"Data loaded: {self.credit_card_df.shape[0]} rows, {self.credit_card_df.shape[1]} columns.")
        return self.credit_card_df