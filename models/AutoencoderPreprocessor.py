from sklearn.preprocessing import StandardScaler
import pandas as pd

from sklearn.preprocessing import StandardScaler

class AutoencoderPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, credit_card_df):
        """Used during training."""
        self.scaler.fit(credit_card_df[['Amount', 'Time']])
        credit_card_df[['Amount', 'Time']] = self.scaler.transform(credit_card_df[['Amount', 'Time']])
        features = credit_card_df.drop(columns=['Class'])
        labels = credit_card_df['Class'].values
        return features, labels

    def transform_only(self, credit_card_df):
        """Used during evaluation."""
        credit_card_df[['Amount', 'Time']] = self.scaler.transform(credit_card_df[['Amount', 'Time']])
        features = credit_card_df.drop(columns=['Class'])
        labels = credit_card_df['Class'].values
        return features, labels

    def get_train_data(self, credit_card_df: pd.DataFrame) -> pd.DataFrame:
        clean_df = credit_card_df[credit_card_df['Class'] == 0].drop(columns=['Class']).reset_index(drop=True) # Only non-fraud samples
        return self.fit_transform(clean_df)

    def get_full_data_and_labels(self, credit_card_df: pd.DataFrame):
        labels = credit_card_df['Class'].values
        features = self.transform_only(credit_card_df.drop(columns=['Class']))
        return features, labels