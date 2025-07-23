from sklearn.preprocessing import StandardScaler
import pandas as pd

class AutoencoderPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, credit_card_df: pd.DataFrame) -> pd.DataFrame:
        credit_card_df = credit_card_df.copy()
        credit_card_df[['Amount', 'Time']] = self.scaler.fit_transform(credit_card_df[['Amount', 'Time']]) #Learn from data + transform
        return credit_card_df

    def transform(self, credit_card_df: pd.DataFrame) -> pd.DataFrame:
        credit_card_df = credit_card_df.copy()
        credit_card_df[['Amount', 'Time']] = self.scaler.transform(credit_card_df[['Amount', 'Time']]) #Ltransform
        return credit_card_df

    def get_train_data(self, credit_card_df: pd.DataFrame) -> pd.DataFrame:
        clean_df = credit_card_df[credit_card_df['Class'] == 0].drop(columns=['Class']).reset_index(drop=True) # Only non-fraud samples
        return self.fit_transform(clean_df)

    def get_full_data_and_labels(self, credit_card_df: pd.DataFrame):
        labels = credit_card_df['Class'].values
        features = self.transform(credit_card_df.drop(columns=['Class']))
        return features, labels