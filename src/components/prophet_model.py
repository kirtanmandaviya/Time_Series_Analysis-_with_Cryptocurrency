import pandas as pd
import pickle
import sys
from prophet import Prophet
from src.exception.exception import CustomException


class CryptoProphetModel:
    def __init__(self):
        self.model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.2
        )

    def prepare_data(self, df):
        """
        Convert raw dataframe to Prophet format
        """
        if 'date' not in df.columns or 'price' not in df.columns:
            raise ValueError("DataFrame must contain date and price columns")
        prophet_df = df[['date', 'price']].dropna().copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        return prophet_df

    def train(self, prophet_df):
        """
        Train Prophet model
        """
        self.model.fit(prophet_df)

    def predict(self, periods=30):
        """
        Forecast future prices
        """
        future = self.model.make_future_dataframe(periods=periods)
        return self.model.predict(future)

    def save(self, path="prophet.pkl"):
        """
        Save trained model
        """
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    @staticmethod
    def load(path="prophet.pkl"):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model
        except Exception as e:
            raise CustomException(e, sys)
