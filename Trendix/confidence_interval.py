
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

class PredictionIntervalEstimator:
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.lower_model = GradientBoostingRegressor(loss="quantile", alpha=alpha / 2)
        self.upper_model = GradientBoostingRegressor(loss="quantile", alpha=1 - alpha / 2)
        self.median_model = GradientBoostingRegressor(loss="ls")

    def fit(self, X, y):
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        self.median_model.fit(X, y)

    def predict_with_interval(self, X):
        lower = self.lower_model.predict(X)
        upper = self.upper_model.predict(X)
        median = self.median_model.predict(X)
        return pd.DataFrame({
            "prediction": median,
            "lower_bound": lower,
            "upper_bound": upper
        })
