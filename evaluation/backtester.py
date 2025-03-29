import logging
logging.basicConfig(level=logging.INFO)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def backtest_from_csv(csv_path, close_column="Close", split_year=2023):
    """
    وظيفة: backtest_from_csv
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.set_index("Date")
    df = df.sort_index()

    train = df[df.index.year < split_year]
    test = df[df.index.year == split_year]

    if len(test) < 10:
        raise ValueError("بيانات الاختبار قليلة جدًا.")

    test["Predicted"] = test[close_column].shift(1)

    mae = mean_absolute_error(test[close_column][1:], test["Predicted"][1:])
    rmse = mean_squared_error(test[close_column][1:], test["Predicted"][1:], squared=False)
    direction = np.mean(np.sign(np.diff(test[close_column])) == np.sign(np.diff(test["Predicted"]))) * 100

    logging.info(f"MAE: {mae:.2f}")
    logging.info(f"RMSE: {rmse:.2f}")
    logging.info(f"دقة الاتجاه: {direction:.2f}%")

    plt.figure(figsize=(14,6))
    plt.plot(test.index, test[close_column], label="فعلي")
    plt.plot(test.index, test["Predicted"], label="توقع", linestyle="--")
    plt.title("Backtest توقع السعر")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return test[["Predicted", close_column]]