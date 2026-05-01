import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Load models (same as your main file)
xgb_model  = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

# Load dataset
df = pd.read_csv("data/processed/RELIANCE.NS_processed.csv")

FEATURES = [
    "Volume","Return","MA_10","MA_20","MA_50",
    "Volatility","RSI","MACD","Signal_Line",
    "Dist_MA_50","Lag_1","Lag_2"
]

# Assume target exists (IMPORTANT)
# If not, you must create it
df = df.dropna()

# Example: classification target
df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

X = df[FEATURES]
y_true = df["Target"]

# XGBoost prediction
X_scaled = xgb_scaler.transform(X)
y_pred = xgb_model.predict(X_scaled)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sell","Buy"])
disp.plot()

plt.title("Confusion Matrix (XGBoost)")
plt.savefig("confusion_matrix.png")
plt.show()