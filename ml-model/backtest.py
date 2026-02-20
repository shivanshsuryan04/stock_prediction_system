import pandas as pd
import joblib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hides unnecessary TensorFlow terminal warnings
from tensorflow.keras.models import load_model

print("\nLOADING AI MODELS (This takes a few seconds)...")
# Load XGBoost Brain
xgb_model = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")

# Load LSTM Brain
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

def create_sequences(X, time_steps=10):
    """Reshapes the data into 10-day movies for the LSTM"""
    X_s = []
    for i in range(len(X) - time_steps):
        X_s.append(X[i : i + time_steps])
    return np.array(X_s)

def backtest_both(ticker):
    df = pd.read_csv(f"data/processed/{ticker}_processed.csv")
    features = ["Volume", "Return", "MA_10", "MA_20", "MA_50", "Volatility", 
                "RSI", "MACD", "Signal_Line", "Dist_MA_50", "Lag_1", "Lag_2"]
    
    # ==========================
    # 1. MARKET PROFIT (Buy & Hold)
    # ==========================
    market_profit = ((1 + df['Return']).cumprod().iloc[-1] - 1) * 100
    
    # ==========================
    # 2. XGBOOST PROFIT
    # ==========================
    X_xgb = xgb_scaler.transform(df[features])
    xgb_preds = xgb_model.predict(X_xgb)
    xgb_signals = pd.Series(xgb_preds).map({0: -1, 1: 1}).values
    xgb_profit = ((1 + (xgb_signals * df['Return'])).cumprod().iloc[-1] - 1) * 100
    
    # ==========================
    # 3. LSTM PROFIT
    # ==========================
    X_lstm = lstm_scaler.transform(df[features])
    X_seq = create_sequences(X_lstm, time_steps=10)
    
    # predict() gives probabilities (e.g. 0.65). We convert > 0.5 to 1 (Up), else 0 (Down)
    lstm_prob = lstm_model.predict(X_seq, verbose=0)
    lstm_preds = (lstm_prob > 0.5).astype(int).flatten()
    lstm_signals = pd.Series(lstm_preds).map({0: -1, 1: 1}).values
    
    # Because LSTM needs 10 days to form its first sequence, we align it with returns starting from day 10
    actual_returns = df['Return'].iloc[10:].values
    lstm_profit = ((1 + (lstm_signals * actual_returns)).cumprod()[-1] - 1) * 100
    
    # ==========================
    # 4. PRINT RESULTS
    # ==========================
    print(f"📈 {ticker:15} | Market: {market_profit:>8.2f}% | XGB: {xgb_profit:>8.2f}% | LSTM: {lstm_profit:>8.2f}%")

print("\n" + "="*80)
print("      AI FINTECH: THE ULTIMATE SHOWDOWN (XGBOOST vs LSTM vs MARKET)")
print("="*80)

data_dir = "data/processed"
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_processed.csv")])

for file in all_files:
    ticker = file.replace("_processed.csv", "")
    backtest_both(ticker)

print("="*80)