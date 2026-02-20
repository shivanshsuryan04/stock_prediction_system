import pandas as pd
import joblib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Hides TensorFlow spam
from tensorflow.keras.models import load_model

print("\n🧠 LOADING ENSEMBLE AI (XGBoost + LSTM)...")
# Load Models and Scalers
xgb_model = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

features = ["Volume", "Return", "MA_10", "MA_20", "MA_50", "Volatility", 
            "RSI", "MACD", "Signal_Line", "Dist_MA_50", "Lag_1", "Lag_2"]

print("\n" + "="*70)
print("      AI FINTECH: DAILY MARKET PREDICTIONS (ENSEMBLE)")
print("="*70)
print(f"{'TICKER':<15} | {'XGBOOST':<10} | {'LSTM':<10} | {'FINAL SIGNAL'}")
print("-" * 70)

data_dir = "data/processed"
all_files = sorted([f for f in os.listdir(data_dir) if f.endswith("_processed.csv")])

for file in all_files:
    ticker = file.replace("_processed.csv", "")
    df = pd.read_csv(os.path.join(data_dir, file))
    
    # Safety check: LSTM needs at least 10 days of data to look back at
    if len(df) < 10:
        continue
        
    # ==========================================
    # 1. XGBOOST PREDICTION (Looks at today only)
    # ==========================================
    latest_row = df[features].iloc[-1:]
    X_xgb = xgb_scaler.transform(latest_row)
    xgb_pred = xgb_model.predict(X_xgb)[0] # Returns 0 or 1
    xgb_text = "BUY" if xgb_pred == 1 else "SELL"
    
    # ==========================================
    # 2. LSTM PREDICTION (Looks at the last 10 days)
    # ==========================================
    last_10_days = df[features].iloc[-10:]
    X_lstm = lstm_scaler.transform(last_10_days)
    X_seq = np.array([X_lstm]) # Reshapes into a 10-day "movie" sequence
    
    lstm_prob = lstm_model.predict(X_seq, verbose=0)[0][0]
    lstm_pred = 1 if lstm_prob > 0.5 else 0
    lstm_text = "BUY" if lstm_pred == 1 else "SELL"
    
    # ==========================================
    # 3. ENSEMBLE VOTING LOGIC
    # ==========================================
    if xgb_pred == 1 and lstm_pred == 1:
        signal = "🟢 STRONG BUY"
    elif xgb_pred == 0 and lstm_pred == 0:
        signal = "🔴 STRONG SELL"
    else:
        signal = "🟡 HOLD (Uncertain)"
        
    print(f"{ticker:<15} | {xgb_text:<10} | {lstm_text:<10} | {signal}")

print("="*70)