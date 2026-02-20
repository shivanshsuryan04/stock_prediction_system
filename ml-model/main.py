from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import load_model

app = FastAPI(title="Fintech AI Ensemble API")

print("LOADING ENSEMBLE AI...")
xgb_model = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

features = ["Volume", "Return", "MA_10", "MA_20", "MA_50", "Volatility", 
            "RSI", "MACD", "Signal_Line", "Dist_MA_50", "Lag_1", "Lag_2"]

@app.get("/")
def home():
    return {"message": "Fintech ML API is running!"}

@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    file_path = f"data/processed/{ticker}_processed.csv"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stock data not found")
        
    df = pd.read_csv(file_path)
    if len(df) < 10:
        raise HTTPException(status_code=400, detail="Not enough data for LSTM")
        
    # XGBoost
    latest_row = df[features].iloc[-1:]
    X_xgb = xgb_scaler.transform(latest_row)
    xgb_pred = int(xgb_model.predict(X_xgb)[0])
    
    # LSTM
    last_10_days = df[features].iloc[-10:]
    X_lstm = lstm_scaler.transform(last_10_days)
    X_seq = np.array([X_lstm])
    lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
    lstm_pred = 1 if lstm_prob > 0.5 else 0
    
    # Ensemble Logic
    if xgb_pred == 1 and lstm_pred == 1:
        signal = "STRONG BUY"
    elif xgb_pred == 0 and lstm_pred == 0:
        signal = "STRONG SELL"
    else:
        signal = "HOLD"
        
    return {
        "ticker": ticker,
        "xgboost_prediction": "BUY" if xgb_pred == 1 else "SELL",
        "lstm_prediction": "BUY" if lstm_pred == 1 else "SELL",
        "lstm_confidence": round(lstm_prob, 2),
        "final_signal": signal
    }