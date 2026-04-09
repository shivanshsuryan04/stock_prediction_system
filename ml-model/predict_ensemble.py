import argparse
import pandas as pd
import joblib
import numpy as np
import os
import yfinance as yf
from datetime import datetime
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

# ── Load models ──────────────────────────────
print("\n🧠 Loading Ensemble AI (XGBoost + LSTM)...")
xgb_model  = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model  = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

FEATURES = [
    "Volume", "Return", "MA_10", "MA_20", "MA_50",
    "Volatility", "RSI", "MACD", "Signal_Line",
    "Dist_MA_50", "Lag_1", "Lag_2",
]
STOCKS = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
    "ICICIBANK.NS", "SBIN.NS", "AXISBANK.NS", "WIPRO.NS",
    "HCLTECH.NS", "ITC.NS", "MARUTI.NS", "BHARTIARTL.NS",
]


def apply_live_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open","High","Low","Close","Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Return"]     = df["Close"].pct_change()
    df["MA_10"]      = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA_20"]      = df["Close"].rolling(window=20, min_periods=1).mean()
    df["MA_50"]      = df["Close"].rolling(window=50, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(window=10, min_periods=1).std()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(window=14, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Dist_MA_50"]  = df["Close"] / df["MA_50"].replace(0, np.nan)
    df["Lag_1"]       = df["Close"].shift(1)
    df["Lag_2"]       = df["Close"].shift(2)
    return df


def fetch_live_price(ticker: str) -> float | None:
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return float(data["Close"].iloc[-1])
    except Exception:
        return None


def predict_ticker(ticker: str) -> dict:
    file_path = f"data/processed/{ticker}_processed.csv"
    if not os.path.exists(file_path):
        return {"error": "processed CSV not found"}

    df = pd.read_csv(file_path)

    # Inject live row
    live_price = fetch_live_price(ticker)
    is_live = False
    if live_price:
        live_row  = pd.DataFrame([{"Close": live_price, "Volume": df["Volume"].iloc[-1]}])
        df = pd.concat([df, live_row], ignore_index=True)
        df = apply_live_indicators(df)
        is_live = True

    if len(df) < 11:
        return {"error": "not enough data"}

    # XGBoost
    latest = df[FEATURES].iloc[-1:].fillna(0)
    X_xgb  = xgb_scaler.transform(latest)
    xgb_pred  = int(xgb_model.predict(X_xgb)[0])
    xgb_proba = xgb_model.predict_proba(X_xgb)[0]
    prob_up, prob_down = float(xgb_proba[1]), float(xgb_proba[0])

    # LSTM
    last_10  = df[FEATURES].iloc[-10:].fillna(0)
    X_lstm   = lstm_scaler.transform(last_10)
    X_seq    = np.array([X_lstm])
    lstm_prob = float(lstm_model.predict(X_seq, verbose=0)[0][0])
    lstm_pred = 1 if lstm_prob > 0.5 else 0

    # 5-tier signals
    if prob_up > 0.60:        xgb_signal = "Strong Buy"
    elif prob_up > 0.53:      xgb_signal = "Buy"
    elif prob_down > 0.60:    xgb_signal = "Strong Sell"
    elif prob_down > 0.53:    xgb_signal = "Sell"
    else:                     xgb_signal = "Hold"

    if lstm_prob > 0.70:      lstm_signal = "Strong Buy"
    elif lstm_prob > 0.55:    lstm_signal = "Buy"
    elif lstm_prob < 0.30:    lstm_signal = "Strong Sell"
    elif lstm_prob < 0.45:    lstm_signal = "Sell"
    else:                     lstm_signal = "Hold"

    combined = prob_up * 0.6 + lstm_prob * 0.4
    if xgb_pred == 1 and lstm_pred == 1:
        final = "Strong Buy" if combined > 0.72 else "Buy"
    elif xgb_pred == 0 and lstm_pred == 0:
        final = "Strong Sell" if combined < 0.28 else "Sell"
    elif combined > 0.58: final = "Buy"
    elif combined < 0.42: final = "Sell"
    else:                 final = "Hold"

    return {
        "ticker":     ticker,
        "live_price": round(live_price, 2) if live_price else "N/A (market closed)",
        "is_live":    is_live,
        "xgb":        xgb_signal,
        "lstm":       lstm_signal,
        "lstm_conf":  f"{lstm_prob:.1%}",
        "final":      final,
    }


# ── CLI ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str, default=None)
args = parser.parse_args()

run_tickers = [args.ticker.upper()] if args.ticker else STOCKS

SIGNAL_ICON = {
    "Strong Buy": "🟢🟢", "Buy": "🟢", "Hold": "🟡",
    "Sell": "🔴", "Strong Sell": "🔴🔴",
}

print(f"\n{'='*75}")
print(f"   NEURALTRADE AI  |  Live Ensemble Predictions  |  {datetime.now().strftime('%d %b %Y %H:%M')}")
print(f"{'='*75}")
print(f"{'TICKER':<17} {'PRICE':>10} {'XGBOOST':<14} {'LSTM (CONF)':<18} {'FINAL SIGNAL'}")
print("-"*75)

for t in run_tickers:
    r = predict_ticker(t)
    if "error" in r:
        print(f"{t:<17} {'ERROR: '+r['error']}")
        continue
    live_tag = "⚡LIVE" if r["is_live"] else "📂CACHE"
    xgb_icon  = SIGNAL_ICON.get(r["xgb"],  "")
    fin_icon  = SIGNAL_ICON.get(r["final"], "")
    print(
        f"{t:<17} {str(r['live_price']):>10}  "
        f"{r['xgb']:<14} {r['lstm']+' ('+r['lstm_conf']+')' :<18} "
        f"{fin_icon} {r['final']}  [{live_tag}]"
    )

print("="*75)
print("💡 Signals: Strong Buy/Sell = both models agree | Buy/Sell = weighted edge | Hold = uncertain")
print("⚠️  FOR RESEARCH PURPOSES ONLY. NOT FINANCIAL ADVICE.\n")