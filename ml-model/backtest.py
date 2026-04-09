import pandas as pd
import joblib
import numpy as np
import os
import argparse

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

print("\nLoading AI models for backtesting (this takes a few seconds)...")
xgb_model  = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model  = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")

FEATURES = [
    "Volume", "Return", "MA_10", "MA_20", "MA_50",
    "Volatility", "RSI", "MACD", "Signal_Line",
    "Dist_MA_50", "Lag_1", "Lag_2",
]


def create_sequences(X, time_steps=10):
    return np.array([X[i:i+time_steps] for i in range(len(X)-time_steps)])


def compute_drawdown(equity: np.ndarray) -> float:
    """Max drawdown as a percentage."""
    peak = np.maximum.accumulate(equity)
    dd   = (equity - peak) / peak
    return float(dd.min() * 100)


def compute_sharpe(returns: np.ndarray, risk_free: float = 0.065/252) -> float:
    """Annualised Sharpe ratio (risk-free ≈ RBI repo rate / 252)."""
    excess = returns - risk_free
    if excess.std() == 0:
        return 0.0
    return float((excess.mean() / excess.std()) * np.sqrt(252))


def backtest_ticker(ticker: str) -> dict:
    df = pd.read_csv(f"data/processed/{ticker}_processed.csv")
    if len(df) < 20:
        return {"error": "not enough data"}

    returns = df["Return"].fillna(0).values

    # ── Market (Buy & Hold) ─────────────────────────────────────
    market_equity  = (1 + pd.Series(returns)).cumprod().values
    market_profit  = (market_equity[-1] - 1) * 100
    market_sharpe  = compute_sharpe(returns)
    market_drawdown = compute_drawdown(market_equity)

    # ── XGBoost ─────────────────────────────────────────────────
    X_xgb      = xgb_scaler.transform(df[FEATURES].fillna(0))
    xgb_preds  = xgb_model.predict(X_xgb)
    xgb_signals = np.where(xgb_preds == 1, 1, -1)          # 1=Long, -1=Short
    xgb_ret    = xgb_signals * returns
    xgb_equity = (1 + pd.Series(xgb_ret)).cumprod().values
    xgb_profit = (xgb_equity[-1] - 1) * 100
    xgb_sharpe  = compute_sharpe(xgb_ret)
    xgb_drawdown = compute_drawdown(xgb_equity)
    xgb_accuracy = float((xgb_preds == df["Trend"].map({-1:0,1:1}).values).mean() * 100)

    # ── LSTM ────────────────────────────────────────────────────
    X_lstm     = lstm_scaler.transform(df[FEATURES].fillna(0))
    X_seq      = create_sequences(X_lstm, time_steps=10)
    lstm_probs  = lstm_model.predict(X_seq, verbose=0).flatten()
    lstm_preds  = (lstm_probs > 0.5).astype(int)
    lstm_signals = np.where(lstm_preds == 1, 1, -1)
    actual_ret  = returns[10:]                               # align with sequences
    lstm_ret    = lstm_signals * actual_ret
    lstm_equity = (1 + pd.Series(lstm_ret)).cumprod().values
    lstm_profit = (lstm_equity[-1] - 1) * 100
    lstm_sharpe  = compute_sharpe(lstm_ret)
    lstm_drawdown = compute_drawdown(lstm_equity)
    lstm_accuracy = float((lstm_preds == df["Trend"].map({-1:0,1:1}).values[10:]).mean() * 100)

    # ── Ensemble ────────────────────────────────────────────────
    # Only trade when both agree (conservative)
    ens_signals = []
    for x, l in zip(xgb_signals[10:], lstm_signals):
        if x == 1 and l == 1:   ens_signals.append(1)
        elif x == -1 and l == -1: ens_signals.append(-1)
        else:                      ens_signals.append(0)   # sit out on disagreement
    ens_signals = np.array(ens_signals)
    ens_ret     = ens_signals * actual_ret
    ens_equity  = (1 + pd.Series(ens_ret)).cumprod().values
    ens_profit  = (ens_equity[-1] - 1) * 100
    ens_sharpe   = compute_sharpe(ens_ret)
    ens_drawdown = compute_drawdown(ens_equity)

    return {
        "ticker": ticker,
        "market":  {"profit": market_profit,  "sharpe": market_sharpe,  "drawdown": market_drawdown},
        "xgboost": {"profit": xgb_profit,     "sharpe": xgb_sharpe,     "drawdown": xgb_drawdown,   "accuracy": xgb_accuracy},
        "lstm":    {"profit": lstm_profit,     "sharpe": lstm_sharpe,    "drawdown": lstm_drawdown,   "accuracy": lstm_accuracy},
        "ensemble":{"profit": ens_profit,     "sharpe": ens_sharpe,     "drawdown": ens_drawdown},
    }


# ── CLI ──────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ticker", type=str, default=None)
args = parser.parse_args()

data_dir = "data/processed"
if args.ticker:
    all_tickers = [args.ticker.upper()]
else:
    all_tickers = sorted([
        f.replace("_processed.csv","")
        for f in os.listdir(data_dir) if f.endswith("_processed.csv")
    ])

print(f"\n{'='*100}")
print(f"   NEURALTRADE AI  |  FULL BACKTEST  |  XGBoost vs LSTM vs Ensemble vs Market")
print(f"{'='*100}")
print(f"{'TICKER':<17} {'MARKET%':>9} {'XGB%':>9} {'LSTM%':>9} {'ENS%':>9}  {'XGB SHP':>8} {'LSTM SHP':>8}  {'XGB ACC':>8} {'LSTM ACC':>8}")
print("-"*100)

summaries = []
for t in all_tickers:
    r = backtest_ticker(t)
    if "error" in r:
        print(f"{t:<17} Error: {r['error']}")
        continue

    def f(v): return f"{v:>+9.2f}"
    def s(v): return f"{v:>8.2f}"

    best = max(r["xgboost"]["profit"], r["lstm"]["profit"], r["ensemble"]["profit"])
    win  = "🏆" if r["ensemble"]["profit"] == best else "  "

    print(
        f"{t:<17} {f(r['market']['profit'])} {f(r['xgboost']['profit'])} "
        f"{f(r['lstm']['profit'])} {f(r['ensemble']['profit'])}  "
        f"{s(r['xgboost']['sharpe'])} {s(r['lstm']['sharpe'])}  "
        f"{r['xgboost']['accuracy']:>7.1f}% {r['lstm']['accuracy']:>7.1f}%  {win}"
    )
    summaries.append(r)

if summaries:
    print("="*100)
    avg_mkt = np.mean([s["market"]["profit"]   for s in summaries])
    avg_xgb = np.mean([s["xgboost"]["profit"]  for s in summaries])
    avg_lst = np.mean([s["lstm"]["profit"]     for s in summaries])
    avg_ens = np.mean([s["ensemble"]["profit"] for s in summaries])
    avg_xacc = np.mean([s["xgboost"]["accuracy"] for s in summaries])
    avg_lacc = np.mean([s["lstm"]["accuracy"]    for s in summaries])
    print(
        f"{'AVERAGE':<17} {avg_mkt:>+9.2f}% {avg_xgb:>+9.2f}% {avg_lst:>+9.2f}% {avg_ens:>+9.2f}%"
        f"{'':>20} {avg_xacc:>7.1f}% {avg_lacc:>7.1f}%"
    )
    print("="*100)

print("\n⚠️  Backtest does not account for transaction costs, slippage, or taxes.")
print("⚠️  Past performance is not indicative of future results.\n")