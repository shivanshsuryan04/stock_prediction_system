import os, logging, asyncio
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yfinance as yf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from tensorflow.keras.models import load_model

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import subprocess

# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("neuraltrade")

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "Volume", "Return", "MA_10", "MA_20", "MA_50",
    "Volatility", "RSI", "MACD", "Signal_Line",
    "Dist_MA_50", "Lag_1", "Lag_2",
]

STOCKS = [
    "RELIANCE.NS", "TCS.NS",   "INFY.NS",      "HDFCBANK.NS",
    "ICICIBANK.NS","SBIN.NS",  "AXISBANK.NS",  "WIPRO.NS",
    "HCLTECH.NS",  "ITC.NS",   "MARUTI.NS",    "BHARTIARTL.NS",
]

# ─────────────────────────────────────────────────────────────
# Load Models (once at import time)
# ─────────────────────────────────────────────────────────────
logger.info("Loading ensemble AI models …")
xgb_model  = joblib.load("models/xgboost.pkl")
xgb_scaler = joblib.load("models/scaler.pkl")
lstm_model  = load_model("models/lstm_model.h5")
lstm_scaler = joblib.load("models/lstm_scaler.pkl")
logger.info("All models loaded ✅")

# ─────────────────────────────────────────────────────────────
# Nightly sync task (replaces daily_update.bat)
# ─────────────────────────────────────────────────────────────
def run_nightly_sync():
    """
    Runs Mon–Fri at 16:30 IST automatically.
    Fetches today's final candle and reprocesses all indicators.
    Equivalent to running daily_update.bat manually.
    """
    logger.info("═" * 55)
    logger.info("  NIGHTLY SYNC STARTING  –  %s", datetime.now().strftime("%d %b %Y %H:%M"))
    logger.info("═" * 55)
    try:
        logger.info("Step 1/2 – Fetching latest stock data …")
        subprocess.run(["python", "fetch_stock_data.py"], check=True)
        logger.info("Step 1/2 – Fetch complete ✅")

        logger.info("Step 2/2 – Reprocessing indicators …")
        subprocess.run(["python", "preprocess_data.py"], check=True)
        logger.info("Step 2/2 – Preprocess complete ✅")

        logger.info("Nightly sync finished successfully 🎉")
    except subprocess.CalledProcessError as e:
        logger.error("Nightly sync FAILED at step: %s", e)
    except Exception as e:
        logger.error("Unexpected error during nightly sync: %s", e)
    logger.info("═" * 55)


# ─────────────────────────────────────────────────────────────
# Scheduler setup
# ─────────────────────────────────────────────────────────────
scheduler = BackgroundScheduler(timezone="Asia/Kolkata")

# Runs every Mon–Fri at 16:30 IST (1 hour after NSE closes)
scheduler.add_job(
    run_nightly_sync,
    trigger=CronTrigger(
        day_of_week="mon-fri",
        hour=16,
        minute=30,
        timezone="Asia/Kolkata",
    ),
    id="nightly_sync",
    name="Nightly NSE Data Sync",
    replace_existing=True,
)

# ─────────────────────────────────────────────────────────────
# App lifespan – starts / stops scheduler cleanly
# ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    scheduler.start()
    next_run = scheduler.get_job("nightly_sync").next_run_time
    logger.info("Scheduler started ✅  Next nightly sync → %s", next_run)
    yield
    # ── shutdown ──
    scheduler.shutdown(wait=False)
    logger.info("Scheduler stopped.")


# ─────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="NeuralTrade AI Ensemble API",
    version="2.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Live indicator recalculation (in-memory, no disk write)
# ─────────────────────────────────────────────────────────────
def apply_live_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Return"]     = df["Close"].pct_change()
    df["MA_10"]      = df["Close"].rolling(10, min_periods=1).mean()
    df["MA_20"]      = df["Close"].rolling(20, min_periods=1).mean()
    df["MA_50"]      = df["Close"].rolling(50, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(10, min_periods=1).std()

    delta = df["Close"].diff()
    gain  = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    df["RSI"]         = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))
    exp1              = df["Close"].ewm(span=12, adjust=False).mean()
    exp2              = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Dist_MA_50"]  = df["Close"] / df["MA_50"].replace(0, np.nan)
    df["Lag_1"]       = df["Close"].shift(1)
    df["Lag_2"]       = df["Close"].shift(2)
    return df


def fetch_live_row(ticker: str) -> dict | None:
    """Pulls the latest 1-minute bar from Yahoo Finance."""
    try:
        data = yf.download(ticker, period="1d", interval="1m",
                           progress=False, auto_adjust=False)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        row = data.iloc[-1]
        return {
            "Date":   datetime.now().strftime("%Y-%m-%d"),
            "Open":   float(row.get("Open",   row.get("Close", 0))),
            "High":   float(row.get("High",   row.get("Close", 0))),
            "Low":    float(row.get("Low",    row.get("Close", 0))),
            "Close":  float(row.get("Close",  0)),
            "Volume": float(row.get("Volume", 0)),
        }
    except Exception as e:
        logger.warning("Live fetch failed for %s: %s", ticker, e)
        return None


def build_live_df(ticker: str) -> tuple[pd.DataFrame, bool]:
    """
    Returns (dataframe_with_live_row, is_live).
    Falls back to latest historical row if market is closed.
    """
    path = f"data/processed/{ticker}_processed.csv"
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No processed data for {ticker}. Run preprocess_data.py first.",
        )

    hist = pd.read_csv(path)
    live = fetch_live_row(ticker)

    if live:
        today = live["Date"]
        last  = str(hist.iloc[-1].get("Date", "")) if "Date" in hist.columns else ""
        if last != today:
            live_df = pd.DataFrame([live])
            shared  = hist.columns.intersection(live_df.columns)
            hist    = pd.concat([hist, live_df[shared]], ignore_index=True)
            hist    = apply_live_indicators(hist)
            logger.info("[%s] Live row appended – Close=%.2f", ticker, live["Close"])
            return hist, True

    logger.info("[%s] Using latest historical row (market closed or fetch failed)", ticker)
    return hist, False


# ─────────────────────────────────────────────────────────────
# Signal helpers
# ─────────────────────────────────────────────────────────────
def xgb_signal_label(prob_up: float, prob_down: float) -> str:
    if prob_up   > 0.60: return "Strong Buy"
    if prob_up   > 0.53: return "Buy"
    if prob_down > 0.60: return "Strong Sell"
    if prob_down > 0.53: return "Sell"
    return "Hold"


def lstm_signal_label(prob: float) -> str:
    if prob > 0.70:  return "Strong Buy"
    if prob > 0.55:  return "Buy"
    if prob < 0.30:  return "Strong Sell"
    if prob < 0.45:  return "Sell"
    return "Hold"


def ensemble_signal(xgb_pred: int, lstm_pred: int,
                    prob_up: float, lstm_prob: float) -> str:
    score = prob_up * 0.6 + lstm_prob * 0.4   # 0 = full bear, 1 = full bull
    if xgb_pred == 1 and lstm_pred == 1:
        return "Strong Buy" if score > 0.72 else "Buy"
    if xgb_pred == 0 and lstm_pred == 0:
        return "Strong Sell" if score < 0.28 else "Sell"
    if score > 0.58: return "Buy"
    if score < 0.42: return "Sell"
    return "Hold"


# ─────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    job       = scheduler.get_job("nightly_sync")
    next_sync = job.next_run_time.strftime("%d %b %Y %H:%M IST") if job else "unknown"
    return {
        "message":          "NeuralTrade AI Ensemble API v2.1 ✅",
        "scheduler_status": "running" if scheduler.running else "stopped",
        "next_nightly_sync": next_sync,
        "endpoints": ["/predict/{ticker}", "/predict/all", "/health", "/admin/sync-now"],
    }


@app.get("/health")
def health():
    job       = scheduler.get_job("nightly_sync")
    next_sync = job.next_run_time.isoformat() if job else None
    return {
        "status":            "ok",
        "models_loaded":     True,
        "scheduler_running": scheduler.running,
        "next_nightly_sync": next_sync,
        "timestamp":         datetime.now().isoformat(),
    }


@app.get("/predict/{ticker}")
def get_prediction(ticker: str):
    """
    Live-adjusted ensemble prediction for a single ticker.
    Injects real-time Yahoo Finance data before running inference.
    """
    ticker = ticker.upper()
    if not ticker.endswith(".NS"):
        ticker += ".NS"

    df, is_live = build_live_df(ticker)

    if len(df) < 11:
        raise HTTPException(status_code=400, detail="Need ≥ 11 rows for LSTM sequence.")

    # ── XGBoost ──────────────────────────────────
    latest      = df[FEATURES].iloc[-1:].fillna(0)
    X_xgb       = xgb_scaler.transform(latest)
    xgb_pred    = int(xgb_model.predict(X_xgb)[0])
    proba       = xgb_model.predict_proba(X_xgb)[0]
    prob_down, prob_up = float(proba[0]), float(proba[1])

    # ── LSTM ─────────────────────────────────────
    last_10     = df[FEATURES].iloc[-10:].fillna(0)
    X_seq       = np.array([lstm_scaler.transform(last_10)])
    lstm_prob   = float(lstm_model.predict(X_seq, verbose=0)[0][0])
    lstm_pred   = 1 if lstm_prob > 0.5 else 0

    return {
        "ticker":      ticker,
        "xgbSignal":   xgb_signal_label(prob_up, prob_down),
        "lstmSignal":  lstm_signal_label(lstm_prob),
        "lstmConf":    round(float(lstm_prob), 4),
        "finalSignal": ensemble_signal(xgb_pred, lstm_pred, prob_up, lstm_prob),
        # Extra metadata for the frontend (optional but helpful)
        "xgbProbUp":   round(float(prob_up), 4),
        "xgbProbDown": round(float(prob_down), 4),
        "fromCache":   not is_live,
        "cachedAt":    datetime.now().isoformat(),
    }


@app.get("/predict/all")
def get_all_predictions():
    """
    Batch prediction for all 12 NSE stocks.
    Matches the PredictionEntry[] type expected by usePredictions.ts.
    """
    results = []
    for ticker in STOCKS:
        try:
            results.append({"ticker": ticker, "data": get_prediction(ticker), "error": None})
        except HTTPException as e:
            results.append({"ticker": ticker, "data": None, "error": e.detail})
        except Exception as e:
            logger.error("Unexpected error for %s: %s", ticker, e)
            results.append({"ticker": ticker, "data": None, "error": str(e)})
    return {"data": results, "fetchedAt": datetime.now().isoformat()}


@app.post("/admin/sync-now")
def trigger_sync_now(background_tasks: BackgroundTasks):
    """
    Manually triggers the nightly sync immediately (useful for testing
    or if the server was off at 16:30 and you missed the auto-sync).
    """
    background_tasks.add_task(run_nightly_sync)
    return {"message": "Manual sync triggered in background. Check server logs for progress."}


@app.get("/scheduler/status")
def scheduler_status():
    """Returns the current status and next run time of the built-in scheduler."""
    job = scheduler.get_job("nightly_sync")
    return {
        "scheduler_running":  scheduler.running,
        "job_id":             job.id if job else None,
        "job_name":           job.name if job else None,
        "next_run_time":      job.next_run_time.isoformat() if job else None,
        "schedule":           "Mon–Fri at 16:30 IST",
    }


# ─────────────────────────────────────────────────────────────
# How to run
# ─────────────────────────────────────────────────────────────
# 1. Install APScheduler:   pip install apscheduler
# 2. Start the server:      uvicorn main:app --reload --port 8000
# 3. That's it — the scheduler wakes up automatically at 16:30 IST
#    every weekday and syncs your data without any manual steps.

#To activate virtual environment 
# venv\Scripts\activate

#To Deactivate virtual environment 
# deactivate