import pandas as pd
import numpy as np
import os

# 1. Configuration
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"
STOCKS = [
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "ITC.NS",
    "MARUTI.NS",
    "BHARTIARTL.NS"
]

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def preprocess_stock_file(ticker):
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found for {ticker}, skipping...")
        return

    # Load data
    # We use header=[0, 1] if yfinance saved with multi-index, else just 0
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # --- FIX FOR 'STR' AND 'STR' ERROR ---
    # 1. Flatten MultiIndex columns (Common with new tickers like TMPV)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 2. Force all numeric columns to float (Removes 'Ticker' text rows)
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 3. Drop any rows that became completely NaN during conversion
    df = df.dropna(subset=["Close"])

    # --- FEATURE ENGINEERING ---

    # Daily Returns
    df["Return"] = df["Close"].pct_change()

    # Moving Averages
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    # Volatility (Rolling Std Dev)
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # Lag Features (VERY IMPORTANT)
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)
    df["Lag_3"] = df["Close"].shift(3)

    # Target Variable: Predict Next Day's Close Price
    df["Target_Next_Close"] = df["Close"].shift(-1)

    # Trend (Classification Target)
    df["Trend"] = (df["Target_Next_Close"] > df["Close"]).astype(int)

    # Remove the first few rows (NaNs from MA) and the last row (NaN Target)
    df = df.dropna()

    # Save processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(output_path)
    print(f"✅ Successfully processed {ticker} ({len(df)} rows)")


# 2. Execution
print("--- Starting Data Preprocessing ---")
for stock in STOCKS:
    try:
        preprocess_stock_file(stock)
    except Exception as e:
        print(f"❌ Error processing {stock}: {e}")

print("\n--- Preprocessing Complete! Files saved in data/processed ---")