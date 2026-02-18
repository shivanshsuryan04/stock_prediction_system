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
    "BHARTIARTL.NS",
]

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def preprocess_stock_file(ticker):
    file_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(file_path):
        print(f"⚠️ File not found for {ticker}, skipping...")
        return 0

    # Load data
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Force all numeric columns to float
    numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            
    # Drop rows where Close is NaN
    df = df.dropna(subset=["Close"])

    # --- ENHANCED FEATURE ENGINEERING ---

    # 1. Basic Indicators
    df["Return"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["Volatility"] = df["Return"].rolling(window=10).std()

    # 2. RSI (Relative Strength Index) - 14 Days
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # 3. MACD (Moving Average Convergence Divergence)
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["Signal_Line"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # 4. Relative Price Distance (Is price high or low relative to its 50-day average?)
    df["Dist_MA_50"] = df["Close"] / df["MA_50"]

    # 5. Lag Features
    df["Lag_1"] = df["Close"].shift(1)
    df["Lag_2"] = df["Close"].shift(2)

    # --- UPDATED TARGETS ---
    df["Target_Next_Return"] = df["Return"].shift(-1)
    # Stronger trend definition (removes noise)
    df["Trend"] = 0
    df.loc[df["Target_Next_Return"] > 0.01, "Trend"] = 1   # Strong UP
    df.loc[df["Target_Next_Return"] < -0.01, "Trend"] = -1 # Strong DOWN

    # Remove weak/noise days
    df = df[df["Trend"] != 0]


    # Drop all NaNs created by indicators (e.g., first 50 rows due to MA_50)
    df = df.dropna()

    # Get row count for this specific stock
    row_count = len(df)

    # Save processed data
    output_path = os.path.join(PROCESSED_DATA_DIR, f"{ticker}_processed.csv")
    df.to_csv(output_path)
    print(f"✅ Enhanced processing complete for {ticker:15} | Rows: {row_count}")
    
    return row_count


# 2. Execution
print("--- Starting Enhanced Preprocessing ---")
total_processed_rows = 0

for stock in STOCKS:
    try:
        count = preprocess_stock_file(stock)
        total_processed_rows += count
    except Exception as e:
        print(f"❌ Error processing {stock}: {e}")

print("-" * 60)
print(f"TOTAL ROWS COMBINED ACROSS ALL STOCKS: {total_processed_rows}")
print("--- Preprocessing Complete! Files saved in data/processed ---")