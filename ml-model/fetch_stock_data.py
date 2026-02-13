import yfinance as yf
import os
import pandas as pd
from datetime import date

# 1. Configuration
today = date.today().strftime("%Y-%m-%d")
start_date = "2015-01-01"

stocks = [
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
    "SBIN.NS", "AXISBANK.NS", "WIPRO.NS", "HCLTECH.NS", "ITC.NS",
    "MARUTI.NS", "BHARTIARTL.NS"
]

os.makedirs("data/raw", exist_ok=True)

def apply_indicators(df):
    """Permanently adds features to the dataframe before saving."""
    # Ensure numeric types to prevent 'str' and 'str' errors
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Calculate Moving Averages with min_periods=1 to avoid 0-row wipeout
    df["MA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    df["MA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    
    # Daily Returns and Volatility
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(window=10, min_periods=1).std()
    
    # Target for ML: Next day's price
    df["Target_Next_Close"] = df["Close"].shift(-1)
    
    return df

print(f"--- Stock Data Sync: {today} ---\n")

for stock in stocks:
    print(f"Checking {stock}...")
    file_path = f"data/raw/{stock}.csv"

    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path)
        if "Date" in old_df.columns and not old_df.empty:
            last_recorded_date = old_df["Date"].max()
            fetch_start = pd.to_datetime(last_recorded_date) + pd.Timedelta(days=1)
        else:
            fetch_start = start_date
    else:
        old_df = pd.DataFrame()
        fetch_start = start_date

    df = yf.download(stock, start=fetch_start, end=today, auto_adjust=False, progress=False)

    if not df.empty:
        df.reset_index(inplace=True)
        # Flatten potential multi-index headers
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

        if not old_df.empty:
            initial_count = len(old_df)
            combined = pd.concat([old_df, df])
            combined = combined.drop_duplicates(subset=["Date"])
            
            # Re-apply indicators to include the new rows
            combined = apply_indicators(combined)
            
            new_rows = len(combined) - initial_count
            if new_rows > 0:
                print(f"✨ Success: {new_rows} new rows added and features updated.")
            else:
                print(f"» Status: Already up to date with {len(combined)} rows.")
        else:
            combined = apply_indicators(df)
            print(f"📂 Status: New file created with {len(combined)} rows.")

        combined.to_csv(file_path, index=False)
    else:
        combined = old_df
        print(f"» Status: Already up to date with {len(combined)} rows.")

    if not combined.empty:
        print(f"📊 Latest Data: {combined['Date'].max()}\n")

print("--- All stock updates finished! ---")