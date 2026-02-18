import pandas as pd
import os
import numpy as np
import joblib  # For saving models and scaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# ===============================
# 1. CONFIGURATION & DIRECTORIES
# ===============================
DATA_DIR = "data/processed"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)  # Creates 'models' folder if it doesn't exist

all_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_processed.csv")]

if not all_files:
    print("❌ No processed files found. Run preprocess_data.py first.")
    exit()

# Combine all stock data for general market training
df_list = []
for file in all_files:
    path = os.path.join(DATA_DIR, file)
    df = pd.read_csv(path)
    df_list.append(df)

data = pd.concat(df_list, ignore_index=True)
print(f"Total rows combined for training: {len(data)}")

# ===============================
# 2. FEATURE SELECTION (SYNCED)
# ===============================
features = [
    "Volume",
    "Return",
    "MA_10",
    "MA_20",
    "MA_50",
    "Volatility",
    "RSI",
    "MACD",
    "Signal_Line",
    "Dist_MA_50",
    "Lag_1",
    "Lag_2",
]

X = data[features]
y = data["Trend"]

# XGBoost requires labels [0, 1]. Map: -1 (Down) -> 0, 1 (Up) -> 1
y_mapped = y.map({-1: 0, 1: 1})

# ===============================
# 3. TIME-SERIES SPLIT & SCALING
# ===============================
# shuffle=False is critical for financial data to prevent look-ahead bias
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y_mapped, test_size=0.2, shuffle=False
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# --- SAVE THE SCALER ---
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("✅ Scaler saved to models/scaler.pkl")

# ===============================
# 4. TRAINING & TOURNAMENT
# ===============================
print("\n--- Training Classification Models ---")

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, eval_metric="logloss"
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name:20} Accuracy: {acc:.4f}")

    # --- SAVE EACH MODEL ---
    model_filename = f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, os.path.join(MODEL_DIR, model_filename))

print(f"✅ All models saved to the '{MODEL_DIR}' folder.")

# ===============================
# 5. FEATURE IMPORTANCE (XGBOOST)
# ===============================
importance = models["XGBoost"].feature_importances_
plt.figure(figsize=(10, 8))
plt.barh(features, importance, color="steelblue")
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Importance Score")
plt.tight_layout()

# --- SAVE THE CHART ---
plt.savefig("feature_importance_results.png")
print("✅ Feature importance chart saved as 'feature_importance_results.png'")

# ===============================
# 6. DAILY RECOMMENDATION ENGINE
# ===============================
print("\n" + "=" * 55)
print("     AI FINTECH SYSTEM: DAILY STOCK RECOMMENDATIONS")
print("=" * 55)
print(f"{'TICKER':<15} | {'CONFIDENCE':<12} | {'SIGNAL'}")
print("-" * 55)

# We use XGBoost for recommendations as it usually has the highest accuracy
best_model = models["XGBoost"]

for file in all_files:
    ticker = file.replace("_processed.csv", "")
    stock_path = os.path.join(DATA_DIR, file)
    stock_df = pd.read_csv(stock_path)

    latest_row = stock_df.iloc[-1:]
    latest_X_scaled = scaler.transform(latest_row[features])

    # prob[0] is Class 0 (Down), prob[1] is Class 1 (Up)
    prob = best_model.predict_proba(latest_X_scaled)[0]
    prob_down, prob_up = prob[0], prob[1]

    confidence = max(prob_up, prob_down)

    if prob_up > 0.60:
        signal = "STRONG BUY"
    elif prob_up > 0.53:
        signal = "BUY"
    elif prob_down > 0.60:
        signal = "STRONG SELL"
    elif prob_down > 0.53:
        signal = "SELL"
    else:
        signal = "HOLD"

    print(f"{ticker:<15} | {confidence:>10.2%} | {signal}")

print("=" * 55)
